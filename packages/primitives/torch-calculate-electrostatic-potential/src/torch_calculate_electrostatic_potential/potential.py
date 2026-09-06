"""Core differentiable electrostatic-potential math, shared by the 2D and 3D APIs.

Based on the D-dimensional isotropic Gaussian parameterization in Peng et al., 1996:

    V_D(r) = C * sum_i a_i * (4*pi)^(D/2) * w_i^(D/2)
             * exp(-4*pi^2 * w_i * |r|_D^2)

The stored Peng coefficients already parameterize elastic electron scattering
factors, not X-ray form factors:

    f_e(s) = sum_i a_i * exp(-b_i * s^2),  s = sin(theta) / wavelength

Consequently, they must not undergo the X-ray-to-electron Mott-Bethe conversion
``f_e(s) = 0.023934 * (Z - f_X(s)) / s²`` again. To produce electrostatic
potential in volts, however, the electron scattering factor must still be
converted to a voltage-normalized Fourier potential. For Fourier spatial
frequency ``g = 2s``:

    V_tilde(g) = C * f_e(g / 2)
    C = 2*pi*hbar^2 / (m_e*e) = h^2 / (2*pi*m_e*e)
      = 47.877647... V Angstrom^2

Each normalized inverse-Fourier Gaussian integrates to its coefficient ``a_i``,
so ``integral(V_D) = C * sum(a_i)``. The 3D inverse transform therefore has
units V, while its analytic projection over one Angstrom-valued spatial axis
has units V Angstrom. See Kirkland, *Advanced Computing in Electron Microscopy*,
section 5.2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import einops
import torch

if TYPE_CHECKING:
    from .grid import GridConfig

# Electron-scattering-factor (Å) to Fourier-potential (V Å³) normalization.
# Computed from CODATA 2022 electron mass and exact SI values for h and e.
PENG_SCATTERING_TO_POTENTIAL = 47.877647240509745  # V Å²


def evaluate_gaussian_sum(
    diff: torch.Tensor,  # (..., N, K, D) voxel - atom displacement, D = 2 or 3
    a: torch.Tensor,  # (..., N, 5) Peng 'a' parameters
    b: torch.Tensor,  # (..., N, 5) Peng 'b' parameters
    bfactor: torch.Tensor,  # (..., N) atomic B-factors
    voxel_size: torch.Tensor,  # (D,)
    per_voxel_averaging: bool,
) -> torch.Tensor:  # (..., N, K)
    """Evaluate the D-dimensional isotropic Gaussian potential sum at each voxel.

    Parameters
    ----------
    diff : torch.Tensor
        Atom displacements from each voxel, shape (..., N, K, D) where N is number of
        atoms, K is flattened voxels in the grid, and D is the spatial dimensions.
    a : torch.Tensor
        Peng 'a' parameters, shape (..., N, 5).
    b : torch.Tensor
        Peng 'b' parameters, shape (..., N, 5).
    bfactor : torch.Tensor
        Atomic B-factors, shape (..., N).
    voxel_size : torch.Tensor
        Size of each voxel, in Angstroms, shape (D,).
    per_voxel_averaging : bool
        Whether to average the potential over each voxel by evaluating at corners and
        averaging (True) or to evaluate at the voxel center (False).

    Returns
    -------
    torch.Tensor
        Per-atom potential at each voxel, shape (..., N, K). Units are V when
        ``D == 3`` and V Angstrom when ``D == 2``.
    """
    D = diff.shape[-1]
    gaussian_width = 1.0 / (b + bfactor.unsqueeze(-1))
    a_expanded = einops.rearrange(a, "... n c -> ... n 1 c")
    gw_expanded = einops.rearrange(gaussian_width, "... n c -> ... n 1 c")

    if per_voxel_averaging:
        gamma = (4 * torch.pi**2 * gaussian_width).sqrt()
        half_voxel = voxel_size / 2

        # Broadcasting tensors for computation
        gamma_expanded = einops.rearrange(gamma, "... n c -> ... n 1 1 c")
        half_voxel_expanded = einops.rearrange(half_voxel, "... d -> ... 1 1 d 1")
        diff_expanded = einops.rearrange(diff, "... n k d -> ... n k d 1")

        erf_diff = torch.erf(
            (diff_expanded + half_voxel_expanded) * gamma_expanded
        ) - torch.erf((diff_expanded - half_voxel_expanded) * gamma_expanded)
        spatial_integral = torch.prod(erf_diff, dim=-2)
        numerator = (spatial_integral * a_expanded).sum(dim=-1)

        result: torch.Tensor = (
            PENG_SCATTERING_TO_POTENTIAL * numerator / (2**D * voxel_size.prod())
        )
    else:
        squared_distance = einops.reduce(diff**2, "... n k d -> ... n k 1", "sum")
        prefactor = a_expanded * (4 * torch.pi) ** (D / 2) * gw_expanded ** (D / 2)
        exponent = torch.exp(-4 * torch.pi**2 * squared_distance * gw_expanded)
        result = PENG_SCATTERING_TO_POTENTIAL * (prefactor * exponent).sum(dim=-1)

    return result


def _calculate_scattering_potential(
    atom_pos: torch.Tensor,  # (..., N, D)
    atom_bfactors: torch.Tensor,  # (..., N)
    atom_params_a: torch.Tensor,  # (..., N, 5)
    atom_params_b: torch.Tensor,  # (..., N, 5)
    grid_config: GridConfig,
    atom_occupancies: torch.Tensor | None,  # (..., N) or None
    per_voxel_averaging: bool,
    batch_size: int,
) -> torch.Tensor:  # (..., *grid_config.grid_shape)
    """Shared dense implementation behind the 2D/3D public entry points."""
    if atom_pos.shape[-1] != grid_config.ndim:
        raise ValueError(
            f"atom_pos last dim ({atom_pos.shape[-1]}) must match "
            f"grid_config.ndim ({grid_config.ndim})"
        )

    device, dtype = grid_config.device, grid_config.dtype
    num_atoms = atom_pos.shape[-2]

    batch_shapes = [
        atom_pos.shape[:-2],
        atom_bfactors.shape[:-1],
        atom_params_a.shape[:-2],
        atom_params_b.shape[:-2],
    ]
    if atom_occupancies is not None:
        batch_shapes.append(atom_occupancies.shape[:-1])
    batch_shape = torch.broadcast_shapes(*batch_shapes)  # type: ignore[no-untyped-call]
    batch_total = 1
    for size in batch_shape:
        batch_total *= size

    # Flatten and expand atom data to match the batch shape.
    # Supports broadcasting of atom data across arbitrary batch dimensions (e.g. atoms
    # in different positions from rotated copes of the same structure).
    pos_flat = (
        atom_pos.to(device=device, dtype=dtype)
        .expand(*batch_shape, num_atoms, grid_config.ndim)
        .reshape(batch_total, num_atoms, grid_config.ndim)
    )
    bfactor_flat = (
        atom_bfactors.to(device=device, dtype=dtype)
        .expand(*batch_shape, num_atoms)
        .reshape(batch_total, num_atoms)
    )
    a_flat = (
        atom_params_a.to(device=device, dtype=dtype)
        .expand(*batch_shape, num_atoms, 5)
        .reshape(batch_total, num_atoms, 5)
    )
    b_flat = (
        atom_params_b.to(device=device, dtype=dtype)
        .expand(*batch_shape, num_atoms, 5)
        .reshape(batch_total, num_atoms, 5)
    )
    occupancy_flat = (
        atom_occupancies.to(device=device, dtype=dtype)
        .expand(*batch_shape, num_atoms)
        .reshape(batch_total, num_atoms)
        if atom_occupancies is not None
        else None
    )

    grid_flat_size = grid_config.grid_flat_size
    volume = torch.zeros(batch_total * grid_flat_size, dtype=dtype, device=device)
    batch_offsets = (
        torch.arange(batch_total, device=device, dtype=torch.int64) * grid_flat_size
    )

    for start in range(0, num_atoms, batch_size):
        end = min(start + batch_size, num_atoms)
        pos_chunk = pos_flat[:, start:end, :]  # (batch_total, n, D)

        with torch.no_grad():
            flat_indices, voxel_coords = grid_config.get_atom_stencil_voxels(pos_chunk)
            flat_indices = flat_indices + batch_offsets[:, None, None]
        voxel_coords = voxel_coords.detach()

        # (batch_total, n, K, D); grad flows to pos_chunk
        diff = voxel_coords - pos_chunk.unsqueeze(-2)
        values = evaluate_gaussian_sum(
            diff,
            a_flat[:, start:end, :],
            b_flat[:, start:end, :],
            bfactor_flat[:, start:end],
            grid_config.voxel_size,
            per_voxel_averaging,
        )  # (batch_total, n, K)

        if occupancy_flat is not None:
            values = values * occupancy_flat[:, start:end].unsqueeze(-1)

        volume.scatter_add_(
            0, flat_indices.reshape(-1), values.reshape(-1).to(volume.dtype)
        )

    return volume.reshape(*batch_shape, *grid_config.grid_shape.tolist())


def calculate_scattering_potential_3d(
    atom_pos_zyx: torch.Tensor,  # (..., N, 3)
    atom_bfactors: torch.Tensor,  # (..., N)
    atom_params_a: torch.Tensor,  # (..., N, 5)
    atom_params_b: torch.Tensor,  # (..., N, 5)
    grid_config: GridConfig,
    *,
    atom_occupancies: torch.Tensor | None = None,
    per_voxel_averaging: bool = True,
    batch_size: int = 4096,
) -> torch.Tensor:  # (..., *grid_config.grid_shape)
    """Compute a differentiable 3D electrostatic-potential volume in volts.

    Parameters
    ----------
    atom_pos_zyx : torch.Tensor
        Atom positions in Angstroms, shape (..., N, 3).
    atom_bfactors : torch.Tensor
        Atomic B-factors, shape (..., N).
    atom_params_a, atom_params_b : torch.Tensor
        Peng 1996 scattering parameters, each shape (..., N, 5). See
        `torch_calculate_electrostatic_potential.utils.peng_model.get_peng_scattering_parameters`.
    grid_config : GridConfig
        3D grid configuration (`grid_config.ndim == 3`).
    atom_occupancies : torch.Tensor or None
        Optional per-atom occupancy weights, shape (..., N).
    per_voxel_averaging : bool
        If True, average the potential over each voxel; if False, sample at
        the voxel center.
    batch_size : int
        Number of atoms processed per chunk (performance/memory tuning only).

    Returns
    -------
    torch.Tensor
        Potential volume(s) in volts, shape (..., *grid_config.grid_shape).
    """
    if grid_config.ndim != 3:
        raise ValueError(
            f"calculate_scattering_potential_3d requires a 3D grid_config, "
            f"got ndim={grid_config.ndim}"
        )
    return _calculate_scattering_potential(
        atom_pos_zyx,
        atom_bfactors,
        atom_params_a,
        atom_params_b,
        grid_config,
        atom_occupancies,
        per_voxel_averaging,
        batch_size,
    )


def calculate_scattering_potential_2d(
    atom_pos_yx: torch.Tensor,  # (..., N, 2)
    atom_bfactors: torch.Tensor,  # (..., N)
    atom_params_a: torch.Tensor,  # (..., N, 5)
    atom_params_b: torch.Tensor,  # (..., N, 5)
    grid_config: GridConfig,
    *,
    atom_occupancies: torch.Tensor | None = None,
    per_voxel_averaging: bool = True,
    batch_size: int = 4096,
) -> torch.Tensor:  # (..., *grid_config.grid_shape)
    """Compute a differentiable 2D projected electrostatic potential.

    Parameters
    ----------
    atom_pos_yx : torch.Tensor
        In-plane atom positions in Angstroms, shape (..., N, 2).
    atom_bfactors : torch.Tensor
        Atomic B-factors, shape (..., N).
    atom_params_a, atom_params_b : torch.Tensor
        Peng 1996 scattering parameters, each shape (..., N, 5). See
        `torch_calculate_electrostatic_potential.utils.peng_model.get_peng_scattering_parameters`.
    grid_config : GridConfig
        2D grid configuration (`grid_config.ndim == 2`).
    atom_occupancies : torch.Tensor or None
        Optional per-atom occupancy weights, shape (..., N).
    per_voxel_averaging : bool
        If True, average the potential over each pixel; if False, sample at
        the pixel center.
    batch_size : int
        Number of atoms processed per chunk (performance/memory tuning only).

    Returns
    -------
    torch.Tensor
        Potential image(s) in V Angstrom, shape (..., *grid_config.grid_shape).
    """
    if grid_config.ndim != 2:
        raise ValueError(
            "calculate_scattering_potential_2d requires a 2D grid_config, "
            f"got ndim={grid_config.ndim}"
        )
    return _calculate_scattering_potential(
        atom_pos_yx,
        atom_bfactors,
        atom_params_a,
        atom_params_b,
        grid_config,
        atom_occupancies,
        per_voxel_averaging,
        batch_size,
    )
