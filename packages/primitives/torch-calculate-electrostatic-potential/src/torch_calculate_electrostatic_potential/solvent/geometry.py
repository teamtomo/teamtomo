"""Distance-to-surface geometry for continuum solvent models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .vdw import VDW_RADII_A, vdw_radii_for_atomic_numbers

if TYPE_CHECKING:
    from ..grid import GridConfig

# Sentinel for voxels never updated by an atom neighborhood.
_DIST_INIT = 1.0e6


def voxel_centers_zyx(grid_config: GridConfig) -> torch.Tensor:
    """Return voxel-center coordinates for a 3D ``GridConfig``.

    Parameters
    ----------
    grid_config : GridConfig
        Must be three-dimensional.

    Returns
    -------
    torch.Tensor
        Shape ``(D, H, W, 3)`` in Angstroms (ZYX).
    """
    if grid_config.ndim != 3:
        raise ValueError(
            f"solvent geometry requires a 3D grid, got ndim={grid_config.ndim}"
        )
    shape = tuple(int(s) for s in grid_config.grid_shape.tolist())
    axes = [
        grid_config.left_bottom_point[axis]
        + torch.arange(shape[axis], device=grid_config.device, dtype=grid_config.dtype)
        * grid_config.voxel_size[axis]
        for axis in range(3)
    ]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((zz, yy, xx), dim=-1)


def distance_to_surface(
    positions_zyx: torch.Tensor,
    atomic_numbers: torch.Tensor,
    grid_config: GridConfig,
    *,
    r_asymptote: float = 7.5,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimum surface distance and nearest atomic number on a 3D grid.

    Surface distance is ``|x - atom| - VdW(Z)``. Voxels never visited by any
    atom neighborhood keep a large positive distance (treated as bulk solvent).
    Atoms with unknown VdW radii are skipped.

    Enclosed cavities are not flood-filled; voids can receive solvent density.

    Parameters
    ----------
    positions_zyx : torch.Tensor
        Atom positions in Angstroms, shape ``(n_atoms, 3)``.
    atomic_numbers : torch.Tensor
        Atomic numbers, shape ``(n_atoms,)``.
    grid_config : GridConfig
        Three-dimensional output grid.
    r_asymptote : float
        Neighborhood radius beyond the VdW surface (Angstroms) used when
        updating each atom's local brick.
    batch_size : int
        Reserved for future chunked vectorization.

    Returns
    -------
    dist_map : torch.Tensor
        Minimum surface distance in Angstroms, shape ``(D, H, W)``.
    nearest_atomic_numbers : torch.Tensor
        Atomic number of the nearest surface atom, shape ``(D, H, W)``, int64.
        Unvisited voxels are ``0``.
    """
    if positions_zyx.ndim != 2 or positions_zyx.shape[-1] != 3:
        raise ValueError("positions_zyx must have shape (n_atoms, 3)")
    if atomic_numbers.ndim != 1 or atomic_numbers.shape[0] != positions_zyx.shape[0]:
        raise ValueError("atomic_numbers must have shape (n_atoms,)")
    if grid_config.ndim != 3:
        raise ValueError(
            f"solvent geometry requires a 3D grid, got ndim={grid_config.ndim}"
        )
    if r_asymptote <= 0:
        raise ValueError("r_asymptote must be positive")

    device = grid_config.device
    dtype = grid_config.dtype
    positions = positions_zyx.to(device=device, dtype=dtype)
    z_numbers = atomic_numbers.to(device=device, dtype=torch.int64)
    radii = vdw_radii_for_atomic_numbers(z_numbers, device=device, dtype=dtype)

    shape = tuple(int(s) for s in grid_config.grid_shape.tolist())
    dist_map = torch.full(shape, _DIST_INIT, device=device, dtype=dtype)
    nearest_z = torch.zeros(shape, device=device, dtype=torch.int64)

    voxel_size = grid_config.voxel_size
    left = grid_config.left_bottom_point
    max_vdw = max(r for r in VDW_RADII_A.values() if r > 0)
    search_radius = r_asymptote + max_vdw

    n_atoms = positions.shape[0]
    for i in range(n_atoms):
        radius = radii[i]
        if radius.item() <= 0:
            continue
        pos = positions[i]
        half = search_radius
        lo = torch.floor((pos - half - left) / voxel_size).to(torch.int64)
        hi = torch.ceil((pos + half - left) / voxel_size).to(torch.int64)
        lo = torch.maximum(lo, torch.zeros(3, dtype=torch.int64, device=device))
        hi = torch.minimum(hi, grid_config.grid_shape)
        if torch.any(lo >= hi):
            continue

        z0, y0, x0 = (int(lo[0]), int(lo[1]), int(lo[2]))
        z1, y1, x1 = (int(hi[0]), int(hi[1]), int(hi[2]))
        zz = left[0] + torch.arange(z0, z1, device=device, dtype=dtype) * voxel_size[0]
        yy = left[1] + torch.arange(y0, y1, device=device, dtype=dtype) * voxel_size[1]
        xx = left[2] + torch.arange(x0, x1, device=device, dtype=dtype) * voxel_size[2]
        grid_z, grid_y, grid_x = torch.meshgrid(zz, yy, xx, indexing="ij")
        curr_r = (
            torch.sqrt(
                (grid_z - pos[0]) ** 2
                + (grid_y - pos[1]) ** 2
                + (grid_x - pos[2]) ** 2
            )
            - radius
        )
        # Only update within the asymptote shell used by Shang–Sigworth.
        in_shell = curr_r < r_asymptote
        slice_dist = dist_map[z0:z1, y0:y1, x0:x1]
        closer = in_shell & (curr_r < slice_dist)
        slice_dist = torch.where(closer, curr_r, slice_dist)
        dist_map[z0:z1, y0:y1, x0:x1] = slice_dist
        nearest_slice = nearest_z[z0:z1, y0:y1, x0:x1]
        nearest_z[z0:z1, y0:y1, x0:x1] = torch.where(
            closer, z_numbers[i].expand_as(nearest_slice), nearest_slice
        )

    # batch_size is reserved for future chunked vectorization
    _ = batch_size
    return dist_map, nearest_z
