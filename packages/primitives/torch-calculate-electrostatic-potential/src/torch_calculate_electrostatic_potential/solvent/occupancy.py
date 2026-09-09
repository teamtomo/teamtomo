"""Binary VdW + probe solvent occupancy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .geometry import distance_to_surface

if TYPE_CHECKING:
    from torch_structure_manipulation import AtomicStructure

    from ..grid import GridConfig


def vdw_probe_occupancy(
    dist_map: torch.Tensor,
    *,
    probe_radius: float = 1.4,
) -> torch.Tensor:
    """Binary solvent occupancy from surface distance.

    Solvent voxels are those with ``dist_map >= probe_radius`` (outside the
    probe-extended protein volume).

    Parameters
    ----------
    dist_map : torch.Tensor
        Surface distances in Angstroms (from :func:`distance_to_surface`).
    probe_radius : float
        Probe radius in Angstroms.

    Returns
    -------
    torch.Tensor
        Float occupancy in ``{0, 1}``, same shape as ``dist_map``
        (``1`` = solvent, ``0`` = protein / excluded).
    """
    if probe_radius < 0:
        raise ValueError("probe_radius must be non-negative")
    return (dist_map >= probe_radius).to(dtype=dist_map.dtype)


def solvent_occupancy_from_structure_3d(
    structure: AtomicStructure,
    grid_config: GridConfig,
    *,
    probe_radius: float = 1.4,
    r_asymptote: float = 7.5,
    batch_size: int = 256,
) -> torch.Tensor:
    """Build VdW+probe solvent occupancy from an ``AtomicStructure`` and grid.

    Parameters
    ----------
    structure : AtomicStructure
        Unbatched structure (``positions_zyx`` shape ``(n_atoms, 3)``).
    grid_config : GridConfig
        Three-dimensional grid.
    probe_radius : float
        Probe radius in Angstroms.
    r_asymptote : float
        Neighborhood radius for distance updates.
    batch_size : int
        Passed through to :func:`distance_to_surface`.

    Returns
    -------
    torch.Tensor
        Solvent occupancy, shape ``(D, H, W)``.
    """
    _require_unbatched_structure(structure)
    dist_map, _ = distance_to_surface(
        structure.positions_zyx,
        structure.atomic_numbers,
        grid_config,
        r_asymptote=r_asymptote,
        batch_size=batch_size,
    )
    return vdw_probe_occupancy(dist_map, probe_radius=probe_radius)


def _require_unbatched_structure(structure: AtomicStructure) -> None:
    if structure.positions_zyx.ndim != 2:
        raise ValueError(
            "solvent occupancy currently requires unbatched positions_zyx "
            f"with shape (n_atoms, 3), got {tuple(structure.positions_zyx.shape)}"
        )
