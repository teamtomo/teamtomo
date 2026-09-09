"""Shared simulation-box geometry for potential simulation and atom transforms."""

from __future__ import annotations

import torch


def simulation_box_center_angstroms(box_size: int, pixel_size: float) -> float:
    """Centre of a cubic simulation box in Angstroms."""
    return (box_size - 1) / 2.0 * pixel_size


def simulation_box_center_zyx(
    box_size: int,
    pixel_size: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Centre of a cubic simulation box as a ``(3,)`` ZYX tensor in Angstroms."""
    center = simulation_box_center_angstroms(box_size, pixel_size)
    return torch.tensor([center, center, center], device=device, dtype=dtype)


def center_positions_in_simulation_box(
    positions_zyx: torch.Tensor,
    box_size: int,
    pixel_size: float,
) -> torch.Tensor:
    """Centre atom positions at the simulation-box centre (ZYX, Angstroms)."""
    box_center_zyx = positions_zyx.new_full(
        (3,), simulation_box_center_angstroms(box_size, pixel_size)
    )
    return positions_zyx - positions_zyx.mean(dim=-2, keepdim=True) + box_center_zyx


def crop_start_zyx(sim_box_size: int, box_shape: tuple[int, int, int]) -> torch.Tensor:
    """Crop offsets when a cubic simulation volume is cropped to ``box_shape``."""
    d, h, w = box_shape
    return torch.tensor(
        [
            max(0, (sim_box_size - d) // 2),
            max(0, (sim_box_size - h) // 2),
            max(0, (sim_box_size - w) // 2),
        ],
        dtype=torch.float32,
    )


def crop_start_xyz(sim_box_size: int, box_shape: tuple[int, int, int]) -> torch.Tensor:
    """Crop offsets in XYZ order (for Cartesian coordinate columns)."""
    return crop_start_zyx(sim_box_size, box_shape).flip(0)


def coords_xyz_to_simulation_voxels(
    coords_xyz: torch.Tensor,
    centroid_xyz: torch.Tensor,
    box_size: int,
    pixel_size: float,
) -> torch.Tensor:
    """Convert atom Cartesian coordinates to simulation voxel indices."""
    box_centre = simulation_box_center_angstroms(box_size, pixel_size)
    return (coords_xyz - centroid_xyz + box_centre) / pixel_size


def coords_xyz_from_simulation_voxels(
    coords_vox_xyz: torch.Tensor,
    centroid_xyz: torch.Tensor,
    box_size: int,
    pixel_size: float,
) -> torch.Tensor:
    """Invert :func:`coords_xyz_to_simulation_voxels`."""
    box_centre = simulation_box_center_angstroms(box_size, pixel_size)
    return coords_vox_xyz * pixel_size + centroid_xyz - box_centre
