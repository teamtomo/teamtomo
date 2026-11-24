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


def _crop_or_pad_start(current: int, target: int) -> int:
    """Source index of output voxel 0 after :func:`crop_or_pad_to_shape`, per axis.

    A crop keeps ``target`` voxels starting at ``(current - target) // 2``.  A pad
    puts ``(target - current) // 2`` zeros in front, so output voxel 0 lies that
    many voxels before source voxel 0 and the start is negative.
    """
    if current >= target:
        return (current - target) // 2
    return -((target - current) // 2)


def crop_start_zyx(sim_box_size: int, box_shape: tuple[int, int, int]) -> torch.Tensor:
    """Offsets from a cubic simulation volume to it cropped or padded to ``box_shape``.

    A simulation voxel ``p`` lands at ``p - crop_start`` after
    :func:`crop_or_pad_to_shape`.  The offset is negative along axes where the
    simulation box is smaller than ``box_shape`` and gets zero-padded.
    """
    return torch.tensor(
        [_crop_or_pad_start(sim_box_size, n) for n in box_shape],
        dtype=torch.float32,
    )


def crop_start_xyz(sim_box_size: int, box_shape: tuple[int, int, int]) -> torch.Tensor:
    """:func:`crop_start_zyx` in XYZ order (for Cartesian coordinate columns)."""
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
