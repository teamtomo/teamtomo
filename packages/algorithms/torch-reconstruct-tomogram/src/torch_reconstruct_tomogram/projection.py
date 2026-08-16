"""Project 3D points into tilt images and crop patches for reconstruction."""

import torch
from torch_grid_utils import dft_center
from torch_subpixel_crop import subpixel_crop_2d
from torch_tilt_series import TiltSeries

from torch_reconstruct_tomogram.io import (
    load_tilt_series_images,
    normalize_on_central_crop,
)


def project_points(tilt_series: TiltSeries, points_zyx: torch.Tensor) -> torch.Tensor:
    """Project 3D points to 2D image pixel coordinates.

    - points are 3D zyx coordinates in Angstroms, relative to the tomogram center
    - tilt_series supplies the projection geometry (`tilt_series.project_points`
      works in Angstroms) and `tilt_series.pixel_spacing` (raises if unset),
      used to convert the projected Angstrom positions to pixels
    - projected 2D points are in pixels, relative to the center of each image
    """
    return tilt_series.project_points(points_zyx) / tilt_series.pixel_spacing


def _extract_particle_tilt_series(
    tilt_series: TiltSeries,
    images: torch.Tensor,
    points_zyx: torch.Tensor,
    sidelength: int,
    return_rfft: bool = True,
) -> torch.Tensor:
    """Extract a subtilt-series given already-loaded images."""
    projected_yx = project_points(tilt_series, points_zyx)
    projected_yx = projected_yx + dft_center(
        images.shape[-2:], rfft=False, fftshift=True, device=images.device
    )
    return subpixel_crop_2d(
        image=images,
        positions=projected_yx,
        sidelength=sidelength,
        return_rfft=return_rfft,
        decenter=return_rfft,
    )


def extract_particle_tilt_series(
    tilt_series: TiltSeries,
    points_zyx: torch.Tensor,
    sidelength: int,
    return_rfft: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Extract a subtilt-series at 3D location(s) in the sample.

    Loads (and, by default, normalizes) the raw tilt images matching
    `tilt_series` via `tilt_series.image_path`/`image_indices`.
    """
    images = load_tilt_series_images(tilt_series)
    if normalize:
        images = normalize_on_central_crop(images)
    return _extract_particle_tilt_series(
        tilt_series,
        images,
        points_zyx,
        sidelength,
        return_rfft=return_rfft,
    )
