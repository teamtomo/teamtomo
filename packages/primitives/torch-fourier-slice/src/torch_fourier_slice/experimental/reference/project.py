"""Torch reference real-space projection: 3D volume -> 2D images, 2D -> 1D.

Same API and pipeline as :mod:`torch_fourier_slice.experimental.project`, over
the torch reference central-slice and central-line extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..project import _project, _project_2d
from .extraction import (
    extract_central_lines_rfft_2d,
    extract_central_lines_rfft_2d_multichannel,
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multichannel,
)

if TYPE_CHECKING:
    import torch


def project_3d_to_2d(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    pad_factor: float = 2.0,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    zyx_shifts: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,
    ewald_flip_sign: bool = False,
    ewald_px_size: float = 1.0,
) -> torch.Tensor:
    """Project a real cubic volume to real 2D images (torch reference).

    See :func:`torch_fourier_slice.experimental.project_3d_to_2d`.
    """
    if volume.dim() != 3:
        raise ValueError(
            "volume must be (d, d, d); use project_3d_to_2d_multichannel for "
            "(bv, d, d, d)"
        )
    return _project(
        volume,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        pad_factor,
        fftfreq_max,
        zyx_matrices,
        zyx_shifts,
        yx_shifts,
        interpolation,
        apply_ewald_curvature,
        ewald_voltage_kv,
        ewald_flip_sign,
        ewald_px_size,
        extract_central_slices_rfft_3d,
    )


def project_3d_to_2d_multichannel(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    pad_factor: float = 2.0,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    zyx_shifts: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,
    ewald_flip_sign: bool = False,
    ewald_px_size: float = 1.0,
) -> torch.Tensor:
    """Project a batch of real cubic volumes to real 2D images (torch reference).

    See :func:`torch_fourier_slice.experimental.project_3d_to_2d_multichannel`.
    """
    if volume.dim() != 4:
        raise ValueError("volume must be (bv, d, d, d) for multi-channel")
    return _project(
        volume,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        pad_factor,
        fftfreq_max,
        zyx_matrices,
        zyx_shifts,
        yx_shifts,
        interpolation,
        apply_ewald_curvature,
        ewald_voltage_kv,
        ewald_flip_sign,
        ewald_px_size,
        extract_central_slices_rfft_3d_multichannel,
    )


def project_2d_to_1d(
    image: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    pad_factor: float = 2.0,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Project a real square image onto lines (torch reference).

    See :func:`torch_fourier_slice.experimental.project_2d_to_1d`.
    """
    if image.dim() != 2:
        raise ValueError(
            "image must be (d, d); use project_2d_to_1d_multichannel for (bv, d, d)"
        )
    return _project_2d(
        image,
        directions,
        shifts_2d,
        pad_factor,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
        extract_central_lines_rfft_2d,
    )


def project_2d_to_1d_multichannel(
    image: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    pad_factor: float = 2.0,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Project a batch of real square images onto lines (torch reference).

    See :func:`torch_fourier_slice.experimental.project_2d_to_1d_multichannel`.
    """
    if image.dim() != 3:
        raise ValueError("image must be (bv, d, d) for multi-channel")
    return _project_2d(
        image,
        directions,
        shifts_2d,
        pad_factor,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
        extract_central_lines_rfft_2d_multichannel,
    )
