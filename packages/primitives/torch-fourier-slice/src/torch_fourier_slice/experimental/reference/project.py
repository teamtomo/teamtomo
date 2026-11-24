"""Torch reference real-space projection: 3D volume -> 2D images.

Same API and pipeline as :mod:`torch_fourier_slice.experimental.project`, over
the torch reference central-slice extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..project import _project
from .extraction import (
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
