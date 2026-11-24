"""Torch reference real-space backprojection: 2D images -> 3D volume.

Same API and pipeline as :mod:`torch_fourier_slice.experimental.backproject`,
over the torch reference central-slice insertion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..backproject import _backproject
from .insertion import (
    insert_central_slices_rfft_3d,
    insert_central_slices_rfft_3d_multichannel,
)

if TYPE_CHECKING:
    import torch


def backproject_2d_to_3d(
    images: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
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
    """Reconstruct a real cubic volume from real 2D images (torch reference).

    See :func:`torch_fourier_slice.experimental.backproject_2d_to_3d`.
    """
    if images.dim() != 3:
        raise ValueError(
            "images must be (bp, d, d); use backproject_2d_to_3d_multichannel for "
            "(bp, bv, d, d)"
        )
    return _backproject(
        images,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        weights,
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
        insert_central_slices_rfft_3d,
    )


def backproject_2d_to_3d_multichannel(
    images: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
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
    """Reconstruct a batch of real cubic volumes from real 2D images (torch reference).

    See :func:`torch_fourier_slice.experimental.backproject_2d_to_3d_multichannel`.
    """
    if images.dim() != 4:
        raise ValueError("images must be (bp, bv, d, d) for multi-channel")
    return _backproject(
        images,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        weights,
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
        insert_central_slices_rfft_3d_multichannel,
    )
