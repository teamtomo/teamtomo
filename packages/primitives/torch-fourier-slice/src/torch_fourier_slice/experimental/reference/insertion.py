"""Torch reference central-slice insertion (rfft layer).

Same API as :mod:`torch_fourier_slice.experimental.insertion`, computed with
torch ops instead of the Mojo kernels. See that module for the parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._backend._common import reconstruction_volume_shape
from .._conventions import ewald_coefficient, to_zyx_matrices, to_zyx_vectors
from ._slice_3d import insert_slices_3d

if TYPE_CHECKING:
    import torch


def _insert_slices_3d(
    image_rfft,
    weights,
    rotation_matrices,
    shifts_3d,
    shifts_2d,
    oversampling,
    fftfreq_max,
    zyx_matrices,
    zyx_shifts,
    yx_shifts,
    interpolation,
    apply_ewald_curvature,
    ewald_voltage_kv,
    ewald_flip_sign,
    ewald_px_size,
):
    """Insert in the kernels' canonical ``(bv, bp, ...)`` layout."""
    volume_sidelength = reconstruction_volume_shape(image_rfft.shape[-2], oversampling)[
        1
    ]
    return insert_slices_3d(
        image_rfft,
        weights,
        to_zyx_matrices(rotation_matrices, zyx_matrices),
        to_zyx_vectors(shifts_2d, yx_shifts),
        to_zyx_vectors(shifts_3d, zyx_shifts),
        oversampling,
        fftfreq_max,
        interpolation,
        ewald_coefficient(
            volume_sidelength,
            apply_ewald_curvature,
            ewald_voltage_kv,
            ewald_flip_sign,
            ewald_px_size,
        ),
    )


def insert_central_slices_rfft_3d(
    image_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    zyx_shifts: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,
    ewald_flip_sign: bool = False,
    ewald_px_size: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 2D central slices into one 3D rfft volume (torch reference).

    See :func:`torch_fourier_slice.experimental.insert_central_slices_rfft_3d`.
    Returns ``(volume, weight_volume)``; ``weight_volume`` is ``None`` if
    ``weights`` is ``None``.
    """
    if image_rfft.dim() != 3:
        raise ValueError(
            "image_rfft must be (bp, h, w) for a single volume; use "
            "insert_central_slices_rfft_3d_multichannel for (bp, bv, h, w)"
        )
    data, weight_vol = _insert_slices_3d(
        image_rfft,
        weights,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        oversampling,
        fftfreq_max,
        zyx_matrices,
        zyx_shifts,
        yx_shifts,
        interpolation,
        apply_ewald_curvature,
        ewald_voltage_kv,
        ewald_flip_sign,
        ewald_px_size,
    )
    if weight_vol is None:
        return data.squeeze(0), None
    return data.squeeze(0), weight_vol.squeeze(0)


def insert_central_slices_rfft_3d_multichannel(
    image_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    zyx_shifts: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,
    ewald_flip_sign: bool = False,
    ewald_px_size: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 2D central slices into a batch of 3D rfft volumes (torch reference).

    See
    :func:`torch_fourier_slice.experimental.insert_central_slices_rfft_3d_multichannel`.
    ``image_rfft`` is ``(bp, bv, h, w)`` (pose-major). Returns
    ``(volumes, weight_volumes)``, each ``(bv, d, h, w)``.
    """
    if image_rfft.dim() != 4:
        raise ValueError("image_rfft must be (bp, bv, h, w) for multi-channel")
    imgs = image_rfft.transpose(0, 1).contiguous()  # (bp, bv, ...) -> (bv, bp, ...)
    w = weights.transpose(0, 1).contiguous() if weights is not None else None
    return _insert_slices_3d(
        imgs,
        w,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        oversampling,
        fftfreq_max,
        zyx_matrices,
        zyx_shifts,
        yx_shifts,
        interpolation,
        apply_ewald_curvature,
        ewald_voltage_kv,
        ewald_flip_sign,
        ewald_px_size,
    )
