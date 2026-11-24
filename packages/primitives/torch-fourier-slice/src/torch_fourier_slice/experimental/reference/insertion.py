"""Torch reference central-slice and central-line insertion (rfft layer).

Same API as :mod:`torch_fourier_slice.experimental.insertion`, computed with
torch ops instead of the Mojo kernels. See that module for the parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._backend._common import reconstruction_volume_shape
from .._conventions import ewald_coefficient, to_zyx_matrices, to_zyx_vectors
from ._line import insert_lines_2d, insert_lines_3d
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


def _insert_lines_3d(
    lines,
    weights,
    directions,
    shifts_3d,
    oversampling,
    fftfreq_max,
    zyx_directions,
    zyx_shifts,
    interpolation,
):
    """Insert in the kernels' canonical ``(bv, bp, w)`` layout."""
    return insert_lines_3d(
        lines,
        weights,
        to_zyx_vectors(directions, zyx_directions),
        to_zyx_vectors(shifts_3d, zyx_shifts),
        oversampling,
        fftfreq_max,
        interpolation,
    )


def insert_central_lines_rfft_3d(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    zyx_directions: bool = False,
    zyx_shifts: bool = False,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 1D central lines into one 3D rfft volume (torch reference).

    See :func:`torch_fourier_slice.experimental.insert_central_lines_rfft_3d`.
    Returns the complex ``(d, h, w)`` volume and the real ``(d, h, w)`` weights
    (``None`` if ``weights`` is ``None``).
    """
    if lines.dim() != 2:
        raise ValueError(
            "lines must be (bp, w) for a single volume; use "
            "insert_central_lines_rfft_3d_multichannel for (bp, bv, w)"
        )
    data, weight_vol = _insert_lines_3d(
        lines,
        weights,
        directions,
        shifts_3d,
        oversampling,
        fftfreq_max,
        zyx_directions,
        zyx_shifts,
        interpolation,
    )
    if weight_vol is None:
        return data.squeeze(0), None
    return data.squeeze(0), weight_vol.squeeze(0)


def insert_central_lines_rfft_3d_multichannel(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    zyx_directions: bool = False,
    zyx_shifts: bool = False,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 1D central lines into a batch of 3D rfft volumes (torch reference).

    See
    :func:`torch_fourier_slice.experimental.insert_central_lines_rfft_3d_multichannel`.
    ``lines`` is ``(bp, bv, w)`` (pose-major). Returns ``(volumes, weights)``,
    each ``(bv, d, h, w)`` (weights ``None`` if ``weights`` is ``None``).
    """
    if lines.dim() != 3:
        raise ValueError("lines must be (bp, bv, w) for multi-channel")
    lines_bv = lines.transpose(0, 1).contiguous()  # (bp, bv, w) -> (bv, bp, w)
    w = weights.transpose(0, 1).contiguous() if weights is not None else None
    return _insert_lines_3d(
        lines_bv,
        w,
        directions,
        shifts_3d,
        oversampling,
        fftfreq_max,
        zyx_directions,
        zyx_shifts,
        interpolation,
    )


def _insert_lines_2d(
    lines,
    weights,
    directions,
    shifts_2d,
    oversampling,
    fftfreq_max,
    yx_directions,
    yx_shifts,
    interpolation,
):
    """Insert in the kernels' canonical ``(bv, bp, w)`` layout."""
    return insert_lines_2d(
        lines,
        weights,
        to_zyx_vectors(directions, yx_directions),
        to_zyx_vectors(shifts_2d, yx_shifts),
        oversampling,
        fftfreq_max,
        interpolation,
    )


def insert_central_lines_rfft_2d(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 1D central lines into one 2D rfft image (torch reference).

    See :func:`torch_fourier_slice.experimental.insert_central_lines_rfft_2d`.
    Returns the complex ``(h, w)`` image and the real ``(h, w)`` weights (``None``
    if ``weights`` is ``None``).
    """
    if lines.dim() != 2:
        raise ValueError(
            "lines must be (bp, w); use insert_central_lines_rfft_2d_multichannel"
        )
    data, weight_image = _insert_lines_2d(
        lines,
        weights,
        directions,
        shifts_2d,
        oversampling,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
    )
    if weight_image is None:
        return data.squeeze(0), None
    return data.squeeze(0), weight_image.squeeze(0)


def insert_central_lines_rfft_2d_multichannel(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 1D central lines into a batch of 2D rfft images (torch reference).

    See
    :func:`torch_fourier_slice.experimental.insert_central_lines_rfft_2d_multichannel`.
    ``lines`` is ``(bp, bv, w)`` (pose-major). Returns ``(images, weights)``,
    each ``(bv, h, w)`` (weights ``None`` if ``weights`` is ``None``).
    """
    if lines.dim() != 3:
        raise ValueError("lines must be (bp, bv, w) for multi-image")
    lines_bv = lines.transpose(0, 1).contiguous()  # (bp, bv, w) -> (bv, bp, w)
    w = weights.transpose(0, 1).contiguous() if weights is not None else None
    return _insert_lines_2d(
        lines_bv,
        w,
        directions,
        shifts_2d,
        oversampling,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
    )
