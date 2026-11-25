"""Torch reference central-slice and central-line insertion (rfft layer).

Same API as :mod:`torch_fourier_slice.experimental.insertion`, computed with
torch ops instead of the Mojo kernels. See that module for the parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._backend._validation import rfft_volume_shape
from .._conventions import ewald_coefficient, to_zyx_matrices, to_zyx_vectors
from ._line import insert_lines_2d, insert_lines_3d
from ._slice import insert_slices_3d

if TYPE_CHECKING:
    import torch


def _insert_slices_3d(
    image_rfft,
    weights,
    rotation_matrices,
    shifts_3d,
    shifts_2d,
    volume_shape,
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
    volume_sidelength = rfft_volume_shape(volume_shape)[1]
    return insert_slices_3d(
        image_rfft,
        weights,
        to_zyx_matrices(rotation_matrices, zyx_matrices),
        to_zyx_vectors(shifts_2d, yx_shifts),
        to_zyx_vectors(shifts_3d, zyx_shifts),
        volume_shape,
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
    volume_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    zyx_shifts: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,
    ewald_flip_sign: bool = False,
    ewald_px_size: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert 2D central slices into one 3D rfft volume (torch reference).

    See :func:`torch_fourier_slice.experimental.insert_central_slices_rfft_3d`.
    Returns ``(volume, weight_volume)``.
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
        volume_shape,
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
    return data.squeeze(0), weight_vol.squeeze(0)


def insert_central_slices_rfft_3d_multichannel(
    image_rfft: torch.Tensor,
    volume_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    zyx_shifts: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,
    ewald_flip_sign: bool = False,
    ewald_px_size: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        volume_shape,
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
    volume_shape,
    fftfreq_max,
    zyx_directions,
    zyx_shifts,
    interpolation,
    line_length,
):
    """Insert in the kernels' canonical ``(bv, bp, w)`` layout."""
    return insert_lines_3d(
        lines,
        weights,
        to_zyx_vectors(directions, zyx_directions),
        to_zyx_vectors(shifts_3d, zyx_shifts),
        volume_shape,
        fftfreq_max,
        interpolation,
        line_length,
    )


def insert_central_lines_rfft_3d(
    lines: torch.Tensor,
    volume_shape: tuple[int, int, int],
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    fftfreq_max: float | None = None,
    line_length: int | None = None,
    zyx_directions: bool = False,
    zyx_shifts: bool = False,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert 1D central lines into one 3D rfft volume (torch reference).

    See :func:`torch_fourier_slice.experimental.insert_central_lines_rfft_3d`.
    Returns the complex ``(d, d, d // 2 + 1)`` volume and the real weights of the
    same shape.
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
        volume_shape,
        fftfreq_max,
        zyx_directions,
        zyx_shifts,
        interpolation,
        line_length,
    )
    return data.squeeze(0), weight_vol.squeeze(0)


def insert_central_lines_rfft_3d_multichannel(
    lines: torch.Tensor,
    volume_shape: tuple[int, int, int],
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    fftfreq_max: float | None = None,
    line_length: int | None = None,
    zyx_directions: bool = False,
    zyx_shifts: bool = False,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert 1D central lines into a batch of 3D rfft volumes (torch reference).

    See
    :func:`torch_fourier_slice.experimental.insert_central_lines_rfft_3d_multichannel`.
    ``lines`` is ``(bp, bv, w)`` (pose-major). Returns ``(volumes, weights)``,
    each ``(bv, d, d, d // 2 + 1)``.
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
        volume_shape,
        fftfreq_max,
        zyx_directions,
        zyx_shifts,
        interpolation,
        line_length,
    )


def _insert_lines_2d(
    lines,
    weights,
    directions,
    shifts_2d,
    image_shape,
    fftfreq_max,
    yx_directions,
    yx_shifts,
    interpolation,
    line_length,
):
    """Insert in the kernels' canonical ``(bv, bp, w)`` layout."""
    return insert_lines_2d(
        lines,
        weights,
        to_zyx_vectors(directions, yx_directions),
        to_zyx_vectors(shifts_2d, yx_shifts),
        image_shape,
        fftfreq_max,
        interpolation,
        line_length,
    )


def insert_central_lines_rfft_2d(
    lines: torch.Tensor,
    image_shape: tuple[int, int],
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    fftfreq_max: float | None = None,
    line_length: int | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert 1D central lines into one 2D rfft image (torch reference).

    See :func:`torch_fourier_slice.experimental.insert_central_lines_rfft_2d`.
    Returns the complex ``(h, h // 2 + 1)`` image and the real weights of the
    same shape.
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
        image_shape,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
        line_length,
    )
    return data.squeeze(0), weight_image.squeeze(0)


def insert_central_lines_rfft_2d_multichannel(
    lines: torch.Tensor,
    image_shape: tuple[int, int],
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    fftfreq_max: float | None = None,
    line_length: int | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert 1D central lines into a batch of 2D rfft images (torch reference).

    See
    :func:`torch_fourier_slice.experimental.insert_central_lines_rfft_2d_multichannel`.
    ``lines`` is ``(bp, bv, w)`` (pose-major). Returns ``(images, weights)``,
    each ``(bv, h, h // 2 + 1)``.
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
        image_shape,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
        line_length,
    )
