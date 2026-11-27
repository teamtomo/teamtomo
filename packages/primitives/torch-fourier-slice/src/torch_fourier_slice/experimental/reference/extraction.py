"""Torch reference central-slice and central-line extraction (rfft layer).

Same API as :mod:`torch_fourier_slice.experimental.extraction`, computed with
torch ops instead of the Mojo kernels. See that module for the parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._conventions import ewald_coefficient, to_zyx_matrices, to_zyx_vectors
from ._line import extract_lines_2d, extract_lines_3d
from ._slice import extract_slices_3d

if TYPE_CHECKING:
    import torch


def _extract_slices_3d(
    volume_rfft,
    rotation_matrices,
    shifts_3d,
    shifts_2d,
    output_shape,
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
    """Extract in the kernels' canonical ``(bv, bp, ...)`` layout."""
    return extract_slices_3d(
        volume_rfft,
        to_zyx_matrices(rotation_matrices, zyx_matrices),
        to_zyx_vectors(shifts_2d, yx_shifts),
        to_zyx_vectors(shifts_3d, zyx_shifts),
        output_shape,
        fftfreq_max,
        interpolation,
        ewald_coefficient(
            volume_rfft.shape[-2],
            apply_ewald_curvature,
            ewald_voltage_kv,
            ewald_flip_sign,
            ewald_px_size,
        ),
    )


def extract_central_slices_rfft_3d(
    volume_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    output_shape: tuple[int, int] | None = None,
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
    """Extract 2D central slices from one 3D rfft volume (torch reference).

    See :func:`torch_fourier_slice.experimental.extract_central_slices_rfft_3d`.
    Returns complex ``(bp, h, w)`` central slices (rfft, DC at origin).
    """
    if volume_rfft.dim() != 3:
        raise ValueError(
            "volume_rfft must be (d, h, w) for a single volume; use "
            "extract_central_slices_rfft_3d_multichannel for (bv, d, h, w)"
        )
    out = _extract_slices_3d(
        volume_rfft,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        output_shape,
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
    return out.squeeze(0)  # (1, bp, h, w) -> (bp, h, w)


def extract_central_slices_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    output_shape: tuple[int, int] | None = None,
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
    """Extract 2D central slices from a batch of 3D rfft volumes (torch reference).

    See
    :func:`torch_fourier_slice.experimental.extract_central_slices_rfft_3d_multichannel`.
    Returns complex ``(bp, bv, h, w)`` central slices (pose-major).
    """
    if volume_rfft.dim() != 4:
        raise ValueError("volume_rfft must be (bv, d, h, w) for multi-channel")
    out = _extract_slices_3d(
        volume_rfft,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        output_shape,
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
    return out.transpose(0, 1).contiguous()  # (bv, bp, ...) -> (bp, bv, ...)


def _extract_lines_3d(
    volume_rfft,
    directions,
    shifts_3d,
    output_length,
    fftfreq_max,
    zyx_directions,
    zyx_shifts,
    interpolation,
):
    """Extract in the kernels' canonical ``(bv, bp, w)`` layout."""
    return extract_lines_3d(
        volume_rfft,
        to_zyx_vectors(directions, zyx_directions),
        to_zyx_vectors(shifts_3d, zyx_shifts),
        output_length,
        fftfreq_max,
        interpolation,
    )


def extract_central_lines_rfft_3d(
    volume_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    output_length: int | None = None,
    fftfreq_max: float | None = None,
    zyx_directions: bool = False,
    zyx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from one 3D rfft volume (torch reference).

    See :func:`torch_fourier_slice.experimental.extract_central_lines_rfft_3d`.
    Returns complex ``(bp, w)`` lines (rfft half-line, DC at origin).
    """
    if volume_rfft.dim() != 3:
        raise ValueError(
            "volume_rfft must be (d, h, w) for a single volume; use "
            "extract_central_lines_rfft_3d_multichannel for (bv, d, h, w)"
        )
    out = _extract_lines_3d(
        volume_rfft,
        directions,
        shifts_3d,
        output_length,
        fftfreq_max,
        zyx_directions,
        zyx_shifts,
        interpolation,
    )
    return out.squeeze(0)  # (1, bp, w) -> (bp, w)


def extract_central_lines_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    output_length: int | None = None,
    fftfreq_max: float | None = None,
    zyx_directions: bool = False,
    zyx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from a batch of 3D rfft volumes (torch reference).

    See
    :func:`torch_fourier_slice.experimental.extract_central_lines_rfft_3d_multichannel`.
    Returns complex ``(bp, bv, w)`` lines (pose-major).
    """
    if volume_rfft.dim() != 4:
        raise ValueError("volume_rfft must be (bv, d, h, w) for multi-channel")
    out = _extract_lines_3d(
        volume_rfft,
        directions,
        shifts_3d,
        output_length,
        fftfreq_max,
        zyx_directions,
        zyx_shifts,
        interpolation,
    )
    return out.transpose(0, 1).contiguous()  # (bv, bp, w) -> (bp, bv, w)


def _extract_lines_2d(
    image_rfft,
    directions,
    shifts_2d,
    output_length,
    fftfreq_max,
    yx_directions,
    yx_shifts,
    interpolation,
):
    """Extract in the kernels' canonical ``(bv, bp, w)`` layout."""
    return extract_lines_2d(
        image_rfft,
        to_zyx_vectors(directions, yx_directions),
        to_zyx_vectors(shifts_2d, yx_shifts),
        output_length,
        fftfreq_max,
        interpolation,
    )


def extract_central_lines_rfft_2d(
    image_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    output_length: int | None = None,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from one 2D rfft image (torch reference).

    See :func:`torch_fourier_slice.experimental.extract_central_lines_rfft_2d`.
    Returns complex ``(bp, w_out)`` half-lines.
    """
    if image_rfft.dim() != 2:
        raise ValueError(
            "image_rfft must be (h, w); use extract_central_lines_rfft_2d_multichannel"
        )
    out = _extract_lines_2d(
        image_rfft,
        directions,
        shifts_2d,
        output_length,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
    )
    return out.squeeze(0)  # (1, bp, w) -> (bp, w)


def extract_central_lines_rfft_2d_multichannel(
    image_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    output_length: int | None = None,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from a batch of 2D rfft images (torch reference).

    See
    :func:`torch_fourier_slice.experimental.extract_central_lines_rfft_2d_multichannel`.
    Returns complex ``(bp, bv, w_out)`` lines (pose-major).
    """
    if image_rfft.dim() != 3:
        raise ValueError("image_rfft must be (bv, h, w) for multi-image")
    out = _extract_lines_2d(
        image_rfft,
        directions,
        shifts_2d,
        output_length,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
    )
    return out.transpose(0, 1).contiguous()  # (bv, bp, w) -> (bp, bv, w)
