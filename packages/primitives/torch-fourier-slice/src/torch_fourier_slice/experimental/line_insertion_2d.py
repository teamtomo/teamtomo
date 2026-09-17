"""Experimental Mojo-backed 1D->2D central-line insertion (rfft layer).

The adjoint of :func:`extract_central_line_rfft_2d`: scatter 1D central lines into
a 2D rfft image (Hermitian, DC at origin), optionally accumulating per-sample
weights for density compensation. Reconstructs an image from its sinogram lines
(2D direct Fourier inversion). Differentiable w.r.t. the input ``lines``.

- single image:  ``lines (bp, w)`` -> ``image (h, w_rfft)``
- multi-image:   ``lines (bp, bv, w)`` -> ``images (bv, h, w_rfft)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._autograd import InsertLine2DForward

if TYPE_CHECKING:
    import torch


def insert_central_line_rfft_2d(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 1D central lines into one 2D rfft image (Mojo scatter kernel).

    ``lines`` is complex ``(bp, w)``; ``directions`` are ``(2,)`` / ``(bp, 2)`` yx
    unit vectors; ``shifts_2d`` optional ``(..., bp, 2)`` yx translations (the
    conjugate phase ramp is applied). Returns ``(image, weight_image)`` -- complex
    ``(h, w_rfft)`` and real weights (``None`` if ``weights`` is ``None``).
    """
    if lines.dim() != 2:
        raise ValueError(
            "lines must be (bp, w); use insert_central_line_rfft_2d_multivolume"
        )
    data, wimg = InsertLine2DForward.apply(
        lines,
        weights,
        directions,
        shifts_2d,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
    )
    if weights is None:
        return data.squeeze(0), None
    return data.squeeze(0), wimg.squeeze(0)


def insert_central_line_rfft_2d_multivolume(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 1D central lines into a batch of 2D rfft images (Mojo kernel).

    ``lines`` is ``(bp, bv, w)`` (pose-major); ``weights`` (if given) matches it.
    Returns ``(images, weight_images)`` -- ``(bv, h, w_rfft)`` -- on input device.
    """
    if lines.dim() != 3:
        raise ValueError("lines must be (bp, bv, w) for multi-image")
    lines_bv = lines.transpose(0, 1).contiguous()  # (bp, bv, w) -> (bv, bp, w)
    w = weights.transpose(0, 1).contiguous() if weights is not None else None
    data, wimg = InsertLine2DForward.apply(
        lines_bv,
        w,
        directions,
        shifts_2d,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
    )
    return data, (wimg if weights is not None else None)
