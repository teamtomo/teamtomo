"""Experimental Mojo-backed 2D->1D central-line extraction (rfft layer).

Sample 1D central lines from a 2D rfft image (DC at origin), indexed by a
**direction** ``u = (u_y, u_x)`` on the circle (yx unit vector). ``line[s] =
F(s*u)``; by the projection-slice theorem this is the 1D FT of the image's
projection onto the ``u`` axis (a Radon sinogram row). This is the graph's node
factory: nodes come from each crop's 2D FT, not the 3D volume.

Differentiable w.r.t. the image (adjoint = 1D->2D line scatter). Direction
gradients (the 2D pose-gradient kernel) are a follow-up.

- single image:  ``image_rfft (h, w)`` -> ``lines (bp, w_out)``
- multi-image:   ``image_rfft (bv, h, w)`` -> ``lines (bp, bv, w_out)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._autograd import ProjectLine2DForward

if TYPE_CHECKING:
    import torch


def extract_central_line_rfft_2d(
    image_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    output_length: int | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from one 2D rfft image (Mojo kernel).

    ``image_rfft`` is complex ``(h, w)`` (DC at origin, ``w = h//2+1``, even).
    ``directions`` are ``(2,)`` / ``(bp, 2)`` yx unit vectors; ``shifts_2d`` are
    optional ``(..., bp, 2)`` yx image translations (phase ramp). Returns complex
    ``(bp, w_out)`` half-lines on the input device.
    """
    if image_rfft.dim() != 2:
        raise ValueError(
            "image_rfft must be (h, w); use extract_central_line_rfft_2d_multivolume"
        )
    out = ProjectLine2DForward.apply(
        image_rfft,
        directions,
        shifts_2d,
        output_length,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
    )
    return out.squeeze(0)  # (1, bp, w) -> (bp, w)


def extract_central_line_rfft_2d_multivolume(
    image_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    output_length: int | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from a batch of 2D rfft images (Mojo kernel).

    ``image_rfft`` is ``(bv, h, w)``; directions shared ``(bp, 2)`` or per-image
    ``(bv, bp, 2)``. Returns complex ``(bp, bv, w_out)`` (pose-major).
    """
    if image_rfft.dim() != 3:
        raise ValueError("image_rfft must be (bv, h, w) for multi-image")
    out = ProjectLine2DForward.apply(
        image_rfft,
        directions,
        shifts_2d,
        output_length,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
    )
    return out.transpose(0, 1).contiguous()  # (bv, bp, w) -> (bp, bv, w)
