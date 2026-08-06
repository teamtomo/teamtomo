"""Experimental Mojo-backed central-*line* insertion (rfft layer).

The rfft-level adjoint of :func:`extract_central_line_rfft_3d`: scatter 1D
central lines into a 3D rfft volume (Hermitian, DC at origin), optionally
accumulating per-sample weights for density compensation. This is the
insert-and-``irfft`` reconstruction path of the frame-free line graph.

Differentiable w.r.t. the input ``lines`` (adjoint forward line projection),
``weights`` (weight-splat adjoint), ``directions`` (a per-node 3-vector
gradient) and ``shifts_3d``.

Two rank forms share one Mojo kernel; the Python layer only squeezes / transposes:

- single volume: ``lines (bp, w)`` -> ``volume (d, h, w)``
- multi-volume:  ``lines (bp, bv, w)`` -> ``volumes (bv, d, h, w)``
  (directions shared across volumes, or per-volume via ``(bv, bp, 3)``).

``directions`` are zyx unit vectors (as in the extractor); the insertion applies
the *conjugate* 3D-shift phase ramp (the forward adjoint). ``weights`` is an
optional real per-sample tensor matching ``lines``, accumulated into a separate
weight volume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._autograd import InsertLineForward

if TYPE_CHECKING:
    import torch


def _insert_line(
    lines,
    weights,
    directions,
    shifts_3d,
    oversampling,
    fourier_radius_cutoff,
    interpolation,
):
    """Run the differentiable kernel in its canonical ``(bv, bp, w)`` layout."""
    return InsertLineForward.apply(
        lines,
        weights,
        directions,
        shifts_3d,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
    )


def insert_central_line_rfft_3d(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 1D central lines into one 3D rfft volume (Mojo scatter kernel).

    ``lines`` is ``(bp, w)`` complex rfft half-lines (DC at origin); its device
    selects the backend. ``directions`` are ``(3,)`` / ``(bp, 3)`` zyx unit
    vectors (same convention as the extractor). ``weights`` (optional, real,
    matching ``lines``) accumulate into a weight volume for density compensation.

    Returns ``(volume, weight_volume)`` -- complex ``(d, h, w)`` accumulated data
    and real ``(d, h, w)`` accumulated weights (``None`` if ``weights`` is
    ``None``) -- on the input device.
    """
    if lines.dim() != 2:
        raise ValueError(
            "lines must be (bp, w) for a single volume; use "
            "insert_central_line_rfft_3d_multivolume for (bp, bv, w)"
        )
    data, weight_vol = _insert_line(
        lines,
        weights,
        directions,
        shifts_3d,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
    )
    if weights is None:
        return data.squeeze(0), None
    return data.squeeze(0), weight_vol.squeeze(0)


def insert_central_line_rfft_3d_multivolume(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 1D central lines into a batch of 3D rfft volumes (Mojo kernel).

    ``lines`` is ``(bp, bv, w)`` (pose-major); ``weights`` (if given) matches it.
    Directions are shared (``(bp, 3)``) or per-volume (``(bv, bp, 3)``).

    Returns ``(volumes, weight_volumes)`` -- complex ``(bv, d, h, w)`` data and
    real ``(bv, d, h, w)`` weights (``None`` if ``weights`` is ``None``) -- on the
    input device.
    """
    if lines.dim() != 3:
        raise ValueError("lines must be (bp, bv, w) for multi-volume")
    lines_bv = lines.transpose(0, 1).contiguous()  # (bp, bv, w) -> (bv, bp, w)
    w = weights.transpose(0, 1).contiguous() if weights is not None else None
    data, weight_vol = _insert_line(
        lines_bv,
        w,
        directions,
        shifts_3d,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
    )
    return data, (weight_vol if weights is not None else None)
