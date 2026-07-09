"""Experimental Mojo-backed central-slice extraction (rfft layer).

The rfft-level forward operator: sample 2D central slices from a 3D rfft volume
(DC at origin), the Fourier-slice theorem's forward direction. This is the
experimental analogue of :func:`torch_fourier_slice.extract_central_slices_rfft_3d`
and is differentiable w.r.t. the volume, rotations, and 2D / 3D shifts.

Two rank forms share one Mojo kernel; the Python layer only squeezes / transposes:

- single volume: ``volume_rfft (d, h, w)`` -> ``slices (bp, h, w)``
- multi-volume:  ``volume_rfft (bv, d, h, w)`` -> ``slices (bp, bv, h, w)``
  (poses shared across volumes, or per-volume via ``rotations (bv, bp, 3, 3)``).

Shared parameters (both forms):

- ``rotations``: ``(3, 3)`` / ``(bp, 3, 3)`` zyx matrices (single volume) or also
  ``(bv, bp, 3, 3)`` for per-volume poses (multi-volume).
- ``shifts_3d``: optional ``(..., bp, 3)`` zyx shifts in the volume frame (before
  rotation); ``shifts_2d``: optional ``(..., bp, 2)`` yx shifts in the projection
  plane (after rotation).
- ``output_shape``: ``(H_out, W_out)`` square/even, default ``(h, h)``.
- ``oversampling``: coordinate scale (>1 oversamples); ``fourier_radius_cutoff``:
  cutoff in cycles, default Nyquist; ``interpolation``: ``"linear"`` / ``"cubic"``;
  ``ewald_curvature``: signed Ewald coefficient (0 = flat slice).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._autograd import ProjectForward

if TYPE_CHECKING:
    import torch


def _extract(
    volume_rfft,
    rotations,
    shifts_3d,
    shifts_2d,
    output_shape,
    oversampling,
    fourier_radius_cutoff,
    interpolation,
    ewald_curvature,
):
    """Run the differentiable kernel in its canonical ``(bv, bp, ...)`` layout."""
    return ProjectForward.apply(
        volume_rfft,
        rotations,
        shifts_2d,
        shifts_3d,
        output_shape,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
        ewald_curvature,
    )


def extract_central_slices_rfft_3d(
    volume_rfft: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    output_shape: tuple[int, int] | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
    ewald_curvature: float = 0.0,
) -> torch.Tensor:
    """Extract 2D central slices from one 3D rfft volume (Mojo kernel).

    ``volume_rfft`` is a complex rfft volume ``(d, h, w)`` with DC at the origin
    (``w = h//2+1``, cubic, even); its device selects the CPU/GPU backend. See the
    module docstring for the shared pose / sampling parameters.

    Returns complex ``(bp, h, w)`` central slices (rfft, DC at origin) on the
    input device.
    """
    if volume_rfft.dim() != 3:
        raise ValueError(
            "volume_rfft must be (d, h, w) for a single volume; use "
            "extract_central_slices_rfft_3d_multivolume for (bv, d, h, w)"
        )
    out = _extract(
        volume_rfft, rotations, shifts_3d, shifts_2d, output_shape,
        oversampling, fourier_radius_cutoff, interpolation, ewald_curvature,
    )
    return out.squeeze(0)  # (1, bp, h, w) -> (bp, h, w)


def extract_central_slices_rfft_3d_multivolume(
    volume_rfft: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    output_shape: tuple[int, int] | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
    ewald_curvature: float = 0.0,
) -> torch.Tensor:
    """Extract 2D central slices from a batch of 3D rfft volumes (Mojo kernel).

    ``volume_rfft`` is ``(bv, d, h, w)`` (complex rfft, DC at origin). Poses are
    shared across volumes (``rotations (bp, 3, 3)``) or per-volume
    (``(bv, bp, 3, 3)``). See the module docstring for the shared parameters.

    Returns complex ``(bp, bv, h, w)`` central slices (pose-major) on the input
    device.
    """
    if volume_rfft.dim() != 4:
        raise ValueError("volume_rfft must be (bv, d, h, w) for multi-volume")
    out = _extract(
        volume_rfft, rotations, shifts_3d, shifts_2d, output_shape,
        oversampling, fourier_radius_cutoff, interpolation, ewald_curvature,
    )
    return out.transpose(0, 1).contiguous()  # (bv, bp, ...) -> (bp, bv, ...)
