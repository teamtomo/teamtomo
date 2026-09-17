"""Experimental Mojo-backed central-*line* extraction (rfft layer).

The atomic primitive of the frame-free line graph: sample 1D central lines from
a 3D rfft volume (DC at origin), indexed by a **direction** on the sphere. A
central line is the degenerate central slice whose in-plane (y) axis is collapsed
to the single DC row -- the projection-slice theorem applied a second time (a
line through the origin of a slice is a line through the origin of the 3D
transform). The node is a complex rfft half-line sampled along a direction ``u``
(a zyx unit vector, the real-space line direction, unchanged in Fourier space);
``line(-u) = conj(line(u))``, so nodes live on RP². A bare line needs only its
direction, not a rotation matrix -- rotating about the line's own axis is a gauge
the values are blind to.

Differentiable w.r.t. the volume (adjoint = 1D->3D line scatter), the
``directions`` (a per-node 3-vector gradient) and ``shifts_3d``.

Two rank forms share one Mojo kernel; the Python layer only squeezes / transposes:

- single volume: ``volume_rfft (d, h, w)`` -> ``lines (bp, w)``
- multi-volume:  ``volume_rfft (bv, d, h, w)`` -> ``lines (bp, bv, w)``
  (directions shared across volumes, or per-volume via ``(bv, bp, 3)``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._autograd import ProjectLineForward

if TYPE_CHECKING:
    import torch


def _extract_line(
    volume_rfft,
    directions,
    shifts_3d,
    output_length,
    oversampling,
    fourier_radius_cutoff,
    interpolation,
):
    """Run the differentiable kernel in its canonical ``(bv, bp, w)`` layout."""
    return ProjectLineForward.apply(
        volume_rfft,
        directions,
        shifts_3d,
        output_length,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
    )


def extract_central_line_rfft_3d(
    volume_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    output_length: int | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from one 3D rfft volume (Mojo kernel).

    Parameters
    ----------
    volume_rfft : torch.Tensor
        Complex rfft volume ``(d, h, w)`` (DC at origin, cubic, even). Its device
        selects the CPU/GPU backend.
    directions : torch.Tensor
        Real ``(3,)`` / ``(bp, 3)`` zyx **unit** direction vectors on the sphere;
        the line is sampled along ``k = s * u``. ``bp`` is the number of line
        nodes. A non-unit ``u`` rescales the line's frequency sampling.
    shifts_3d : torch.Tensor | None
        Optional ``(..., bp, 3)`` zyx shift in the volume frame; applied as the
        per-node ``s * (u . t)`` phase ramp (the design's translation model).
    output_length : int | None
        Even box length ``L`` of the line; the node is the rfft half-line of
        length ``L//2+1``. Defaults to the volume side ``h``.
    oversampling, fourier_radius_cutoff, interpolation
        As for the slice kernels (cutoff defaults to Nyquist ``L/2``;
        interpolation ``"linear"`` / ``"cubic"``).

    Returns complex ``(bp, w)`` lines (rfft half-line, DC at origin) on the input
    device, where ``w = output_length//2 + 1``.
    """
    if volume_rfft.dim() != 3:
        raise ValueError(
            "volume_rfft must be (d, h, w) for a single volume; use "
            "extract_central_line_rfft_3d_multivolume for (bv, d, h, w)"
        )
    out = _extract_line(
        volume_rfft,
        directions,
        shifts_3d,
        output_length,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
    )
    return out.squeeze(0)  # (1, bp, w) -> (bp, w)


def extract_central_line_rfft_3d_multivolume(
    volume_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    output_length: int | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from a batch of 3D rfft volumes (Mojo kernel).

    ``volume_rfft`` is ``(bv, d, h, w)``. Directions are shared across volumes
    (``(bp, 3)``) or per-volume (``(bv, bp, 3)``).

    Returns complex ``(bp, bv, w)`` lines (pose-major) on the input device.
    """
    if volume_rfft.dim() != 4:
        raise ValueError("volume_rfft must be (bv, d, h, w) for multi-volume")
    out = _extract_line(
        volume_rfft,
        directions,
        shifts_3d,
        output_length,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
    )
    return out.transpose(0, 1).contiguous()  # (bv, bp, w) -> (bp, bv, w)
