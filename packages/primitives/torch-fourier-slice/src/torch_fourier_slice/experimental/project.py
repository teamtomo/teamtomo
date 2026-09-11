"""Experimental Mojo-backed real-space projection: 3D volume -> 2D images.

The real-space layer over :mod:`.slice_extraction`: pad, correct for the
interpolation kernel, ``rfftn`` over the spatial dims, extract central slices
with the Mojo kernel, ``irfftn`` back, unpad. Callers work entirely in real
space; the rfft layout is an implementation detail.

Mirrors :func:`torch_fourier_slice.project_3d_to_2d`, but the compute backend
follows the input tensor's device: a CPU tensor runs the multithreaded Mojo CPU
kernel, an ``mps`` / ``cuda`` tensor runs the Mojo GPU kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ._gridding import gridding_correction
from .slice_extraction import (
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multivolume,
)


def _pad_width(sidelength: int, pad_factor: float) -> int:
    """Per-side padding for ``pad_factor``, matching the canonical layer."""
    if pad_factor < 1.0:
        raise ValueError("pad_factor must be >= 1.0")
    if pad_factor == 1.0:
        return 0
    return int((sidelength * (pad_factor - 1.0)) // 2)


def _project(
    volume: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None,
    shifts_2d: torch.Tensor | None,
    pad_factor: float,
    fourier_radius_cutoff: float | None,
    interpolation: str,
    ewald_curvature: float,
    extract_fn,
) -> torch.Tensor:
    """Shared pipeline; ``extract_fn`` picks the single / multivolume rank form."""
    pad = _pad_width(volume.shape[-1], pad_factor)
    if pad > 0:
        volume = F.pad(volume, pad=[pad] * 6)
    box = volume.shape[-1]

    # de-apodize *before* the transform so the interpolation puts it back
    volume = volume / gridding_correction(box, interpolation, volume.device)

    volume_rfft = torch.fft.rfftn(
        torch.fft.fftshift(volume, dim=(-3, -2, -1)), dim=(-3, -2, -1)
    )
    slices = extract_fn(
        volume_rfft.contiguous(),
        rotations,
        shifts_3d=shifts_3d,
        shifts_2d=shifts_2d,
        fourier_radius_cutoff=fourier_radius_cutoff,
        interpolation=interpolation,
        ewald_curvature=ewald_curvature,
    )
    images = torch.fft.fftshift(
        torch.fft.irfftn(slices, dim=(-2, -1), s=(box, box)), dim=(-2, -1)
    )
    if pad > 0:
        images = F.pad(images, pad=[-pad] * 4)
    return images


def project_3d_to_2d(
    volume: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    pad_factor: float = 2.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
    ewald_curvature: float = 0.0,
) -> torch.Tensor:
    """Project a real cubic volume to real 2D images (Mojo kernel).

    Parameters
    ----------
    volume : torch.Tensor
        Real cubic volume ``(d, d, d)`` with an even side length. Its device
        selects the CPU/GPU backend.
    rotations : torch.Tensor
        Real ``(3, 3)`` or ``(bp, 3, 3)`` **zyx** rotation matrices.
    shifts_3d : torch.Tensor | None
        Optional ``(..., bp, 3)`` zyx shifts in the volume frame, applied before
        the rotation.
    shifts_2d : torch.Tensor | None
        Optional ``(..., bp, 2)`` yx shifts in the image plane, applied after.
    pad_factor : float
        Real-space padding applied before the transform; ``2.0`` (default)
        doubles the box. Must be ``>= 1.0``.
    fourier_radius_cutoff : float | None
        Frequency radius in cycles beyond which output pixels are left at zero.
        Defaults to Nyquist for the padded box.
    interpolation : str
        ``"linear"`` (trilinear, default) or ``"cubic"`` (tricubic Catmull-Rom).
        The gridding correction follows this choice.
    ewald_curvature : float
        Signed Ewald-sphere curvature coefficient; ``0.0`` (default) keeps the
        central slice flat.

    Returns
    -------
    images : torch.Tensor
        Real ``(bp, d, d)`` projection images, on the input device.
    """
    if volume.dim() != 3:
        raise ValueError(
            "volume must be (d, d, d); use project_3d_to_2d_multivolume for "
            "(bv, d, d, d)"
        )
    return _project(
        volume,
        rotations,
        shifts_3d,
        shifts_2d,
        pad_factor,
        fourier_radius_cutoff,
        interpolation,
        ewald_curvature,
        extract_central_slices_rfft_3d,
    )


def project_3d_to_2d_multivolume(
    volume: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    pad_factor: float = 2.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
    ewald_curvature: float = 0.0,
) -> torch.Tensor:
    """Project a batch of real cubic volumes to real 2D images (Mojo kernel).

    ``volume`` is ``(bv, d, d, d)``. Rotations are shared across volumes
    (``(bp, 3, 3)``) or per-volume (``(bv, bp, 3, 3)``). See
    :func:`project_3d_to_2d` for the shared parameters.

    Returns real ``(bp, bv, d, d)`` images (pose-major) on the input device.
    """
    if volume.dim() != 4:
        raise ValueError("volume must be (bv, d, d, d) for multi-volume")
    return _project(
        volume,
        rotations,
        shifts_3d,
        shifts_2d,
        pad_factor,
        fourier_radius_cutoff,
        interpolation,
        ewald_curvature,
        extract_central_slices_rfft_3d_multivolume,
    )
