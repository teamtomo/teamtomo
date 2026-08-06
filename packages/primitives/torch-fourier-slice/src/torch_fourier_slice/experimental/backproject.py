"""Experimental Mojo-backed real-space backprojection: 2D images -> 3D volume.

The real-space layer over :mod:`.slice_insertion`, and the adjoint of
:mod:`.project`: pad, ``rfftn`` over the spatial dims, insert central slices with
the Mojo kernel, normalise by the accumulated density, ``irfftn`` back, correct
for the interpolation kernel, unpad.

Mirrors :func:`torch_fourier_slice.backproject_2d_to_3d`, but the compute backend
follows the input tensor's device: a CPU tensor runs the multithreaded Mojo CPU
kernel, an ``mps`` / ``cuda`` tensor runs the Mojo GPU kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ._gridding import gridding_correction
from .project import _pad_width
from .slice_insertion import (
    insert_central_slices_rfft_3d,
    insert_central_slices_rfft_3d_multivolume,
)


def _backproject(
    images: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None,
    shifts_2d: torch.Tensor | None,
    weights: torch.Tensor | None,
    pad_factor: float,
    fourier_radius_cutoff: float | None,
    interpolation: str,
    ewald_curvature: float,
    insert_fn,
) -> torch.Tensor:
    """Shared pipeline; ``insert_fn`` picks the single / multivolume rank form."""
    pad = _pad_width(images.shape[-1], pad_factor)
    if pad > 0:
        images = F.pad(images, pad=[pad] * 4)
    box = images.shape[-1]

    image_rfft = torch.fft.rfftn(
        torch.fft.fftshift(images, dim=(-2, -1)), dim=(-2, -1)
    ).contiguous()

    # unit weights accumulate the sampling density; any caller weights modulate it
    density = torch.ones(
        image_rfft.shape, dtype=torch.float32, device=image_rfft.device
    )
    if weights is not None:
        density = density * weights
    volume_rfft, weight_volume = insert_fn(
        image_rfft,
        rotations,
        shifts_3d=shifts_3d,
        shifts_2d=shifts_2d,
        weights=density,
        fourier_radius_cutoff=fourier_radius_cutoff,
        interpolation=interpolation,
        ewald_curvature=ewald_curvature,
    )
    # clamped so sparsely sampled high frequencies are not amplified into noise
    volume_rfft = volume_rfft / torch.clamp(weight_volume, min=1.0)

    volume = torch.fft.fftshift(
        torch.fft.irfftn(volume_rfft, dim=(-3, -2, -1), s=(box, box, box)),
        dim=(-3, -2, -1),
    )
    # undo the apodization the interpolation kernel imposed during insertion
    volume = volume / gridding_correction(box, interpolation, volume.device)
    if pad > 0:
        volume = F.pad(volume, pad=[-pad] * 6)
    return volume


def backproject_2d_to_3d(
    images: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    pad_factor: float = 2.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
    ewald_curvature: float = 0.0,
) -> torch.Tensor:
    """Reconstruct a real cubic volume from real 2D images (Mojo kernel).

    Density-weighted backprojection: each voxel is the sampling-weighted average
    of the slices that touch it, then corrected for the interpolation kernel.

    Parameters
    ----------
    images : torch.Tensor
        Real ``(bp, d, d)`` square projection images with an even side length.
        Their device selects the CPU/GPU backend.
    rotations : torch.Tensor
        Real ``(3, 3)`` or ``(bp, 3, 3)`` **zyx** rotation matrices, the same
        convention as :func:`.project_3d_to_2d`.
    shifts_3d : torch.Tensor | None
        Optional ``(..., bp, 3)`` zyx shifts in the volume frame; the conjugate
        phase ramp is applied (adjoint of the forward shift).
    shifts_2d : torch.Tensor | None
        Optional ``(..., bp, 2)`` yx image-plane shifts; likewise conjugated.
    weights : torch.Tensor | None
        Optional real per-pixel weights (e.g. CTF^2) on the *padded* rfft
        slices, modulating each sample's contribution to the density.
    pad_factor : float
        Real-space padding applied before the transform; ``2.0`` (default)
        doubles the box. Must be ``>= 1.0``.
    fourier_radius_cutoff : float | None
        Frequency radius in cycles beyond which input pixels are ignored.
        Defaults to Nyquist for the padded box.
    interpolation : str
        ``"linear"`` (trilinear, default) or ``"cubic"`` (tricubic Catmull-Rom).
        The gridding correction follows this choice.
    ewald_curvature : float
        Signed Ewald-sphere curvature coefficient; must match the value used in
        the projection this is the adjoint of.

    Returns
    -------
    volume : torch.Tensor
        Real ``(d, d, d)`` reconstruction, on the input device.
    """
    if images.dim() != 3:
        raise ValueError(
            "images must be (bp, d, d); use backproject_2d_to_3d_multivolume for "
            "(bp, bv, d, d)"
        )
    return _backproject(
        images,
        rotations,
        shifts_3d,
        shifts_2d,
        weights,
        pad_factor,
        fourier_radius_cutoff,
        interpolation,
        ewald_curvature,
        insert_central_slices_rfft_3d,
    )


def backproject_2d_to_3d_multivolume(
    images: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    pad_factor: float = 2.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
    ewald_curvature: float = 0.0,
) -> torch.Tensor:
    """Reconstruct a batch of real cubic volumes from real 2D images.

    ``images`` is ``(bp, bv, d, d)`` (pose-major). Rotations are shared across
    volumes (``(bp, 3, 3)``) or per-volume (``(bv, bp, 3, 3)``). See
    :func:`backproject_2d_to_3d` for the shared parameters.

    Returns real ``(bv, d, d, d)`` reconstructions on the input device.
    """
    if images.dim() != 4:
        raise ValueError("images must be (bp, bv, d, d) for multi-volume")
    return _backproject(
        images,
        rotations,
        shifts_3d,
        shifts_2d,
        weights,
        pad_factor,
        fourier_radius_cutoff,
        interpolation,
        ewald_curvature,
        insert_central_slices_rfft_3d_multivolume,
    )
