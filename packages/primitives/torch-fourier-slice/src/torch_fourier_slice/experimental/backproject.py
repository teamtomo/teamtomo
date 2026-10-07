"""Experimental Mojo-backed real-space backprojection: 2D images -> 3D volume.

The real-space layer over :mod:`.insertion`, and the adjoint of
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
from .insertion import (
    insert_central_slices_rfft_3d,
    insert_central_slices_rfft_3d_multichannel,
)
from .project import _pad_width


def _backproject(
    images: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None,
    shifts_2d: torch.Tensor | None,
    weights: torch.Tensor | None,
    pad_factor: float,
    fftfreq_max: float | None,
    zyx_matrices: bool,
    zyx_shifts: bool,
    yx_shifts: bool,
    interpolation: str,
    apply_ewald_curvature: bool,
    ewald_voltage_kv: float,
    ewald_flip_sign: bool,
    ewald_px_size: float,
    insert_fn,
) -> torch.Tensor:
    """Shared pipeline; ``insert_fn`` picks the single / multichannel rank form."""
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
        rotation_matrices,
        shifts_3d=shifts_3d,
        shifts_2d=shifts_2d,
        weights=density,
        fftfreq_max=fftfreq_max,
        zyx_matrices=zyx_matrices,
        zyx_shifts=zyx_shifts,
        yx_shifts=yx_shifts,
        interpolation=interpolation,
        apply_ewald_curvature=apply_ewald_curvature,
        ewald_voltage_kv=ewald_voltage_kv,
        ewald_flip_sign=ewald_flip_sign,
        ewald_px_size=ewald_px_size,
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
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    pad_factor: float = 2.0,
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
    """Reconstruct a real cubic volume from real 2D images (Mojo kernel).

    Density-weighted backprojection: each voxel is the sampling-weighted average
    of the slices that touch it, then corrected for the interpolation kernel.

    Parameters
    ----------
    images : torch.Tensor
        Real ``(bp, d, d)`` square projection images with an even side length.
        Their device selects the CPU/GPU backend.
    rotation_matrices : torch.Tensor
        Real ``(3, 3)`` or ``(bp, 3, 3)`` rotation matrices (see ``zyx_matrices``).
    shifts_3d : torch.Tensor | None
        Optional ``(..., bp, 3)`` shifts in the volume frame; the conjugate
        phase ramp is applied (adjoint of the forward shift).
    shifts_2d : torch.Tensor | None
        Optional ``(..., bp, 2)`` image-plane shifts; likewise conjugated.
    weights : torch.Tensor | None
        Optional real per-pixel weights (e.g. CTF^2) on the *padded* rfft
        slices, modulating each sample's contribution to the density.
    pad_factor : float
        Real-space padding applied before the transform; ``2.0`` (default)
        doubles the box. Must be ``>= 1.0``.
    fftfreq_max : float | None
        Maximum frequency (cycles/pixel, Nyquist = 0.5) to include; input
        pixels beyond it are ignored. Defaults to Nyquist.
    zyx_matrices : bool
        If True, ``rotation_matrices`` act on zyx vectors. If False (default)
        they act on xyz vectors and are converted by flipping the last two axes.
    zyx_shifts : bool
        If True, ``shifts_3d`` are in zyx order. If False (default) they are xyz.
    yx_shifts : bool
        If True, ``shifts_2d`` are in yx order. If False (default) they are xy.
    interpolation : str
        ``"linear"`` (trilinear, default) or ``"cubic"`` (tricubic Catmull-Rom).
        The gridding correction follows this choice.
    apply_ewald_curvature : bool
        If True, bend the central slice onto an Ewald sphere. If False (default),
        use a flat central slice.
    ewald_voltage_kv : float
        Acceleration voltage in kV (default 300.0); sets the wavelength.
    ewald_flip_sign : bool
        If True, flip the sign of the Ewald curvature.
    ewald_px_size : float
        Pixel size in Angstroms / pixel.

    Returns
    -------
    volume : torch.Tensor
        Real ``(d, d, d)`` reconstruction, on the input device.
    """
    if images.dim() != 3:
        raise ValueError(
            "images must be (bp, d, d); use backproject_2d_to_3d_multichannel for "
            "(bp, bv, d, d)"
        )
    return _backproject(
        images,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        weights,
        pad_factor,
        fftfreq_max,
        zyx_matrices,
        zyx_shifts,
        yx_shifts,
        interpolation,
        apply_ewald_curvature,
        ewald_voltage_kv,
        ewald_flip_sign,
        ewald_px_size,
        insert_central_slices_rfft_3d,
    )


def backproject_2d_to_3d_multichannel(
    images: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    pad_factor: float = 2.0,
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
    """Reconstruct a batch of real cubic volumes from real 2D images.

    ``images`` is ``(bp, bv, d, d)`` (pose-major). Rotations are shared across
    volumes (``(bp, 3, 3)``) or per-volume (``(bv, bp, 3, 3)``). See
    :func:`backproject_2d_to_3d` for the shared parameters.

    Returns real ``(bv, d, d, d)`` reconstructions on the input device.
    """
    if images.dim() != 4:
        raise ValueError("images must be (bp, bv, d, d) for multi-channel")
    return _backproject(
        images,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        weights,
        pad_factor,
        fftfreq_max,
        zyx_matrices,
        zyx_shifts,
        yx_shifts,
        interpolation,
        apply_ewald_curvature,
        ewald_voltage_kv,
        ewald_flip_sign,
        ewald_px_size,
        insert_central_slices_rfft_3d_multichannel,
    )
