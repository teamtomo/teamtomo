"""Real-space convenience layer over the rfft-level extract / insert kernels.

These wrap :mod:`slice_extraction` / :mod:`slice_insertion` with the FFTs,
``pad_factor`` real-space padding, and the ``sinc**2`` gridding correction (the
multichannel-style correction for the trilinear/tricubic interpolation kernel),
so callers work entirely in real space:

- :func:`project_3d_to_2d` : real volume ``(d,d,d)`` -> real images ``(bp,d,d)``
- :func:`backproject_2d_to_3d` : real images ``(bp,d,d)`` -> real volume ``(d,d,d)``

Each has a ``_multivolume`` form mirroring the rfft layer (``(bv,d,d,d)`` volumes,
``(bp,bv,d,d)`` images). Rotations are zyx; shifts follow the kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_grid_utils import fftfreq_grid

from .slice_extraction import (
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multivolume,
)
from .slice_insertion import (
    insert_central_slices_rfft_3d,
    insert_central_slices_rfft_3d_multivolume,
)


def _pad_amount(side: int, pad_factor: float) -> int:
    if pad_factor <= 1.0:
        return 0
    return int((side * (pad_factor - 1.0)) // 2)


def _sinc2(shape, device) -> torch.Tensor:
    """``sinc(|fftfreq|)**2`` over a centered real-space grid (gridding kernel)."""
    grid = fftfreq_grid(
        image_shape=shape, rfft=False, fftshift=True, norm=True, device=device
    )
    return torch.sinc(grid) ** 2


def _rfft_origin_3d(volume: torch.Tensor) -> torch.Tensor:
    """Real cubic volume -> rfft with DC at the origin (the kernel's layout)."""
    return torch.fft.rfftn(
        torch.fft.fftshift(volume, dim=(-3, -2, -1)), dim=(-3, -2, -1)
    )


def _irfft_origin_2d(slices: torch.Tensor, box: int) -> torch.Tensor:
    """Rfft slices (DC at origin) -> centered real images ``(..., box, box)``."""
    img = torch.fft.irfftn(slices, dim=(-2, -1), s=(box, box))
    return torch.fft.fftshift(img, dim=(-2, -1))


def _rfft_origin_2d(images: torch.Tensor) -> torch.Tensor:
    """Real images -> rfft slices with DC at the origin (the kernel's layout)."""
    return torch.fft.rfftn(torch.fft.fftshift(images, dim=(-2, -1)), dim=(-2, -1))


def _irfft_origin_3d(volume_rfft: torch.Tensor, box: int) -> torch.Tensor:
    """Rfft volume (DC at origin) -> centered real volume ``(..., box, box, box)``."""
    vol = torch.fft.irfftn(volume_rfft, dim=(-3, -2, -1), s=(box, box, box))
    return torch.fft.fftshift(vol, dim=(-3, -2, -1))


def _project(
    volume,
    rotations,
    shifts_3d,
    shifts_2d,
    pad_factor,
    fourier_radius_cutoff,
    interpolation,
    ewald_curvature,
    extract_fn,
):
    """Shared forward pipeline (single / multivolume via ``extract_fn``)."""
    d = volume.shape[-1]
    p = _pad_amount(d, pad_factor)
    if p > 0:
        volume = F.pad(volume, pad=[p] * 6)
    box = volume.shape[-1]
    volume = volume * _sinc2(volume.shape[-3:], volume.device)
    slices = extract_fn(
        _rfft_origin_3d(volume),
        rotations,
        shifts_3d=shifts_3d,
        shifts_2d=shifts_2d,
        fourier_radius_cutoff=fourier_radius_cutoff,
        interpolation=interpolation,
        ewald_curvature=ewald_curvature,
    )
    images = _irfft_origin_2d(slices, box)
    if p > 0:
        images = F.pad(images, pad=[-p] * 4)
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
    """Project one real cubic volume ``(d, d, d)`` to real images ``(bp, d, d)``."""
    if volume.dim() != 3:
        raise ValueError(
            "volume must be (d, d, d); use project_3d_to_2d_multivolume for "
            "(bv, d, d, d)"
        )
    return _project(
        volume, rotations, shifts_3d, shifts_2d, pad_factor,
        fourier_radius_cutoff, interpolation, ewald_curvature,
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
    """Project real volumes ``(bv, d, d, d)`` to real images ``(bp, bv, d, d)``."""
    if volume.dim() != 4:
        raise ValueError("volume must be (bv, d, d, d) for multi-volume")
    return _project(
        volume, rotations, shifts_3d, shifts_2d, pad_factor,
        fourier_radius_cutoff, interpolation, ewald_curvature,
        extract_central_slices_rfft_3d_multivolume,
    )


def _backproject(
    images,
    rotations,
    shifts_3d,
    shifts_2d,
    weights,
    pad_factor,
    fourier_radius_cutoff,
    interpolation,
    ewald_curvature,
    insert_fn,
):
    """Shared adjoint pipeline (single / multivolume via ``insert_fn``)."""
    d = images.shape[-1]
    p = _pad_amount(d, pad_factor)
    if p > 0:
        images = F.pad(images, pad=[p] * 4)
    box = images.shape[-1]
    img_rfft = _rfft_origin_2d(images)
    # density weights (ones) -> num / den normalisation, like canonical TFS
    w = torch.ones(img_rfft.shape, dtype=torch.float32, device=img_rfft.device)
    if weights is not None:
        w = w * weights
    data, weight_vol = insert_fn(
        img_rfft,
        rotations,
        shifts_3d=shifts_3d,
        shifts_2d=shifts_2d,
        weights=w,
        fourier_radius_cutoff=fourier_radius_cutoff,
        interpolation=interpolation,
        ewald_curvature=ewald_curvature,
    )
    data = data / torch.clamp(weight_vol, min=1.0)
    vol = _irfft_origin_3d(data, box)
    vol = vol / _sinc2(vol.shape[-3:], vol.device)
    if p > 0:
        vol = F.pad(vol, pad=[-p] * 6)
    return vol


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
    """Reconstruct one real volume ``(d, d, d)`` from real images ``(bp, d, d)``.

    Weighted (density-normalised) backprojection: each voxel is the slice-weighted
    average, then sinc**2-corrected, mirroring canonical TFS ``backproject_2d_to_3d``.
    """
    if images.dim() != 3:
        raise ValueError(
            "images must be (bp, d, d); use backproject_2d_to_3d_multivolume for "
            "(bp, bv, d, d)"
        )
    return _backproject(
        images, rotations, shifts_3d, shifts_2d, weights, pad_factor,
        fourier_radius_cutoff, interpolation, ewald_curvature,
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
    """Reconstruct volumes ``(bv, d, d, d)`` from images ``(bp, bv, d, d)``."""
    if images.dim() != 4:
        raise ValueError("images must be (bp, bv, d, d) for multi-volume")
    return _backproject(
        images, rotations, shifts_3d, shifts_2d, weights, pad_factor,
        fourier_radius_cutoff, interpolation, ewald_curvature,
        insert_central_slices_rfft_3d_multivolume,
    )
