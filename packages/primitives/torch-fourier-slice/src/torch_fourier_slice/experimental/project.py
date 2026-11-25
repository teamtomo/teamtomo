"""Experimental Mojo-backed real-space projection: 3D volume -> 2D images, 2D -> 1D.

The real-space layer over :mod:`.extraction`: pad, correct for the
interpolation kernel, ``rfftn`` over the spatial dims, extract central slices
(or lines) with the Mojo kernel, ``irfftn`` back, unpad. Callers work entirely
in real space; the rfft layout is an implementation detail.

Mirrors :func:`torch_fourier_slice.project_3d_to_2d` and
:func:`torch_fourier_slice.project_2d_to_1d`, but the compute backend follows
the input tensor's device: a CPU tensor runs the multithreaded Mojo CPU kernel,
an ``mps`` / ``cuda`` tensor runs the Mojo GPU kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ._gridding import gridding_correction
from .extraction import (
    extract_central_lines_rfft_2d,
    extract_central_lines_rfft_2d_multichannel,
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multichannel,
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
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None,
    shifts_2d: torch.Tensor | None,
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
    extract_fn,
) -> torch.Tensor:
    """Shared pipeline; ``extract_fn`` picks the single / multichannel rank form."""
    pad = _pad_width(volume.shape[-1], pad_factor)
    if pad > 0:
        volume = F.pad(volume, pad=[pad] * 6)
    box = volume.shape[-1]

    # de-apodize *before* the transform so the interpolation puts it back
    volume = volume / gridding_correction(box, interpolation, volume.device)

    # the centre voxel (n // 2) to the origin; for odd n only ifftshift does that
    volume_rfft = torch.fft.rfftn(
        torch.fft.ifftshift(volume, dim=(-3, -2, -1)), dim=(-3, -2, -1)
    )
    # the rfft layer takes the inverse (image -> volume) rotation of the active
    # rotation the real-space layer accepts
    slices = extract_fn(
        volume_rfft.contiguous(),
        rotation_matrices.transpose(-1, -2),
        shifts_3d=shifts_3d,
        shifts_2d=shifts_2d,
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
    images = torch.fft.fftshift(
        torch.fft.irfftn(slices, dim=(-2, -1), s=(box, box)), dim=(-2, -1)
    )
    if pad > 0:
        images = F.pad(images, pad=[-pad] * 4)
    return images


def project_3d_to_2d(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
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
    """Project a real cubic volume to real 2D images (Mojo kernel).

    Parameters
    ----------
    volume : torch.Tensor
        Real cubic volume ``(d, d, d)``, centred on voxel ``d // 2``, of any
        side. Its device selects the CPU/GPU backend.
    rotation_matrices : torch.Tensor
        Real ``(3, 3)`` or ``(bp, 3, 3)`` rotation matrices (see ``zyx_matrices``).
        Each image is a projection along z of the volume rotated by its matrix
        (an active rotation), the same convention as
        :func:`torch_fourier_slice.project_3d_to_2d`. These are the transpose of
        the matrices :func:`.extract_central_slices_rfft_3d` takes for the same
        image.
    shifts_3d : torch.Tensor | None
        Optional ``(..., bp, 3)`` shifts in the volume frame, applied before
        the rotation.
    shifts_2d : torch.Tensor | None
        Optional ``(..., bp, 2)`` shifts in the image plane, applied after.
    pad_factor : float
        Real-space padding applied before the transform; ``2.0`` (default)
        doubles the box. Must be ``>= 1.0``.
    fftfreq_max : float | None
        Maximum frequency (cycles/pixel, Nyquist = 0.5) to include; output
        pixels beyond it are left at zero. Defaults to Nyquist.
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
    images : torch.Tensor
        Real ``(bp, d, d)`` projection images, on the input device.
    """
    if volume.dim() != 3:
        raise ValueError(
            "volume must be (d, d, d); use project_3d_to_2d_multichannel for "
            "(bv, d, d, d)"
        )
    return _project(
        volume,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
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
        extract_central_slices_rfft_3d,
    )


def project_3d_to_2d_multichannel(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
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
    """Project a batch of real cubic volumes to real 2D images (Mojo kernel).

    ``volume`` is ``(bv, d, d, d)``. Rotations are shared across volumes
    (``(bp, 3, 3)``) or per-volume (``(bv, bp, 3, 3)``). See
    :func:`project_3d_to_2d` for the shared parameters.

    Returns real ``(bp, bv, d, d)`` images (pose-major) on the input device.
    """
    if volume.dim() != 4:
        raise ValueError("volume must be (bv, d, d, d) for multi-channel")
    return _project(
        volume,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
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
        extract_central_slices_rfft_3d_multichannel,
    )


def _project_2d(
    image: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    pad_factor: float,
    fftfreq_max: float | None,
    yx_directions: bool,
    yx_shifts: bool,
    interpolation: str,
    extract_fn,
) -> torch.Tensor:
    """Shared 2D -> 1D pipeline; ``extract_fn`` picks the single / multichannel form."""
    pad = _pad_width(image.shape[-1], pad_factor)
    if pad > 0:
        image = F.pad(image, pad=[pad] * 4)
    box = image.shape[-1]

    # de-apodize *before* the transform so the interpolation puts it back
    image = image / gridding_correction(box, interpolation, image.device, ndim=2)

    # the centre pixel (n // 2) to the origin; for odd n only ifftshift does that
    image_rfft = torch.fft.rfftn(torch.fft.ifftshift(image, dim=(-2, -1)), dim=(-2, -1))
    # a line along a direction is the same in both layers: no transpose
    lines = extract_fn(
        image_rfft.contiguous(),
        directions,
        shifts_2d=shifts_2d,
        fftfreq_max=fftfreq_max,
        yx_directions=yx_directions,
        yx_shifts=yx_shifts,
        interpolation=interpolation,
    )
    lines = torch.fft.fftshift(torch.fft.irfft(lines, n=box, dim=-1), dim=-1)
    if pad > 0:
        lines = F.pad(lines, pad=[-pad] * 2)
    return lines


def project_2d_to_1d(
    image: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    pad_factor: float = 2.0,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Project a real square image onto lines (Mojo kernel).

    Each line is the image's projection onto a direction ``u``: its pixel ``s``
    (counted from the centre pixel ``d // 2``) sums the image along the line
    through ``s * u`` perpendicular to ``u``, a row of the Radon transform. By the
    projection-slice theorem it is the central line of the image's DFT along
    ``u``, which is how it is computed.

    Parameters
    ----------
    image : torch.Tensor
        Real square image ``(d, d)``, centred on pixel ``d // 2``, of any side.
        Its device selects the CPU/GPU backend.
    directions : torch.Tensor
        Real ``(2,)`` or ``(bp, 2)`` unit vectors ``u``, xy unless
        ``yx_directions``. A direction poses a line in real and Fourier space
        alike, so these are the directions :func:`.extract_central_lines_rfft_2d`
        takes for the same lines. For the active rotation ``R`` (xy) that
        :func:`torch_fourier_slice.project_2d_to_1d` takes, ``u`` is the first
        row of ``R``.
    shifts_2d : torch.Tensor | None
        Optional ``(..., bp, 2)`` shifts of the image, applied before projecting.
    pad_factor : float
        Real-space padding applied before the transform; ``2.0`` (default)
        doubles the box. Must be ``>= 1.0``.
    fftfreq_max : float | None
        Maximum frequency (cycles/pixel, Nyquist = 0.5) to include. Defaults to
        Nyquist.
    yx_directions : bool
        If True, ``directions`` are in yx order. If False (default) they are xy.
    yx_shifts : bool
        If True, ``shifts_2d`` are in yx order. If False (default) they are xy.
    interpolation : str
        ``"linear"`` (bilinear, default) or ``"cubic"`` (bicubic Catmull-Rom). The
        gridding correction follows this choice.

    Returns
    -------
    lines : torch.Tensor
        Real ``(bp, d)`` projections, on the input device.
    """
    if image.dim() != 2:
        raise ValueError(
            "image must be (d, d); use project_2d_to_1d_multichannel for (bv, d, d)"
        )
    return _project_2d(
        image,
        directions,
        shifts_2d,
        pad_factor,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
        extract_central_lines_rfft_2d,
    )


def project_2d_to_1d_multichannel(
    image: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    pad_factor: float = 2.0,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Project a batch of real square images onto lines (Mojo kernel).

    ``image`` is ``(bv, d, d)``. Directions are shared across images
    (``(bp, 2)``) or per-image (``(bv, bp, 2)``). See :func:`project_2d_to_1d`
    for the shared parameters.

    Returns real ``(bp, bv, d)`` lines (pose-major) on the input device.
    """
    if image.dim() != 3:
        raise ValueError("image must be (bv, d, d) for multi-channel")
    return _project_2d(
        image,
        directions,
        shifts_2d,
        pad_factor,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
        extract_central_lines_rfft_2d_multichannel,
    )
