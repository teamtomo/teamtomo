"""Experimental Mojo-backed central-slice / central-line extraction (rfft layer).

Central slices: 3D volume -> 2D slices
--------------------------------------

The rfft-level forward operator: sample 2D central slices from a 3D rfft volume
(DC at origin), the Fourier-slice theorem's forward direction. This is the
experimental analogue of :func:`torch_fourier_slice.extract_central_slices_rfft_3d`
and is differentiable w.r.t. the volume, rotation_matrices, and 2D / 3D shifts.

Two rank forms share one Mojo kernel; the Python layer only squeezes / transposes:

- single volume: ``volume_rfft (d, h, w)`` -> ``slices (bp, h, w)``
- multi-channel:  ``volume_rfft (bv, d, h, w)`` -> ``slices (bp, bv, h, w)``
  (poses shared across volumes, or per-volume via
  ``rotation_matrices (bv, bp, 3, 3)``).

Shared parameters (both forms):

- ``rotation_matrices``: ``(3, 3)`` / ``(bp, 3, 3)`` (single volume) or also
  ``(bv, bp, 3, 3)`` for per-volume poses (multi-channel). They act on xyz
  vectors unless ``zyx_matrices=True``.
- ``shifts_3d``: optional ``(..., bp, 3)`` shifts in the volume frame (before
  rotation), xyz unless ``zyx_shifts=True``; ``shifts_2d``: optional
  ``(..., bp, 2)`` shifts in the projection plane (after rotation), xy unless
  ``yx_shifts=True``.
- ``output_shape``: ``(H_out, W_out)`` square/even, default ``(h, h)``.
- ``oversampling``: coordinate scale (>1 oversamples); ``fftfreq_max``:
  cutoff in cycles/pixel, default Nyquist (0.5); ``interpolation``: ``"linear"``
  / ``"cubic"``.
- ``apply_ewald_curvature`` / ``ewald_voltage_kv`` / ``ewald_flip_sign`` /
  ``ewald_px_size``: bend the slice onto an Ewald sphere, as in the canonical
  layer (default: flat slice).

Central lines: 3D volume -> 1D lines
------------------------------------

The atomic primitive of the frame-free line graph: sample 1D central lines from
a 3D rfft volume (DC at origin), indexed by a **direction** on the sphere. A
central line is the degenerate central slice whose in-plane (y) axis is collapsed
to the single DC row -- the projection-slice theorem applied a second time (a
line through the origin of a slice is a line through the origin of the 3D
transform). The node is a complex rfft half-line sampled along a direction ``u``
(a unit vector, the real-space line direction, unchanged in Fourier space);
``line(-u) = conj(line(u))``, so nodes live on RP². A bare line needs only its
direction, not a rotation matrix -- rotating about the line's own axis is a gauge
the values are blind to.

Differentiable w.r.t. the volume (adjoint = 1D->3D line scatter), the
``directions`` (a per-node 3-vector gradient) and ``shifts_3d``.

Two rank forms share one Mojo kernel; the Python layer only squeezes / transposes:

- single volume: ``volume_rfft (d, h, w)`` -> ``lines (bp, w)``
- multi-channel:  ``volume_rfft (bv, d, h, w)`` -> ``lines (bp, bv, w)``
  (directions shared across volumes, or per-volume via ``(bv, bp, 3)``).

Central lines: 2D image -> 1D lines
-----------------------------------

Sample 1D central lines from a 2D rfft image (DC at origin), indexed by a
**direction** ``u`` on the circle (a unit vector). ``line[s] =
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

from ._backend._line_2d import ExtractLines2D
from ._backend._line_3d import ExtractLines3D
from ._backend._slice_3d import ExtractSlices3D
from ._conventions import ewald_coefficient, to_zyx_matrices, to_zyx_vectors

if TYPE_CHECKING:
    import torch


def _extract_slices_3d(
    volume_rfft,
    rotation_matrices,
    shifts_3d,
    shifts_2d,
    output_shape,
    oversampling,
    fftfreq_max,
    zyx_matrices,
    zyx_shifts,
    yx_shifts,
    interpolation,
    apply_ewald_curvature,
    ewald_voltage_kv,
    ewald_flip_sign,
    ewald_px_size,
):
    """Run the differentiable kernel in its canonical ``(bv, bp, ...)`` layout."""
    return ExtractSlices3D.apply(
        volume_rfft,
        to_zyx_matrices(rotation_matrices, zyx_matrices),
        to_zyx_vectors(shifts_2d, yx_shifts),
        to_zyx_vectors(shifts_3d, zyx_shifts),
        output_shape,
        oversampling,
        fftfreq_max,
        interpolation,
        ewald_coefficient(
            volume_rfft.shape[-2],
            apply_ewald_curvature,
            ewald_voltage_kv,
            ewald_flip_sign,
            ewald_px_size,
        ),
    )


def extract_central_slices_rfft_3d(
    volume_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    output_shape: tuple[int, int] | None = None,
    oversampling: float = 1.0,
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
            "extract_central_slices_rfft_3d_multichannel for (bv, d, h, w)"
        )
    out = _extract_slices_3d(
        volume_rfft,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        output_shape,
        oversampling,
        fftfreq_max,
        zyx_matrices,
        zyx_shifts,
        yx_shifts,
        interpolation,
        apply_ewald_curvature,
        ewald_voltage_kv,
        ewald_flip_sign,
        ewald_px_size,
    )
    return out.squeeze(0)  # (1, bp, h, w) -> (bp, h, w)


def extract_central_slices_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    output_shape: tuple[int, int] | None = None,
    oversampling: float = 1.0,
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
    """Extract 2D central slices from a batch of 3D rfft volumes (Mojo kernel).

    ``volume_rfft`` is ``(bv, d, h, w)`` (complex rfft, DC at origin). Poses are
    shared across volumes (``rotation_matrices (bp, 3, 3)``) or per-volume
    (``(bv, bp, 3, 3)``). See the module docstring for the shared parameters.

    Returns complex ``(bp, bv, h, w)`` central slices (pose-major) on the input
    device.
    """
    if volume_rfft.dim() != 4:
        raise ValueError("volume_rfft must be (bv, d, h, w) for multi-channel")
    out = _extract_slices_3d(
        volume_rfft,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
        output_shape,
        oversampling,
        fftfreq_max,
        zyx_matrices,
        zyx_shifts,
        yx_shifts,
        interpolation,
        apply_ewald_curvature,
        ewald_voltage_kv,
        ewald_flip_sign,
        ewald_px_size,
    )
    return out.transpose(0, 1).contiguous()  # (bv, bp, ...) -> (bp, bv, ...)


def _extract_lines_3d(
    volume_rfft,
    directions,
    shifts_3d,
    output_length,
    oversampling,
    fftfreq_max,
    zyx_directions,
    zyx_shifts,
    interpolation,
):
    """Run the differentiable kernel in its canonical ``(bv, bp, w)`` layout."""
    return ExtractLines3D.apply(
        volume_rfft,
        to_zyx_vectors(directions, zyx_directions),
        to_zyx_vectors(shifts_3d, zyx_shifts),
        output_length,
        oversampling,
        fftfreq_max,
        interpolation,
    )


def extract_central_lines_rfft_3d(
    volume_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    output_length: int | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    zyx_directions: bool = False,
    zyx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from one 3D rfft volume (Mojo kernel).

    Parameters
    ----------
    volume_rfft : torch.Tensor
        Complex rfft volume ``(d, h, w)`` (DC at origin, cubic, even). Its device
        selects the CPU/GPU backend.
    directions : torch.Tensor
        Real ``(3,)`` / ``(bp, 3)`` **unit** direction vectors on the sphere;
        the line is sampled along ``k = s * u``. ``bp`` is the number of line
        nodes. A non-unit ``u`` rescales the line's frequency sampling.
    shifts_3d : torch.Tensor | None
        Optional ``(..., bp, 3)`` shift in the volume frame; applied as the
        per-node ``s * (u . t)`` phase ramp (the design's translation model).
    output_length : int | None
        Even box length ``L`` of the line; the node is the rfft half-line of
        length ``L//2+1``. Defaults to the volume side ``h``.
    oversampling, fftfreq_max, interpolation
        As for the slice extraction (cutoff defaults to Nyquist ``0.5``;
        interpolation ``"linear"`` / ``"cubic"``).
    zyx_directions, zyx_shifts : bool
        If True, ``directions`` / ``shifts_3d`` are in zyx order. If False
        (default) they are xyz.

    Returns complex ``(bp, w)`` lines (rfft half-line, DC at origin) on the input
    device, where ``w = output_length//2 + 1``.
    """
    if volume_rfft.dim() != 3:
        raise ValueError(
            "volume_rfft must be (d, h, w) for a single volume; use "
            "extract_central_lines_rfft_3d_multichannel for (bv, d, h, w)"
        )
    out = _extract_lines_3d(
        volume_rfft,
        directions,
        shifts_3d,
        output_length,
        oversampling,
        fftfreq_max,
        zyx_directions,
        zyx_shifts,
        interpolation,
    )
    return out.squeeze(0)  # (1, bp, w) -> (bp, w)


def extract_central_lines_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    output_length: int | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    zyx_directions: bool = False,
    zyx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from a batch of 3D rfft volumes (Mojo kernel).

    ``volume_rfft`` is ``(bv, d, h, w)``. Directions are shared across volumes
    (``(bp, 3)``) or per-volume (``(bv, bp, 3)``).

    Returns complex ``(bp, bv, w)`` lines (pose-major) on the input device.
    """
    if volume_rfft.dim() != 4:
        raise ValueError("volume_rfft must be (bv, d, h, w) for multi-channel")
    out = _extract_lines_3d(
        volume_rfft,
        directions,
        shifts_3d,
        output_length,
        oversampling,
        fftfreq_max,
        zyx_directions,
        zyx_shifts,
        interpolation,
    )
    return out.transpose(0, 1).contiguous()  # (bv, bp, w) -> (bp, bv, w)


def _extract_lines_2d(
    image_rfft,
    directions,
    shifts_2d,
    output_length,
    oversampling,
    fftfreq_max,
    yx_directions,
    yx_shifts,
    interpolation,
):
    """Run the differentiable kernel in its canonical ``(bv, bp, w)`` layout."""
    return ExtractLines2D.apply(
        image_rfft,
        to_zyx_vectors(directions, yx_directions),
        to_zyx_vectors(shifts_2d, yx_shifts),
        output_length,
        oversampling,
        fftfreq_max,
        interpolation,
    )


def extract_central_lines_rfft_2d(
    image_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    output_length: int | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from one 2D rfft image (Mojo kernel).

    ``image_rfft`` is complex ``(h, w)`` (DC at origin, ``w = h//2+1``, even).
    ``directions`` are ``(2,)`` / ``(bp, 2)`` unit vectors; ``shifts_2d`` are
    optional ``(..., bp, 2)`` image translations (phase ramp). Both are xy unless
    ``yx_directions`` / ``yx_shifts`` is True. Returns complex ``(bp, w_out)``
    half-lines on the input device.
    """
    if image_rfft.dim() != 2:
        raise ValueError(
            "image_rfft must be (h, w); use extract_central_lines_rfft_2d_multichannel"
        )
    out = _extract_lines_2d(
        image_rfft,
        directions,
        shifts_2d,
        output_length,
        oversampling,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
    )
    return out.squeeze(0)  # (1, bp, w) -> (bp, w)


def extract_central_lines_rfft_2d_multichannel(
    image_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    output_length: int | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> torch.Tensor:
    """Extract 1D central lines from a batch of 2D rfft images (Mojo kernel).

    ``image_rfft`` is ``(bv, h, w)``; directions shared ``(bp, 2)`` or per-image
    ``(bv, bp, 2)``. Returns complex ``(bp, bv, w_out)`` (pose-major).
    """
    if image_rfft.dim() != 3:
        raise ValueError("image_rfft must be (bv, h, w) for multi-image")
    out = _extract_lines_2d(
        image_rfft,
        directions,
        shifts_2d,
        output_length,
        oversampling,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
    )
    return out.transpose(0, 1).contiguous()  # (bv, bp, w) -> (bp, bv, w)
