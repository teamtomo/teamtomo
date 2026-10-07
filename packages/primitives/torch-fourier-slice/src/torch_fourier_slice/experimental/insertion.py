"""Experimental Mojo-backed central-slice / central-line insertion (rfft layer).

Central slices: 2D slices -> 3D volume
--------------------------------------

The rfft-level adjoint operator: scatter 2D central slices into a 3D rfft volume
(Hermitian, DC at origin), optionally accumulating per-pixel weights for a
Wiener-style normalization. This is the experimental analogue of
:func:`torch_fourier_slice.insert_central_slices_rfft_3d` and is differentiable
w.r.t. the slices, weights, rotation_matrices, and 2D / 3D shifts.

Two rank forms share one Mojo kernel; the Python layer only squeezes / transposes:

- single volume: ``image_rfft (bp, h, w)`` -> ``volume (d, h, w)``
- multi-channel:  ``image_rfft (bp, bv, h, w)`` -> ``volumes (bv, d, h, w)``
  (poses shared across volumes, or per-volume via
  ``rotation_matrices (bv, bp, 3, 3)``).

``rotation_matrices``/``shifts_3d``/``shifts_2d`` and the Ewald arguments follow
the slice extraction in :mod:`.extraction`; the insertion applies the
*conjugate* shift phase ramps (the forward adjoint).
``weights`` is an optional real per-pixel tensor matching ``image_rfft``,
accumulated into a separate weight volume (returned alongside the data volume).

Central lines: 1D lines -> 3D volume
------------------------------------

The rfft-level adjoint of :func:`extract_central_lines_rfft_3d`: scatter 1D
central lines into a 3D rfft volume (Hermitian, DC at origin), optionally
accumulating per-sample weights for density compensation. This is the
insert-and-``irfft`` reconstruction path of the frame-free line graph.

Differentiable w.r.t. the input ``lines`` (adjoint line extraction),
``weights`` (weight-splat adjoint), ``directions`` (a per-node 3-vector
gradient) and ``shifts_3d``.

Two rank forms share one Mojo kernel; the Python layer only squeezes / transposes:

- single volume: ``lines (bp, w)`` -> ``volume (d, h, w)``
- multi-channel:  ``lines (bp, bv, w)`` -> ``volumes (bv, d, h, w)``
  (directions shared across volumes, or per-volume via ``(bv, bp, 3)``).

``directions`` are unit vectors (as in the extractor); the insertion applies
the *conjugate* 3D-shift phase ramp (the forward adjoint). ``weights`` is an
optional real per-sample tensor matching ``lines``, accumulated into a separate
weight volume.

Central lines: 1D lines -> 2D image
-----------------------------------

The adjoint of :func:`extract_central_lines_rfft_2d`: scatter 1D central lines into
a 2D rfft image (Hermitian, DC at origin), optionally accumulating per-sample
weights for density compensation. Reconstructs an image from its sinogram lines
(2D direct Fourier inversion). Differentiable w.r.t. the input ``lines``.

- single image:  ``lines (bp, w)`` -> ``image (h, w_rfft)``
- multi-image:   ``lines (bp, bv, w)`` -> ``images (bv, h, w_rfft)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._backend._common import reconstruction_volume_shape
from ._backend._line_2d import InsertLines2D
from ._backend._line_3d import InsertLines3D
from ._backend._slice_3d import InsertSlices3D
from ._conventions import ewald_coefficient, to_zyx_matrices, to_zyx_vectors

if TYPE_CHECKING:
    import torch


def _insert_slices_3d(
    image_rfft,
    weights,
    rotation_matrices,
    shifts_3d,
    shifts_2d,
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
    volume_sidelength = reconstruction_volume_shape(image_rfft.shape[-2], oversampling)[
        1
    ]
    return InsertSlices3D.apply(
        image_rfft,
        weights,
        to_zyx_matrices(rotation_matrices, zyx_matrices),
        to_zyx_vectors(shifts_2d, yx_shifts),
        to_zyx_vectors(shifts_3d, zyx_shifts),
        oversampling,
        fftfreq_max,
        interpolation,
        ewald_coefficient(
            volume_sidelength,
            apply_ewald_curvature,
            ewald_voltage_kv,
            ewald_flip_sign,
            ewald_px_size,
        ),
    )


def insert_central_slices_rfft_3d(
    image_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 2D central slices into one 3D rfft volume (Mojo scatter kernel).

    ``image_rfft`` is ``(bp, h, w)`` complex rfft slices (DC at origin); its device
    selects the backend. See the module docstring for the pose / weight parameters.

    Returns ``(volume, weight_volume)`` -- complex ``(d, h, w)`` accumulated data
    and real ``(d, h, w)`` accumulated weights (``None`` if ``weights`` is ``None``)
    -- on the input device.
    """
    if image_rfft.dim() != 3:
        raise ValueError(
            "image_rfft must be (bp, h, w) for a single volume; use "
            "insert_central_slices_rfft_3d_multichannel for (bp, bv, h, w)"
        )
    data, weight_vol = _insert_slices_3d(
        image_rfft,
        weights,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
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
    if weights is None:
        return data.squeeze(0), None
    return data.squeeze(0), weight_vol.squeeze(0)


def insert_central_slices_rfft_3d_multichannel(
    image_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 2D central slices into a batch of 3D rfft volumes (Mojo kernel).

    ``image_rfft`` is ``(bp, bv, h, w)`` (pose-major); ``weights`` (if given)
    matches it. Poses are shared (``rotation_matrices (bp, 3, 3)``) or per-volume
    (``(bv, bp, 3, 3)``). See the module docstring for the shared parameters.

    Returns ``(volumes, weight_volumes)`` -- complex ``(bv, d, h, w)`` data and
    real ``(bv, d, h, w)`` weights (``None`` if ``weights`` is ``None``) -- on the
    input device.
    """
    if image_rfft.dim() != 4:
        raise ValueError("image_rfft must be (bp, bv, h, w) for multi-channel")
    imgs = image_rfft.transpose(0, 1).contiguous()  # (bp, bv, ...) -> (bv, bp, ...)
    w = weights.transpose(0, 1).contiguous() if weights is not None else None
    data, weight_vol = _insert_slices_3d(
        imgs,
        w,
        rotation_matrices,
        shifts_3d,
        shifts_2d,
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
    return data, (weight_vol if weights is not None else None)


def _insert_lines_3d(
    lines,
    weights,
    directions,
    shifts_3d,
    oversampling,
    fftfreq_max,
    zyx_directions,
    zyx_shifts,
    interpolation,
):
    """Run the differentiable kernel in its canonical ``(bv, bp, w)`` layout."""
    return InsertLines3D.apply(
        lines,
        weights,
        to_zyx_vectors(directions, zyx_directions),
        to_zyx_vectors(shifts_3d, zyx_shifts),
        oversampling,
        fftfreq_max,
        interpolation,
    )


def insert_central_lines_rfft_3d(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    zyx_directions: bool = False,
    zyx_shifts: bool = False,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 1D central lines into one 3D rfft volume (Mojo scatter kernel).

    ``lines`` is ``(bp, w)`` complex rfft half-lines (DC at origin); its device
    selects the backend. ``directions`` are ``(3,)`` / ``(bp, 3)`` unit vectors
    and ``shifts_3d`` volume-frame shifts, both xyz unless ``zyx_directions`` /
    ``zyx_shifts`` is True. ``weights`` (optional, real,
    matching ``lines``) accumulate into a weight volume for density compensation.

    Returns ``(volume, weight_volume)`` -- complex ``(d, h, w)`` accumulated data
    and real ``(d, h, w)`` accumulated weights (``None`` if ``weights`` is
    ``None``) -- on the input device.
    """
    if lines.dim() != 2:
        raise ValueError(
            "lines must be (bp, w) for a single volume; use "
            "insert_central_lines_rfft_3d_multichannel for (bp, bv, w)"
        )
    data, weight_vol = _insert_lines_3d(
        lines,
        weights,
        directions,
        shifts_3d,
        oversampling,
        fftfreq_max,
        zyx_directions,
        zyx_shifts,
        interpolation,
    )
    if weights is None:
        return data.squeeze(0), None
    return data.squeeze(0), weight_vol.squeeze(0)


def insert_central_lines_rfft_3d_multichannel(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    zyx_directions: bool = False,
    zyx_shifts: bool = False,
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
        raise ValueError("lines must be (bp, bv, w) for multi-channel")
    lines_bv = lines.transpose(0, 1).contiguous()  # (bp, bv, w) -> (bv, bp, w)
    w = weights.transpose(0, 1).contiguous() if weights is not None else None
    data, weight_vol = _insert_lines_3d(
        lines_bv,
        w,
        directions,
        shifts_3d,
        oversampling,
        fftfreq_max,
        zyx_directions,
        zyx_shifts,
        interpolation,
    )
    return data, (weight_vol if weights is not None else None)


def _insert_lines_2d(
    lines,
    weights,
    directions,
    shifts_2d,
    oversampling,
    fftfreq_max,
    yx_directions,
    yx_shifts,
    interpolation,
):
    """Run the differentiable kernel in its canonical ``(bv, bp, w)`` layout."""
    return InsertLines2D.apply(
        lines,
        weights,
        to_zyx_vectors(directions, yx_directions),
        to_zyx_vectors(shifts_2d, yx_shifts),
        oversampling,
        fftfreq_max,
        interpolation,
    )


def insert_central_lines_rfft_2d(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 1D central lines into one 2D rfft image (Mojo scatter kernel).

    ``lines`` is complex ``(bp, w)``; ``directions`` are ``(2,)`` / ``(bp, 2)``
    unit vectors; ``shifts_2d`` optional ``(..., bp, 2)`` translations (the
    conjugate phase ramp is applied). Both are xy unless ``yx_directions`` /
    ``yx_shifts`` is True. Returns ``(image, weight_image)`` -- complex
    ``(h, w_rfft)`` and real weights (``None`` if ``weights`` is ``None``).
    """
    if lines.dim() != 2:
        raise ValueError(
            "lines must be (bp, w); use insert_central_lines_rfft_2d_multichannel"
        )
    data, wimg = _insert_lines_2d(
        lines,
        weights,
        directions,
        shifts_2d,
        oversampling,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
    )
    if weights is None:
        return data.squeeze(0), None
    return data.squeeze(0), wimg.squeeze(0)


def insert_central_lines_rfft_2d_multichannel(
    lines: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    yx_directions: bool = False,
    yx_shifts: bool = False,
    interpolation: str = "linear",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 1D central lines into a batch of 2D rfft images (Mojo kernel).

    ``lines`` is ``(bp, bv, w)`` (pose-major); ``weights`` (if given) matches it.
    Returns ``(images, weight_images)`` -- ``(bv, h, w_rfft)`` -- on input device.
    """
    if lines.dim() != 3:
        raise ValueError("lines must be (bp, bv, w) for multi-image")
    lines_bv = lines.transpose(0, 1).contiguous()  # (bp, bv, w) -> (bv, bp, w)
    w = weights.transpose(0, 1).contiguous() if weights is not None else None
    data, wimg = _insert_lines_2d(
        lines_bv,
        w,
        directions,
        shifts_2d,
        oversampling,
        fftfreq_max,
        yx_directions,
        yx_shifts,
        interpolation,
    )
    return data, (wimg if weights is not None else None)
