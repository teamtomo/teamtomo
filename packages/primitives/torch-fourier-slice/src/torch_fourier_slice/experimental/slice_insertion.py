"""Experimental Mojo-backed central-slice insertion (rfft layer).

The rfft-level adjoint operator: scatter 2D central slices into a 3D rfft volume
(Hermitian, DC at origin), optionally accumulating per-pixel weights for a
Wiener-style normalization. This is the experimental analogue of
:func:`torch_fourier_slice.insert_central_slices_rfft_3d` and is differentiable
w.r.t. the slices, weights, rotations, and 2D / 3D shifts.

Two rank forms share one Mojo kernel; the Python layer only squeezes / transposes:

- single volume: ``image_rfft (bp, h, w)`` -> ``volume (d, h, w)``
- multi-volume:  ``image_rfft (bp, bv, h, w)`` -> ``volumes (bv, d, h, w)``
  (poses shared across volumes, or per-volume via ``rotations (bv, bp, 3, 3)``).

``rotations``/``shifts_3d``/``shifts_2d`` follow the extraction module; the
backprojection applies the *conjugate* shift phase ramps (the forward adjoint).
``weights`` is an optional real per-pixel tensor matching ``image_rfft``,
accumulated into a separate weight volume (returned alongside the data volume).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._autograd import BackprojectForward

if TYPE_CHECKING:
    import torch


def _insert(
    image_rfft,
    weights,
    rotations,
    shifts_3d,
    shifts_2d,
    oversampling,
    fourier_radius_cutoff,
    interpolation,
    ewald_curvature,
):
    """Run the differentiable kernel in its canonical ``(bv, bp, ...)`` layout."""
    return BackprojectForward.apply(
        image_rfft,
        weights,
        rotations,
        shifts_2d,
        shifts_3d,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
        ewald_curvature,
    )


def insert_central_slices_rfft_3d(
    image_rfft: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
    ewald_curvature: float = 0.0,
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
            "insert_central_slices_rfft_3d_multivolume for (bp, bv, h, w)"
        )
    data, weight_vol = _insert(
        image_rfft, weights, rotations, shifts_3d, shifts_2d,
        oversampling, fourier_radius_cutoff, interpolation, ewald_curvature,
    )
    if weights is None:
        return data.squeeze(0), None
    return data.squeeze(0), weight_vol.squeeze(0)


def insert_central_slices_rfft_3d_multivolume(
    image_rfft: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
    ewald_curvature: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert 2D central slices into a batch of 3D rfft volumes (Mojo kernel).

    ``image_rfft`` is ``(bp, bv, h, w)`` (pose-major); ``weights`` (if given)
    matches it. Poses are shared (``rotations (bp, 3, 3)``) or per-volume
    (``(bv, bp, 3, 3)``). See the module docstring for the shared parameters.

    Returns ``(volumes, weight_volumes)`` -- complex ``(bv, d, h, w)`` data and
    real ``(bv, d, h, w)`` weights (``None`` if ``weights`` is ``None``) -- on the
    input device.
    """
    if image_rfft.dim() != 4:
        raise ValueError("image_rfft must be (bp, bv, h, w) for multi-volume")
    imgs = image_rfft.transpose(0, 1).contiguous()  # (bp, bv, ...) -> (bv, bp, ...)
    w = weights.transpose(0, 1).contiguous() if weights is not None else None
    data, weight_vol = _insert(
        imgs, w, rotations, shifts_3d, shifts_2d,
        oversampling, fourier_radius_cutoff, interpolation, ewald_curvature,
    )
    return data, (weight_vol if weights is not None else None)
