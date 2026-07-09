"""Experimental Mojo-backed 2D->3D Fourier-space backprojection.

The adjoint/transpose of the forward central-slice projection: scatters 2D rfft
slices into a 3D rfft volume (Hermitian, DC at origin), optionally accumulating
per-pixel weights for later Wiener-style normalization.

The backend follows the input device (GPU inputs are materialised on the CPU and
the result returned on the input device). Differentiable w.r.t. ``projections``,
``weights``, ``rotations`` and ``shifts_2d``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._autograd import BackprojectForward

if TYPE_CHECKING:
    import torch


def backproject_2d_to_3d_forw(
    projections: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
    ewald_curvature: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Backproject 2D rfft slices into a 3D rfft volume (Mojo scatter kernel).

    The pose arguments are ordered ``rotations, shifts_3d, shifts_2d`` to mirror
    the forward projector this is the adjoint of.

    Parameters
    ----------
    projections : torch.Tensor
        Complex ``(bp, h, w)`` or ``(bv, bp, h, w)`` rfft slices (DC at origin),
        square with even side (``w = h//2+1``).
    rotations : torch.Tensor
        Real ``(3, 3)``, ``(bp, 3, 3)`` or ``(bv_rot, bp, 3, 3)`` zyx rotation
        matrices (same convention as the forward projector).
    shifts_3d : torch.Tensor | None
        Optional ``(..., P, 3)`` 3D shifts (zyx, volume frame); the conjugate
        phase ramp is applied (adjoint of the forward 3D shift).
    shifts_2d : torch.Tensor | None
        Optional ``(..., P, 2)`` 2D shifts (yx); the conjugate phase ramp is
        applied (adjoint of the forward shift).
    weights : torch.Tensor | None
        Optional real per-pixel weights (e.g. CTF^2) matching ``projections``;
        accumulated into a separate weight volume for downstream normalization.
    oversampling : float
        Coordinate scaling factor; the output volume is cubic with side
        ``ceil(h * oversampling)`` rounded up to even.
    fourier_radius_cutoff : float | None
        Frequency radius (cycles) beyond which input pixels are ignored.
        Defaults to ``h / 2`` (Nyquist).
    interpolation : str
        ``"linear"`` (trilinear, default) or ``"cubic"`` (tricubic Catmull-Rom).
    ewald_curvature : float
        Signed Ewald-sphere curvature coefficient. ``0.0`` (default) keeps the
        slice flat; positive / negative bends it onto the sphere (slice z-offset
        ``ewald_curvature * |k_xy|^2``). Must match the value used in the forward
        projection it is the adjoint of.

    Returns
    -------
    (data_volume, weight_volume) : complex ``(bv, d, h, w)`` accumulated data,
    and real ``(bv, d, h, w)`` accumulated weights if ``weights`` was given
    (else ``None``). On the same device as ``projections``.
    """
    # The autograd Function is the functional entry point: its forward runs the
    # scatter kernel, its backward the forward-projection + pose/shift/weight
    # kernels. torch builds no graph when nothing requires grad, so this is also
    # the non-differentiable path.
    data_vol, weight_vol = BackprojectForward.apply(
        projections,
        weights,
        rotations,
        shifts_2d,
        shifts_3d,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
        ewald_curvature,
    )
    return data_vol, (weight_vol if weights is not None else None)
