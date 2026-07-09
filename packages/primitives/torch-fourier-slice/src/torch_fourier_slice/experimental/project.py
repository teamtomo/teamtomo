"""Experimental Mojo-backed forward 3D->2D Fourier-slice projection.

The forward 3D->2D Fourier-slice projection, with the
inner loop written in Mojo (see ``_mojo/projectors.mojo``) and exposed to Python
via Mojo's Python interop. It operates on volumes in **rfft layout, DC at origin**
(unshifted).

This differs from :func:`torch_fourier_slice.extract_central_slices_rfft_3d`
*only* in array layout: that function expects an ``fftshift``ed rfft (DC centered
on the z/y axes). Bridge between them with a single ``torch.fft.fftshift`` /
``ifftshift`` over the non-redundant dims. Within the Nyquist band the two
produce identical results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._autograd import ProjectForward

if TYPE_CHECKING:
    import torch


def project_3d_to_2d(
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts_3d: torch.Tensor | None = None,
    shifts_2d: torch.Tensor | None = None,
    output_shape: tuple[int, int] | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
    ewald_curvature: float = 0.0,
) -> torch.Tensor:
    """Forward project a 3D rfft volume to 2D central slices (Mojo kernel).

    The pose arguments are ordered ``rotations, shifts_3d, shifts_2d``. The
    operations they parametrise compose as: a 3D shift in the volume frame
    (``shifts_3d``, applied before the rotation), then the ``rotations``, then a
    2D shift in the projection plane (``shifts_2d``, applied after).

    The compute backend follows ``reconstruction.device``: a CPU tensor runs a
    multithreaded Mojo CPU kernel; a GPU tensor (``mps`` / ``cuda``) runs a
    Mojo GPU kernel. Output is returned on the same device as ``reconstruction``.

    Parameters
    ----------
    reconstruction : torch.Tensor
        Complex 3D Fourier volume in rfft layout (DC at origin), shape
        ``(d, h, w)`` or batched ``(bv, d, h, w)`` where ``w = h//2+1``. ``h`` must
        be even and equal to ``2*(w - 1)`` (cubic, even side). Its device selects
        the backend.
    rotations : torch.Tensor
        Real ``(3, 3)``, ``(bp, 3, 3)`` or ``(bv_rot, bp, 3, 3)`` zyx rotation
        matrices. ``bv_rot`` must be 1 or match ``bv``.
    shifts_3d : torch.Tensor | None
        Optional real ``(..., P, 3)`` 3D shifts (zyx) in the volume reference
        frame, applied *before* rotation (as a phase ramp on the rotated sample
        coordinate). ``None`` (default) applies no 3D shift.
    shifts_2d : torch.Tensor | None
        Optional real ``(..., P, 2)`` 2D shifts (yx) in the projection plane,
        applied as a Fourier-space phase ramp *after* rotation. ``None`` (default)
        applies no shift.
    output_shape : tuple[int, int] | None
        ``(H_out, W_out)`` of the projection (square, even). Defaults to
        ``(H, H)``.
    oversampling : float
        Coordinate scaling factor (>1 oversamples the volume). Default 1.0.
    fourier_radius_cutoff : float | None
        Frequency radius (in cycles, i.e. ``fftfreq * sidelength``) beyond which
        output pixels are left at zero. Defaults to ``output_sidelength / 2``
        (Nyquist).
    interpolation : str
        ``"linear"`` (trilinear, default) or ``"cubic"`` (tricubic Catmull-Rom).
    ewald_curvature : float
        Signed Ewald-sphere curvature coefficient. ``0.0`` (default) keeps the
        central slice flat; a positive / negative value bends it onto the sphere
        (the slice z-offset is ``ewald_curvature * |k_xy|^2``). The magnitude
        folds the wavelength / pixel-size constants; the sign selects the
        diffraction half (none / positive / negative).

    Differentiable w.r.t. ``reconstruction`` (adjoint = 2D->3D scatter),
    ``rotations`` and ``shifts_2d`` (dedicated backward kernels), for both
    interpolations and on CPU/GPU.

    Returns
    -------
    projections : torch.Tensor
        Complex ``(bv, bp, h_out, w_out)`` central slices in rfft layout (DC at
        origin), on the same device as ``reconstruction``.
    """
    # The autograd Function is the functional entry point: its forward runs the
    # projection kernel, its backward the adjoint (scatter) + pose/shift kernels.
    # torch builds no graph when nothing requires grad, so this is also the
    # non-differentiable path.
    return ProjectForward.apply(
        reconstruction,
        rotations,
        shifts_2d,
        shifts_3d,
        output_shape,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
        ewald_curvature,
    )
