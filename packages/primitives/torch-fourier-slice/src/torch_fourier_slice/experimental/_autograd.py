"""torch.autograd.Function wrappers for the differentiable projectors.

The forward 3D->2D projection and the 2D->3D backprojection are adjoints of one
another, so each one's *data* gradient is the other's kernel:

- d/d(volume) of forward projection  = scatter (pure adjoint) of grad_output
- d/d(slices) of backprojection       = forward projection of grad_output

Gradients w.r.t. ``rotations`` and ``shifts_2d`` (and ``weights`` for the
backprojection) are computed by dedicated backward kernels: the rotation grad
chains the analytical spatial gradient of the interpolated field through the
rotated sample coordinate; the shift grad differentiates the phase ramp; the
weight grad is the adjoint of the weight splat.
"""

from __future__ import annotations

import torch

from ._ops import (
    backproject_scatter,
    forward_project,
    run_pose_grad,
    run_scatter,
    run_weight_grad,
)


def _reduce_to(grad: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum a per-(bv_rot/bv_shift_2d, bp, ...) grad down to ``target``'s shape.

    The pose tensors broadcast over the volume batch (leading size-1 dims and/or
    ``bv_rot/bv_shift_2d == 1``); the corresponding gradient sums those axes.
    """
    while grad.dim() > target.dim():
        grad = grad.sum(0)
    for dim in range(grad.dim()):
        if target.shape[dim] == 1 and grad.shape[dim] != 1:
            grad = grad.sum(dim, keepdim=True)
    return grad.reshape(target.shape).to(device=target.device, dtype=target.dtype)


class ProjectForward(torch.autograd.Function):
    """Differentiable forward 3D->2D projection (volume, rotations, shifts_2d)."""

    @staticmethod
    def forward(
        ctx,
        reconstruction: torch.Tensor,
        rotations: torch.Tensor,
        shifts_2d: torch.Tensor | None,
        shifts_3d: torch.Tensor | None,
        output_shape: tuple[int, int] | None,
        oversampling: float,
        fourier_radius_cutoff: float | None,
        interpolation: str,
        ewald_curvature: float = 0.0,
    ) -> torch.Tensor:
        proj = forward_project(
            reconstruction,
            rotations,
            shifts_2d,
            output_shape,
            oversampling,
            fourier_radius_cutoff,
            interpolation,
            ewald_curvature,
            shifts_3d,
        )
        rec4d = reconstruction
        if reconstruction.dim() == 3:
            rec4d = reconstruction.unsqueeze(0)
        ctx.vol_shape = (rec4d.shape[1], rec4d.shape[2], rec4d.shape[3])
        ctx.reconstruction = reconstruction
        ctx.rotations = rotations
        ctx.shifts_2d = shifts_2d
        ctx.shifts_3d = shifts_3d
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        ctx.interpolation = interpolation
        ctx.ewald_curvature = ewald_curvature
        ctx.input_dim = reconstruction.dim()
        return proj

    @staticmethod
    def backward(ctx, grad_proj: torch.Tensor):
        needs = ctx.needs_input_grad
        grad_proj = grad_proj.contiguous()
        grad_rec = None
        if needs[0]:
            # adjoint of the forward projection: scatter grad back into the volume
            # (pure transpose, no x=0 skip / Hermitian double-insert).
            grad_rec, _ = run_scatter(
                grad_proj,
                ctx.rotations,
                ctx.vol_shape,
                shifts_2d=ctx.shifts_2d,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
                friedel_double=False,
                skip_redundant=False,
                ewald_curvature=ctx.ewald_curvature,
                shifts_3d=ctx.shifts_3d,
            )
            if ctx.input_dim == 3:
                grad_rec = grad_rec[0]

        grad_rot = None
        grad_shift = None
        grad_shift_3d = None
        if needs[1] or needs[2] or needs[3]:
            gr, gs, gs3 = run_pose_grad(
                ctx.reconstruction,
                ctx.rotations,
                ctx.shifts_2d,
                grad_proj,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
                backproject=False,
                ewald_curvature=ctx.ewald_curvature,
                shifts_3d=ctx.shifts_3d,
            )
            if needs[1]:
                grad_rot = _reduce_to(gr, ctx.rotations)
            if needs[2] and ctx.shifts_2d is not None and gs is not None:
                grad_shift = _reduce_to(gs, ctx.shifts_2d)
            if needs[3] and ctx.shifts_3d is not None and gs3 is not None:
                grad_shift_3d = _reduce_to(gs3, ctx.shifts_3d)

        return (
            grad_rec,
            grad_rot,
            grad_shift,
            grad_shift_3d,
            None,
            None,
            None,
            None,
            None,
        )


class BackprojectForward(torch.autograd.Function):
    """Differentiable 2D->3D backprojection (projections, weights, rotations, shifts).

    The data-gradient is the exact transpose of the (friedel_double +
    skip_redundant) scatter: symmetrise the volume gradient on the kx=0 plane,
    forward-project it, then zero the gradient of the skipped redundant x=0 line.
    The same symmetrised volume gradient feeds the rotation/shift backward kernel.
    """

    @staticmethod
    def forward(
        ctx,
        projections: torch.Tensor,
        weights: torch.Tensor | None,
        rotations: torch.Tensor,
        shifts_2d: torch.Tensor | None,
        shifts_3d: torch.Tensor | None,
        oversampling: float,
        fourier_radius_cutoff: float | None,
        interpolation: str,
        ewald_curvature: float = 0.0,
    ):
        data_vol, weight_vol = backproject_scatter(
            projections,
            rotations,
            weights,
            shifts_2d,
            oversampling,
            fourier_radius_cutoff,
            interpolation,
            ewald_curvature,
            shifts_3d,
        )
        p4 = projections.unsqueeze(0) if projections.dim() == 3 else projections
        ctx.proj_sidelength = int(p4.shape[2])
        ctx.projections = projections
        ctx.weights = weights
        ctx.rotations = rotations
        ctx.shifts_2d = shifts_2d
        ctx.shifts_3d = shifts_3d
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        ctx.interpolation = interpolation
        ctx.ewald_curvature = ewald_curvature
        ctx.input_dim = projections.dim()
        if weight_vol is None:
            weight_vol = torch.empty(0, device=data_vol.device)
        return data_vol, weight_vol

    @staticmethod
    def backward(ctx, grad_data: torch.Tensor, grad_weight: torch.Tensor):
        needs = ctx.needs_input_grad
        side = ctx.proj_sidelength

        # Symmetrise the kx=0 plane: the adjoint of the Hermitian double-insert.
        # self-mirror points (z,y in {0, N/2}) were inserted once, not doubled.
        g = None
        if needs[0] or needs[2] or needs[3] or needs[4]:
            g = grad_data.contiguous().clone()
            plane = grad_data[..., 0]
            mirror = torch.conj(
                plane.flip(dims=(-2, -1)).roll(shifts=(1, 1), dims=(-2, -1))
            )
            self_mask = torch.zeros_like(plane, dtype=torch.bool)
            d, h = plane.shape[-2], plane.shape[-1]
            for zi in (0, d // 2):
                for yi in (0, h // 2):
                    self_mask[..., zi, yi] = True
            g[..., 0] = plane + torch.where(self_mask, plane.new_zeros(()), mirror)

        grad_proj = None
        if needs[0]:
            grad_proj = forward_project(
                g,
                ctx.rotations,
                ctx.shifts_2d,
                (side, side),
                ctx.oversampling,
                ctx.fourier_radius_cutoff,
                ctx.interpolation,
                ctx.ewald_curvature,
                ctx.shifts_3d,
            ).clone()
            # adjoint of skip_redundant: skipped input pixels contributed nothing
            grad_proj[..., side // 2 :, 0] = 0
            if ctx.input_dim == 3:
                grad_proj = grad_proj[0]

        grad_weights = None
        if needs[1] and ctx.weights is not None and grad_weight is not None:
            gw = run_weight_grad(
                grad_weight,
                ctx.rotations,
                side,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
                ewald_curvature=ctx.ewald_curvature,
            )
            if ctx.input_dim == 3:
                gw = gw[0]
            grad_weights = gw.to(device=ctx.weights.device, dtype=ctx.weights.dtype)

        grad_rot = None
        grad_shift = None
        grad_shift_3d = None
        if needs[2] or needs[3] or needs[4]:
            gr, gs, gs3 = run_pose_grad(
                g,
                ctx.rotations,
                ctx.shifts_2d,
                ctx.projections,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
                backproject=True,
                ewald_curvature=ctx.ewald_curvature,
                shifts_3d=ctx.shifts_3d,
            )
            if needs[2]:
                grad_rot = _reduce_to(gr, ctx.rotations)
            if needs[3] and ctx.shifts_2d is not None and gs is not None:
                grad_shift = _reduce_to(gs, ctx.shifts_2d)
            if needs[4] and ctx.shifts_3d is not None and gs3 is not None:
                grad_shift_3d = _reduce_to(gs3, ctx.shifts_3d)

        return (
            grad_proj,
            grad_weights,
            grad_rot,
            grad_shift,
            grad_shift_3d,
            None,
            None,
            None,
            None,
        )
