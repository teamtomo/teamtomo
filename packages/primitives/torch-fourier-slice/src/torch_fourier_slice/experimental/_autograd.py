"""torch.autograd.Function wrappers for the differentiable Fourier-slice ops.

Every extraction and its matching insertion are adjoints of one another, so each
one's *data* gradient is the other's kernel:

- d/d(volume) of an extraction = the scatter (pure adjoint) of grad_output
- d/d(slices)  of an insertion = the gather of grad_output

The same pairing holds for the central-slice ops (``ProjectForward`` /
``BackprojectForward``), the 3D central-line ops (``ProjectLineForward`` /
``InsertLineForward``) and the 2D central-line ops (``ProjectLine2DForward`` /
``InsertLine2DForward``).

The insertions used for reconstruction also Hermitian double-insert on the kx=0
plane, whose adjoint is a symmetrisation of the volume/image gradient before the
gather (see :func:`_symmetrise_kx0_plane` / :func:`_symmetrise_kx0_column`).

Gradients w.r.t. the pose (``rotations`` or ``directions``), the 2D / 3D shifts
and the insertion ``weights`` come from dedicated backward kernels: the pose grad
chains the analytical spatial gradient of the interpolated field through the
rotated sample coordinate; the shift grad differentiates the phase ramp; the
weight grad is the adjoint of the weight splat.
"""

from __future__ import annotations

import torch

from ._ops import (
    backproject_scatter,
    forward_project,
    forward_project_line,
    forward_project_line_2d,
    reconstruction_volume_shape,
    run_line2d_pose_grad,
    run_line2d_weight_grad,
    run_line_pose_grad,
    run_line_weight_grad,
    run_pose_grad,
    run_scatter,
    run_scatter_line,
    run_scatter_line_2d,
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


def _symmetrise_kx0_plane(grad_volume: torch.Tensor) -> torch.Tensor:
    """Adjoint of the Hermitian double-insert on a volume's kx=0 plane.

    The insertion writes each kx=0 sample and its (-z, -y) conjugate mirror, so
    the transpose adds the mirror's cotangent back onto each sample. Self-mirror
    points (z, y each 0 or N/2) map to themselves and were inserted once, not
    doubled, so they are excluded.
    """
    out = grad_volume.contiguous().clone()
    plane = grad_volume[..., 0]
    mirror = torch.conj(plane.flip(dims=(-2, -1)).roll(shifts=(1, 1), dims=(-2, -1)))
    self_mask = torch.zeros_like(plane, dtype=torch.bool)
    d, h = plane.shape[-2], plane.shape[-1]
    for zi in (0, d // 2):
        for yi in (0, h // 2):
            self_mask[..., zi, yi] = True
    out[..., 0] = plane + torch.where(self_mask, plane.new_zeros(()), mirror)
    return out


def _symmetrise_kx0_column(grad_image: torch.Tensor) -> torch.Tensor:
    """Adjoint of the Hermitian double-insert on an image's kx=0 column.

    The 2D analogue of :func:`_symmetrise_kx0_plane`: add the ky -> (h - ky)
    conjugate mirror, leaving the self-mirror rows (0 and h/2) single.
    """
    out = grad_image.contiguous().clone()
    column = grad_image[..., 0]
    mirror = torch.conj(column.flip(-1).roll(1, -1))
    h = column.shape[-1]
    self_mask = torch.zeros(h, dtype=torch.bool, device=column.device)
    self_mask[0] = True
    self_mask[h // 2] = True
    out[..., 0] = column + torch.where(self_mask, column.new_zeros(()), mirror)
    return out


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


class ProjectLineForward(torch.autograd.Function):
    """Differentiable forward 3D->1D central-line projection.

    Differentiable w.r.t. the ``reconstruction`` (adjoint line scatter), the
    ``directions`` (a per-node 3-vector gradient) and ``shifts_3d`` (phase-ramp
    gradient) -- the latter two via the line pose-gradient kernel.
    """

    @staticmethod
    def forward(
        ctx,
        reconstruction: torch.Tensor,
        directions: torch.Tensor,
        shifts_3d: torch.Tensor | None,
        output_length: int | None,
        oversampling: float,
        fourier_radius_cutoff: float | None,
        interpolation: str,
    ) -> torch.Tensor:
        line = forward_project_line(
            reconstruction,
            directions,
            output_length,
            oversampling,
            fourier_radius_cutoff,
            interpolation,
            shifts_3d,
        )
        rec4d = reconstruction
        if reconstruction.dim() == 3:
            rec4d = reconstruction.unsqueeze(0)
        ctx.vol_shape = (rec4d.shape[1], rec4d.shape[2], rec4d.shape[3])
        ctx.reconstruction = reconstruction
        ctx.directions = directions
        ctx.shifts_3d = shifts_3d
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        ctx.interpolation = interpolation
        ctx.input_dim = reconstruction.dim()
        return line

    @staticmethod
    def backward(ctx, grad_line: torch.Tensor):
        needs = ctx.needs_input_grad
        grad_line = grad_line.contiguous()
        grad_rec = None
        if needs[0]:
            # adjoint of the forward line projection: scatter grad back into the
            # volume (pure transpose, no Hermitian double-insert).
            grad_rec, _ = run_scatter_line(
                grad_line,
                ctx.directions,
                ctx.vol_shape,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
                friedel_double=False,
                shifts_3d=ctx.shifts_3d,
            )
            if ctx.input_dim == 3:
                grad_rec = grad_rec[0]

        grad_dir = None
        grad_shift_3d = None
        if needs[1] or needs[2]:
            gd, gs3 = run_line_pose_grad(
                ctx.reconstruction,
                ctx.directions,
                grad_line,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
                backproject=False,
                shifts_3d=ctx.shifts_3d,
            )
            if needs[1]:
                grad_dir = _reduce_to(gd, ctx.directions)
            if needs[2] and ctx.shifts_3d is not None and gs3 is not None:
                grad_shift_3d = _reduce_to(gs3, ctx.shifts_3d)
        return (grad_rec, grad_dir, grad_shift_3d, None, None, None, None)


class InsertLineForward(torch.autograd.Function):
    """Differentiable 1D->3D central-line insertion (reconstruction).

    Hermitian double-inserts on the volume's kx=0 plane. Differentiable w.r.t.
    the input ``lines`` (adjoint forward line projection of the kx=0-symmetrised
    volume gradient), ``weights`` (weight-splat adjoint), ``directions`` and
    ``shifts_3d`` (line pose-gradient kernel).
    """

    @staticmethod
    def forward(
        ctx,
        lines: torch.Tensor,
        weights: torch.Tensor | None,
        directions: torch.Tensor,
        shifts_3d: torch.Tensor | None,
        oversampling: float,
        fourier_radius_cutoff: float | None,
        interpolation: str,
    ):
        lines4 = lines.unsqueeze(0) if lines.dim() == 2 else lines
        line_sidelength = 2 * (int(lines4.shape[2]) - 1)
        data_vol, weight_vol = run_scatter_line(
            lines,
            directions,
            reconstruction_volume_shape(line_sidelength, oversampling),
            weights=weights,
            oversampling=oversampling,
            fourier_radius_cutoff=fourier_radius_cutoff,
            interpolation=interpolation,
            friedel_double=True,
            shifts_3d=shifts_3d,
        )
        ctx.line_sidelength = line_sidelength
        ctx.lines = lines
        ctx.weights = weights
        ctx.directions = directions
        ctx.shifts_3d = shifts_3d
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        ctx.interpolation = interpolation
        ctx.input_dim = lines.dim()
        if weight_vol is None:
            weight_vol = torch.empty(0, device=data_vol.device)
        return data_vol, weight_vol

    @staticmethod
    def backward(ctx, grad_data: torch.Tensor, grad_weight: torch.Tensor):
        needs = ctx.needs_input_grad

        g = None
        if needs[0] or needs[2] or needs[3]:
            g = _symmetrise_kx0_plane(grad_data)

        grad_lines = None
        if needs[0]:
            grad_lines = forward_project_line(
                g,
                ctx.directions,
                ctx.line_sidelength,
                ctx.oversampling,
                ctx.fourier_radius_cutoff,
                ctx.interpolation,
                ctx.shifts_3d,
            ).clone()
            if ctx.input_dim == 2:
                grad_lines = grad_lines[0]

        grad_weights = None
        if needs[1] and ctx.weights is not None and grad_weight is not None:
            gw = run_line_weight_grad(
                grad_weight,
                ctx.directions,
                ctx.line_sidelength,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
            )
            if ctx.input_dim == 2:
                gw = gw[0]
            grad_weights = gw.to(device=ctx.weights.device, dtype=ctx.weights.dtype)

        grad_dir = None
        grad_shift_3d = None
        if needs[2] or needs[3]:
            gd, gs3 = run_line_pose_grad(
                g,
                ctx.directions,
                ctx.lines,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
                backproject=True,
                shifts_3d=ctx.shifts_3d,
            )
            if needs[2]:
                grad_dir = _reduce_to(gd, ctx.directions)
            if needs[3] and ctx.shifts_3d is not None and gs3 is not None:
                grad_shift_3d = _reduce_to(gs3, ctx.shifts_3d)
        return (grad_lines, grad_weights, grad_dir, grad_shift_3d, None, None, None)


class ProjectLine2DForward(torch.autograd.Function):
    """Differentiable forward 2D->1D central-line projection.

    Differentiable w.r.t. the ``image_rfft`` (adjoint = 1D->2D line scatter), the
    ``directions`` (a per-node 2-vector gradient) and ``shifts_2d`` (phase ramp).
    """

    @staticmethod
    def forward(
        ctx,
        image_rfft: torch.Tensor,
        directions: torch.Tensor,
        shifts_2d: torch.Tensor | None,
        output_length: int | None,
        oversampling: float,
        fourier_radius_cutoff: float | None,
        interpolation: str,
    ) -> torch.Tensor:
        line = forward_project_line_2d(
            image_rfft,
            directions,
            output_length,
            oversampling,
            fourier_radius_cutoff,
            interpolation,
            shifts_2d,
        )
        img3 = image_rfft.unsqueeze(0) if image_rfft.dim() == 2 else image_rfft
        ctx.image_shape = (int(img3.shape[-2]), int(img3.shape[-1]))
        ctx.image_rfft = image_rfft
        ctx.directions = directions
        ctx.shifts_2d = shifts_2d
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        ctx.interpolation = interpolation
        ctx.input_dim = image_rfft.dim()
        return line

    @staticmethod
    def backward(ctx, grad_line: torch.Tensor):
        needs = ctx.needs_input_grad
        grad_line = grad_line.contiguous()
        grad_img = None
        if needs[0]:
            grad_img, _ = run_scatter_line_2d(
                grad_line,
                ctx.directions,
                ctx.image_shape,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
                friedel_double=False,
                shifts_2d=ctx.shifts_2d,
            )
            if ctx.input_dim == 2:
                grad_img = grad_img[0]

        grad_dir = None
        grad_shift = None
        if needs[1] or needs[2]:
            gd, gs = run_line2d_pose_grad(
                ctx.image_rfft,
                ctx.directions,
                grad_line,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
                backproject=False,
                shifts_2d=ctx.shifts_2d,
            )
            if needs[1]:
                grad_dir = _reduce_to(gd, ctx.directions)
            if needs[2] and ctx.shifts_2d is not None and gs is not None:
                grad_shift = _reduce_to(gs, ctx.shifts_2d)
        return (grad_img, grad_dir, grad_shift, None, None, None, None)


class InsertLine2DForward(torch.autograd.Function):
    """Differentiable 1D->2D central-line insertion (Hermitian, kx=0 double-insert).

    Differentiable w.r.t. the input ``lines`` (adjoint forward 2D line projection
    of the kx=0-symmetrised image gradient), ``weights`` (weight-splat adjoint),
    ``directions`` and ``shifts_2d`` (2D pose-grad kernel).
    """

    @staticmethod
    def forward(
        ctx,
        lines: torch.Tensor,
        weights: torch.Tensor | None,
        directions: torch.Tensor,
        shifts_2d: torch.Tensor | None,
        oversampling: float,
        fourier_radius_cutoff: float | None,
        interpolation: str,
    ):
        lines3 = lines.unsqueeze(0) if lines.dim() == 2 else lines
        line_sidelength = 2 * (int(lines3.shape[2]) - 1)
        image_shape = (line_sidelength, line_sidelength // 2 + 1)
        img, wimg = run_scatter_line_2d(
            lines,
            directions,
            image_shape,
            weights=weights,
            oversampling=oversampling,
            fourier_radius_cutoff=fourier_radius_cutoff,
            interpolation=interpolation,
            friedel_double=True,
            shifts_2d=shifts_2d,
        )
        ctx.line_sidelength = line_sidelength
        ctx.lines = lines
        ctx.weights = weights
        ctx.directions = directions
        ctx.shifts_2d = shifts_2d
        ctx.oversampling = oversampling
        ctx.fourier_radius_cutoff = fourier_radius_cutoff
        ctx.interpolation = interpolation
        ctx.input_dim = lines.dim()
        if wimg is None:
            wimg = torch.empty(0, device=img.device)
        return img, wimg

    @staticmethod
    def backward(ctx, grad_img: torch.Tensor, grad_weight: torch.Tensor):
        needs = ctx.needs_input_grad

        g = None
        if needs[0] or needs[2] or needs[3]:
            g = _symmetrise_kx0_column(grad_img)

        grad_lines = None
        if needs[0]:
            grad_lines = forward_project_line_2d(
                g,
                ctx.directions,
                ctx.line_sidelength,
                ctx.oversampling,
                ctx.fourier_radius_cutoff,
                ctx.interpolation,
                ctx.shifts_2d,
            ).clone()
            if ctx.input_dim == 2:
                grad_lines = grad_lines[0]

        grad_weights = None
        if needs[1] and ctx.weights is not None and grad_weight is not None:
            gw = run_line2d_weight_grad(
                grad_weight,
                ctx.directions,
                ctx.line_sidelength,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
            )
            if ctx.input_dim == 2:
                gw = gw[0]
            grad_weights = gw.to(device=ctx.weights.device, dtype=ctx.weights.dtype)

        grad_dir = None
        grad_shift = None
        if needs[2] or needs[3]:
            gd, gs = run_line2d_pose_grad(
                g,
                ctx.directions,
                ctx.lines,
                oversampling=ctx.oversampling,
                fourier_radius_cutoff=ctx.fourier_radius_cutoff,
                interpolation=ctx.interpolation,
                backproject=True,
                shifts_2d=ctx.shifts_2d,
            )
            if needs[2]:
                grad_dir = _reduce_to(gd, ctx.directions)
            if needs[3] and ctx.shifts_2d is not None and gs is not None:
                grad_shift = _reduce_to(gs, ctx.shifts_2d)
        return (grad_lines, grad_weights, grad_dir, grad_shift, None, None, None)


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

        g = None
        if needs[0] or needs[2] or needs[3] or needs[4]:
            g = _symmetrise_kx0_plane(grad_data)

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
