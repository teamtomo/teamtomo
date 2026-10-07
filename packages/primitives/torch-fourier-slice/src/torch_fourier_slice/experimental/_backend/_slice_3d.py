"""3D central-slice backend: 3D volume <-> 2D central slices.

The ``run_*`` ops validate inputs, build device buffers and dispatch to the CPU
or GPU Mojo kernel; ``ExtractSlices3D`` / ``InsertSlices3D`` wire them into
autograd.
"""

from __future__ import annotations

import torch

from ._common import reconstruction_volume_shape, reduce_to, symmetrise_kx0_plane
from ._device import prepare_launch
from ._loader import device_session, kernels
from ._validation import (
    KernelParams,
    interp_code,
    prep_rotations,
    prep_shifts_2d,
    prep_shifts_3d,
    validate_reconstruction,
)


def run_extract_slices_3d(
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    output_shape: tuple[int, int] | None,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
    ewald_curvature: float = 0.0,
    shifts_3d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract 2D central slices from an rfft volume (CPU or GPU per device)."""
    device = reconstruction.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    reconstruction, bv, sidelength, _sh = validate_reconstruction(reconstruction)
    rot, _bv_rot, bp = prep_rotations(rotations, bv, tgt)
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, tgt)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, tgt)

    if output_shape is None:
        proj_sidelength = sidelength
    else:
        if len(output_shape) != 2 or output_shape[0] != output_shape[1]:
            raise ValueError(f"output_shape {output_shape} must be square")
        if output_shape[0] % 2 != 0:
            raise ValueError(f"output side length {output_shape[0]} must be even")
        proj_sidelength = int(output_shape[0])
    proj_sidelength_half = proj_sidelength // 2 + 1

    radius = (
        proj_sidelength / 2.0
        if fftfreq_max is None
        else float(fftfreq_max) * proj_sidelength
    )
    # pre-zeroed: the kernel leaves radius-cut pixels untouched
    proj_r = torch.zeros(
        bv,
        bp,
        proj_sidelength,
        proj_sidelength_half,
        2,
        dtype=torch.float32,
        device=tgt,
    )
    params = KernelParams(
        oversampling=float(oversampling),
        radius_cutoff_sq=float(radius * radius),
        has_shifts_2d=int(has_shifts_2d),
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        interp=interp_code(interpolation),
        ewald_curvature=float(ewald_curvature),
        has_shifts_3d=int(has_shifts_3d),
    )

    rec_r = torch.view_as_real(
        reconstruction.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    bufs = (rec_r, rot, shifts_2d_t, shifts_3d_t, proj_r)
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        kernels().extract_central_slices_rfft_3d_gpu(
            device_session(), bufs, params, addrs
        )
    else:
        kernels().extract_central_slices_rfft_3d(bufs, params)
    return torch.view_as_complex(proj_r)


def run_insert_slices_3d(
    slices: torch.Tensor,
    rotations: torch.Tensor,
    volume_shape: tuple[int, int, int],
    *,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    interpolation: str = "linear",
    friedel_double: bool = False,
    skip_redundant: bool = False,
    ewald_curvature: float = 0.0,
    shifts_3d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert complex ``slices`` ``(bv, bp, h, w)`` into a 3D rfft volume.

    ``volume_shape`` is ``(d, sidelength, sidelength_half)`` (no batch).
    ``shifts_2d`` applies the *conjugate* phase ramp (adjoint of the forward shift).
    ``friedel_double`` also inserts the Hermitian x=0 counterpart and
    ``skip_redundant`` skips the redundant half of the x=0 line (both for
    reconstruction). Returns ``(volume, weight_volume)`` -- complex
    ``(bv, *volume_shape)`` and real weights if ``weights`` was given else
    ``None`` -- on the input device.
    """
    device = slices.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if slices.dim() == 3:
        slices = slices.unsqueeze(0)
    if slices.dim() != 4 or not slices.is_complex():
        raise ValueError("slices must be complex (bv, bp, h, w)")
    bv, bp, h, w = slices.shape
    d, sidelength, sidelength_half = volume_shape

    rot, _bv_rot, bp_rot = prep_rotations(rotations, bv, tgt)
    if bp_rot != bp:
        raise ValueError("rotations pose count must match slices")
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, tgt)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, tgt)

    slices_r = torch.view_as_real(
        slices.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    vol_r = torch.zeros(
        bv, d, sidelength, sidelength_half, 2, dtype=torch.float32, device=tgt
    )
    wvol = torch.zeros(
        bv, d, sidelength, sidelength_half, dtype=torch.float32, device=tgt
    )
    if weights is not None:
        w_in = weights.to(device=tgt, dtype=torch.float32).contiguous()
    else:
        w_in = torch.zeros(bv, bp, h, w, dtype=torch.float32, device=tgt)  # unused

    radius = sidelength / 2.0 if fftfreq_max is None else float(fftfreq_max) * h
    params = KernelParams(
        oversampling=float(oversampling),
        radius_cutoff_sq=float(radius * radius),
        has_shifts_2d=int(has_shifts_2d),
        has_weights=int(weights is not None),
        friedel_double=int(bool(friedel_double)),
        skip_redundant=int(bool(skip_redundant)),
        interp=interp_code(interpolation),
        ewald_curvature=float(ewald_curvature),
        has_shifts_3d=int(has_shifts_3d),
    )
    bufs = (slices_r, w_in, rot, shifts_2d_t, shifts_3d_t, vol_r, wvol)
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        kernels().insert_central_slices_rfft_3d_gpu(
            device_session(), bufs, params, addrs
        )
    else:
        kernels().insert_central_slices_rfft_3d(bufs, params)

    out_vol = torch.view_as_complex(vol_r)
    out_weights = wvol if weights is not None else None
    return out_vol, out_weights


def run_insert_slices_3d_hermitian(
    projections: torch.Tensor,
    rotations: torch.Tensor,
    weights: torch.Tensor | None,
    shifts_2d: torch.Tensor | None,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
    ewald_curvature: float = 0.0,
    shifts_3d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert central slices into a Hermitian 3D rfft volume (+ weights)."""
    if projections.dim() == 3:
        projections = projections.unsqueeze(0)
        if weights is not None and weights.dim() == 3:
            weights = weights.unsqueeze(0)
    if projections.dim() != 4 or not projections.is_complex():
        raise ValueError("projections must be complex (bv, bp, h, w)")
    _bv, _bp, h, w = projections.shape
    if h % 2 != 0 or h != (w - 1) * 2:
        raise ValueError("projections must be square with even box (h == 2*(w-1))")
    if weights is not None and weights.shape != projections.shape:
        raise ValueError("weights must match projections shape")

    return run_insert_slices_3d(
        projections,
        rotations,
        reconstruction_volume_shape(h, oversampling),
        shifts_2d=shifts_2d,
        weights=weights,
        oversampling=oversampling,
        fftfreq_max=fftfreq_max,
        interpolation=interpolation,
        friedel_double=True,
        skip_redundant=True,
        ewald_curvature=ewald_curvature,
        shifts_3d=shifts_3d,
    )


def run_slice_3d_pose_grad(
    volume: torch.Tensor,
    rotations: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    pixels: torch.Tensor,
    *,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
    insert: bool,
    ewald_curvature: float = 0.0,
    shifts_3d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Gradients w.r.t. rotations and shifts_2d (rotation/shift backward kernel).

    ``volume`` is the field whose spatial gradient is sampled (the rfft
    ``reconstruction`` for the extraction, or the data-volume gradient
    ``grad_data_rec`` for the insertion); ``pixels`` is the per-pixel
    cotangent (``grad_projections`` extraction / ``projections`` insertion).
    Returns ``(grad_rotations (bv_rot, bp, 3, 3), grad_shifts_2d (bv_shift_2d,
    bp, 2) or None, grad_shifts_3d (bv_shift_3d, bp, 3) or None)`` on the input
    device.
    """
    device = pixels.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if volume.dim() == 3:
        volume = volume.unsqueeze(0)
    bv, _d, _sidelength, _sh = volume.shape
    rot, bv_rot, bp = prep_rotations(rotations, bv, tgt)
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, tgt)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, tgt)
    bv_shift_2d = shifts_2d_t.shape[0]
    bv_shift_3d = shifts_3d_t.shape[0]
    if pixels.dim() == 3:
        pixels = pixels.unsqueeze(0)
    proj_sidelength = int(pixels.shape[2])

    vol_r = torch.view_as_real(
        volume.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    pix_r = torch.view_as_real(
        pixels.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    grad_rot = torch.zeros(bv_rot, bp, 3, 3, dtype=torch.float32, device=tgt)
    grad_shift = torch.zeros(bv_shift_2d, bp, 2, dtype=torch.float32, device=tgt)
    grad_shift_3d = torch.zeros(bv_shift_3d, bp, 3, dtype=torch.float32, device=tgt)

    radius = (
        proj_sidelength / 2.0
        if fftfreq_max is None
        else float(fftfreq_max) * proj_sidelength
    )
    params = KernelParams(
        oversampling=float(oversampling),
        radius_cutoff_sq=float(radius * radius),
        has_shifts_2d=int(has_shifts_2d),
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        interp=interp_code(interpolation),
        ewald_curvature=float(ewald_curvature),
        has_shifts_3d=int(has_shifts_3d),
    )
    bufs = (
        vol_r,
        rot,
        shifts_2d_t,
        shifts_3d_t,
        pix_r,
        grad_rot,
        grad_shift,
        grad_shift_3d,
    )
    if insert:
        fn = (
            kernels().insert_central_slices_rfft_3d_pose_grad_gpu
            if use_gpu
            else kernels().insert_central_slices_rfft_3d_pose_grad
        )
    else:
        fn = (
            kernels().extract_central_slices_rfft_3d_pose_grad_gpu
            if use_gpu
            else kernels().extract_central_slices_rfft_3d_pose_grad
        )
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        fn(device_session(), bufs, params, addrs)
    else:
        fn(bufs, params)
    return (
        grad_rot,
        (grad_shift if has_shifts_2d else None),
        (grad_shift_3d if has_shifts_3d else None),
    )


def run_slice_3d_weight_grad(
    grad_weight_vol: torch.Tensor,
    rotations: torch.Tensor,
    proj_sidelength: int,
    *,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
    ewald_curvature: float = 0.0,
) -> torch.Tensor:
    """Gradient w.r.t. insertion weights: gather ``grad_weight_vol``.

    Adjoint of the (signed) weight splat. Returns real ``(bv, bp, h, w)`` on the
    input device.
    """
    device = grad_weight_vol.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if grad_weight_vol.dim() == 3:
        grad_weight_vol = grad_weight_vol.unsqueeze(0)
    bv = grad_weight_vol.shape[0]
    rot, _bv_rot, bp = prep_rotations(rotations, bv, tgt)
    proj_sidelength_half = proj_sidelength // 2 + 1

    gwvol = grad_weight_vol.to(device=tgt, dtype=torch.float32).contiguous()
    grad_weight = torch.zeros(
        bv, bp, proj_sidelength, proj_sidelength_half, dtype=torch.float32, device=tgt
    )
    shifts_dummy = torch.zeros(1, bp, 2, dtype=torch.float32, device=tgt)
    shifts_3d_dummy = torch.zeros(1, bp, 3, dtype=torch.float32, device=tgt)
    radius = (
        proj_sidelength / 2.0
        if fftfreq_max is None
        else float(fftfreq_max) * proj_sidelength
    )
    params = KernelParams(
        oversampling=float(oversampling),
        radius_cutoff_sq=float(radius * radius),
        has_shifts_2d=0,
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        interp=interp_code(interpolation),
        ewald_curvature=float(ewald_curvature),
        has_shifts_3d=0,
    )
    bufs = (gwvol, rot, shifts_dummy, shifts_3d_dummy, grad_weight)
    fn = (
        kernels().insert_central_slices_rfft_3d_weight_grad_gpu
        if use_gpu
        else kernels().insert_central_slices_rfft_3d_weight_grad
    )
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        fn(device_session(), bufs, params, addrs)
    else:
        fn(bufs, params)
    return grad_weight


class ExtractSlices3D(torch.autograd.Function):
    """Differentiable 3D->2D central-slice extraction (volume, rotations, shifts_2d)."""

    @staticmethod
    def forward(
        ctx,
        reconstruction: torch.Tensor,
        rotations: torch.Tensor,
        shifts_2d: torch.Tensor | None,
        shifts_3d: torch.Tensor | None,
        output_shape: tuple[int, int] | None,
        oversampling: float,
        fftfreq_max: float | None,
        interpolation: str,
        ewald_curvature: float = 0.0,
    ) -> torch.Tensor:
        proj = run_extract_slices_3d(
            reconstruction,
            rotations,
            shifts_2d,
            output_shape,
            oversampling,
            fftfreq_max,
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
        ctx.fftfreq_max = fftfreq_max
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
            # adjoint of the extraction: scatter grad back into the volume
            # (pure transpose, no x=0 skip / Hermitian double-insert).
            grad_rec, _ = run_insert_slices_3d(
                grad_proj,
                ctx.rotations,
                ctx.vol_shape,
                shifts_2d=ctx.shifts_2d,
                oversampling=ctx.oversampling,
                fftfreq_max=ctx.fftfreq_max,
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
            gr, gs, gs3 = run_slice_3d_pose_grad(
                ctx.reconstruction,
                ctx.rotations,
                ctx.shifts_2d,
                grad_proj,
                oversampling=ctx.oversampling,
                fftfreq_max=ctx.fftfreq_max,
                interpolation=ctx.interpolation,
                insert=False,
                ewald_curvature=ctx.ewald_curvature,
                shifts_3d=ctx.shifts_3d,
            )
            if needs[1]:
                grad_rot = reduce_to(gr, ctx.rotations)
            if needs[2] and ctx.shifts_2d is not None and gs is not None:
                grad_shift = reduce_to(gs, ctx.shifts_2d)
            if needs[3] and ctx.shifts_3d is not None and gs3 is not None:
                grad_shift_3d = reduce_to(gs3, ctx.shifts_3d)

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


class InsertSlices3D(torch.autograd.Function):
    """Differentiable 2D->3D central-slice insertion.

    Differentiable w.r.t. projections, weights, rotations and shifts.

    The data-gradient is the exact transpose of the (friedel_double +
    skip_redundant) scatter: symmetrise the volume gradient on the kx=0 plane,
    extract slices from it, then zero the gradient of the skipped redundant x=0 line.
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
        fftfreq_max: float | None,
        interpolation: str,
        ewald_curvature: float = 0.0,
    ):
        data_vol, weight_vol = run_insert_slices_3d_hermitian(
            projections,
            rotations,
            weights,
            shifts_2d,
            oversampling,
            fftfreq_max,
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
        ctx.fftfreq_max = fftfreq_max
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
            g = symmetrise_kx0_plane(grad_data)

        grad_proj = None
        if needs[0]:
            grad_proj = run_extract_slices_3d(
                g,
                ctx.rotations,
                ctx.shifts_2d,
                (side, side),
                ctx.oversampling,
                ctx.fftfreq_max,
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
            gw = run_slice_3d_weight_grad(
                grad_weight,
                ctx.rotations,
                side,
                oversampling=ctx.oversampling,
                fftfreq_max=ctx.fftfreq_max,
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
            gr, gs, gs3 = run_slice_3d_pose_grad(
                g,
                ctx.rotations,
                ctx.shifts_2d,
                ctx.projections,
                oversampling=ctx.oversampling,
                fftfreq_max=ctx.fftfreq_max,
                interpolation=ctx.interpolation,
                insert=True,
                ewald_curvature=ctx.ewald_curvature,
                shifts_3d=ctx.shifts_3d,
            )
            if needs[2]:
                grad_rot = reduce_to(gr, ctx.rotations)
            if needs[3] and ctx.shifts_2d is not None and gs is not None:
                grad_shift = reduce_to(gs, ctx.shifts_2d)
            if needs[4] and ctx.shifts_3d is not None and gs3 is not None:
                grad_shift_3d = reduce_to(gs3, ctx.shifts_3d)

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
