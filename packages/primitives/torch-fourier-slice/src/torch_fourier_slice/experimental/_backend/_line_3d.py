"""3D central-line backend: 3D volume <-> 1D central lines.

A central line is the degenerate central slice with its in-plane (y) axis
collapsed to the single DC row: a 1D rfft node sampled along a direction on the
sphere. The 2D image-plane shift and Ewald curvature do not apply; only the 3D
(volume-frame) shift is honoured (the design's per-node ``u . t`` phase ramp).

The ``run_*`` ops validate inputs, build device buffers and dispatch to the CPU
or GPU Mojo kernel; ``ExtractLines3D`` / ``InsertLines3D`` wire them into
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
    prep_directions_3d,
    prep_shifts_3d,
    validate_reconstruction,
)


def run_extract_lines_3d(
    reconstruction: torch.Tensor,
    directions: torch.Tensor,
    output_length: int | None,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
    shifts_3d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract 1D central lines from an rfft volume (CPU or GPU per device).

    ``directions`` are zyx unit vectors ``(bv_dir, bp, 3)`` (broadcastable).
    Returns complex ``(bv, bp, w)`` lines (rfft half-line, DC at origin), where
    ``w = output_length // 2 + 1`` (defaults to the volume's rfft half-width).
    """
    device = reconstruction.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    reconstruction, bv, sidelength, _sh = validate_reconstruction(reconstruction)
    dir_t, _bv_dir, bp = prep_directions_3d(directions, bv, tgt)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, tgt)

    if output_length is None:
        line_sidelength = sidelength
    else:
        line_sidelength = int(output_length)
        if line_sidelength % 2 != 0:
            raise ValueError(f"output_length {line_sidelength} must be even")
    line_half = line_sidelength // 2 + 1

    radius = (
        line_sidelength / 2.0
        if fftfreq_max is None
        else float(fftfreq_max) * line_sidelength
    )
    params = KernelParams(
        oversampling=float(oversampling),
        radius_cutoff_sq=float(radius * radius),
        has_shifts_2d=0,
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        interp=interp_code(interpolation),
        ewald_curvature=0.0,
        has_shifts_3d=int(has_shifts_3d),
    )
    line_r = torch.zeros(bv, bp, line_half, 2, dtype=torch.float32, device=tgt)
    rec_r = torch.view_as_real(
        reconstruction.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    bufs = (rec_r, dir_t, shifts_3d_t, line_r)
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        kernels().extract_central_lines_rfft_3d_gpu(
            device_session(), bufs, params, addrs
        )
    else:
        kernels().extract_central_lines_rfft_3d(bufs, params)
    return torch.view_as_complex(line_r)


def run_insert_lines_3d(
    lines: torch.Tensor,
    directions: torch.Tensor,
    volume_shape: tuple[int, int, int],
    *,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    interpolation: str = "linear",
    friedel_double: bool = False,
    shifts_3d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert complex ``lines`` ``(bv, bp, w)`` into a 3D rfft volume.

    ``directions`` are zyx unit vectors; ``volume_shape`` is ``(d, sidelength,
    sidelength_half)`` (no batch). ``friedel_double`` also inserts the Hermitian
    kx=0 counterpart (for reconstruction). Returns ``(volume, weight_volume)``.
    """
    device = lines.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if lines.dim() == 2:
        lines = lines.unsqueeze(0)
    if lines.dim() != 3 or not lines.is_complex():
        raise ValueError("lines must be complex (bv, bp, w)")
    bv, bp, w = lines.shape
    d, sidelength, sidelength_half = volume_shape

    dir_t, _bv_dir, bp_dir = prep_directions_3d(directions, bv, tgt)
    if bp_dir != bp:
        raise ValueError("directions pose count must match lines")
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, tgt)

    lines_r = torch.view_as_real(
        lines.to(device=tgt, dtype=torch.complex64).contiguous()
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
        w_in = torch.zeros(bv, bp, w, dtype=torch.float32, device=tgt)  # unused

    line_sidelength = 2 * (w - 1)
    radius = (
        line_sidelength / 2.0
        if fftfreq_max is None
        else float(fftfreq_max) * line_sidelength
    )
    params = KernelParams(
        oversampling=float(oversampling),
        radius_cutoff_sq=float(radius * radius),
        has_shifts_2d=0,
        has_weights=int(weights is not None),
        friedel_double=int(bool(friedel_double)),
        skip_redundant=0,
        interp=interp_code(interpolation),
        ewald_curvature=0.0,
        has_shifts_3d=int(has_shifts_3d),
    )
    bufs = (lines_r, w_in, dir_t, shifts_3d_t, vol_r, wvol)
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        kernels().insert_central_lines_rfft_3d_gpu(
            device_session(), bufs, params, addrs
        )
    else:
        kernels().insert_central_lines_rfft_3d(bufs, params)

    out_vol = torch.view_as_complex(vol_r)
    out_weights = wvol if weights is not None else None
    return out_vol, out_weights


def run_line_3d_pose_grad(
    volume: torch.Tensor,
    directions: torch.Tensor,
    line_pixels: torch.Tensor,
    *,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
    insert: bool,
    shifts_3d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gradients w.r.t. line directions and 3D shifts (pose backward kernel).

    ``volume`` is the field whose spatial gradient is sampled (the ``rec`` for the
    extraction, or the data-volume gradient for the insertion);
    ``line_pixels`` is the per-node cotangent. Returns ``(grad_dir (bv_dir, bp,
    3), grad_shift_3d (bv_shift_3d, bp, 3) or None)`` on the input device.
    """
    device = line_pixels.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if volume.dim() == 3:
        volume = volume.unsqueeze(0)
    bv = volume.shape[0]
    dir_t, bv_dir, bp = prep_directions_3d(directions, bv, tgt)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, tgt)
    bv_shift_3d = shifts_3d_t.shape[0]
    if line_pixels.dim() == 2:
        line_pixels = line_pixels.unsqueeze(0)
    line_sidelength = 2 * (int(line_pixels.shape[2]) - 1)

    vol_r = torch.view_as_real(
        volume.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    pix_r = torch.view_as_real(
        line_pixels.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    grad_dir = torch.zeros(bv_dir, bp, 3, dtype=torch.float32, device=tgt)
    grad_shift_3d = torch.zeros(bv_shift_3d, bp, 3, dtype=torch.float32, device=tgt)

    radius = (
        line_sidelength / 2.0
        if fftfreq_max is None
        else float(fftfreq_max) * line_sidelength
    )
    params = KernelParams(
        oversampling=float(oversampling),
        radius_cutoff_sq=float(radius * radius),
        has_shifts_2d=0,
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        interp=interp_code(interpolation),
        ewald_curvature=0.0,
        has_shifts_3d=int(has_shifts_3d),
    )
    bufs = (vol_r, dir_t, shifts_3d_t, pix_r, grad_dir, grad_shift_3d)
    if insert:
        fn = (
            kernels().insert_central_lines_rfft_3d_pose_grad_gpu
            if use_gpu
            else kernels().insert_central_lines_rfft_3d_pose_grad
        )
    else:
        fn = (
            kernels().extract_central_lines_rfft_3d_pose_grad_gpu
            if use_gpu
            else kernels().extract_central_lines_rfft_3d_pose_grad
        )
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        fn(device_session(), bufs, params, addrs)
    else:
        fn(bufs, params)
    return grad_dir, (grad_shift_3d if has_shifts_3d else None)


def run_line_3d_weight_grad(
    grad_weight_vol: torch.Tensor,
    directions: torch.Tensor,
    line_sidelength: int,
    *,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
) -> torch.Tensor:
    """Gradient w.r.t. line insertion weights: gather ``grad_weight_vol``.

    Adjoint of the (Hermitian-doubled) line weight splat. Returns real
    ``(bv, bp, w)`` on the input device.
    """
    device = grad_weight_vol.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if grad_weight_vol.dim() == 3:
        grad_weight_vol = grad_weight_vol.unsqueeze(0)
    bv = grad_weight_vol.shape[0]
    dir_t, _bv_dir, bp = prep_directions_3d(directions, bv, tgt)
    line_half = line_sidelength // 2 + 1

    gwvol = grad_weight_vol.to(device=tgt, dtype=torch.float32).contiguous()
    grad_weight = torch.zeros(bv, bp, line_half, dtype=torch.float32, device=tgt)
    radius = (
        line_sidelength / 2.0
        if fftfreq_max is None
        else float(fftfreq_max) * line_sidelength
    )
    params = KernelParams(
        oversampling=float(oversampling),
        radius_cutoff_sq=float(radius * radius),
        has_shifts_2d=0,
        has_weights=0,
        friedel_double=1,
        skip_redundant=0,
        interp=interp_code(interpolation),
        ewald_curvature=0.0,
        has_shifts_3d=0,
    )
    bufs = (gwvol, dir_t, grad_weight)
    fn = (
        kernels().insert_central_lines_rfft_3d_weight_grad_gpu
        if use_gpu
        else kernels().insert_central_lines_rfft_3d_weight_grad
    )
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        fn(device_session(), bufs, params, addrs)
    else:
        fn(bufs, params)
    return grad_weight


class ExtractLines3D(torch.autograd.Function):
    """Differentiable 3D->1D central-line extraction.

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
        fftfreq_max: float | None,
        interpolation: str,
    ) -> torch.Tensor:
        line = run_extract_lines_3d(
            reconstruction,
            directions,
            output_length,
            oversampling,
            fftfreq_max,
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
        ctx.fftfreq_max = fftfreq_max
        ctx.interpolation = interpolation
        ctx.input_dim = reconstruction.dim()
        return line

    @staticmethod
    def backward(ctx, grad_line: torch.Tensor):
        needs = ctx.needs_input_grad
        grad_line = grad_line.contiguous()
        grad_rec = None
        if needs[0]:
            # adjoint of the line extraction: scatter grad back into the
            # volume (pure transpose, no Hermitian double-insert).
            grad_rec, _ = run_insert_lines_3d(
                grad_line,
                ctx.directions,
                ctx.vol_shape,
                oversampling=ctx.oversampling,
                fftfreq_max=ctx.fftfreq_max,
                interpolation=ctx.interpolation,
                friedel_double=False,
                shifts_3d=ctx.shifts_3d,
            )
            if ctx.input_dim == 3:
                grad_rec = grad_rec[0]

        grad_dir = None
        grad_shift_3d = None
        if needs[1] or needs[2]:
            gd, gs3 = run_line_3d_pose_grad(
                ctx.reconstruction,
                ctx.directions,
                grad_line,
                oversampling=ctx.oversampling,
                fftfreq_max=ctx.fftfreq_max,
                interpolation=ctx.interpolation,
                insert=False,
                shifts_3d=ctx.shifts_3d,
            )
            if needs[1]:
                grad_dir = reduce_to(gd, ctx.directions)
            if needs[2] and ctx.shifts_3d is not None and gs3 is not None:
                grad_shift_3d = reduce_to(gs3, ctx.shifts_3d)
        return (grad_rec, grad_dir, grad_shift_3d, None, None, None, None)


class InsertLines3D(torch.autograd.Function):
    """Differentiable 1D->3D central-line insertion (reconstruction).

    Hermitian double-inserts on the volume's kx=0 plane. Differentiable w.r.t.
    the input ``lines`` (adjoint line extraction of the kx=0-symmetrised
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
        fftfreq_max: float | None,
        interpolation: str,
    ):
        lines4 = lines.unsqueeze(0) if lines.dim() == 2 else lines
        line_sidelength = 2 * (int(lines4.shape[2]) - 1)
        data_vol, weight_vol = run_insert_lines_3d(
            lines,
            directions,
            reconstruction_volume_shape(line_sidelength, oversampling),
            weights=weights,
            oversampling=oversampling,
            fftfreq_max=fftfreq_max,
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
        ctx.fftfreq_max = fftfreq_max
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
            g = symmetrise_kx0_plane(grad_data)

        grad_lines = None
        if needs[0]:
            grad_lines = run_extract_lines_3d(
                g,
                ctx.directions,
                ctx.line_sidelength,
                ctx.oversampling,
                ctx.fftfreq_max,
                ctx.interpolation,
                ctx.shifts_3d,
            ).clone()
            if ctx.input_dim == 2:
                grad_lines = grad_lines[0]

        grad_weights = None
        if needs[1] and ctx.weights is not None and grad_weight is not None:
            gw = run_line_3d_weight_grad(
                grad_weight,
                ctx.directions,
                ctx.line_sidelength,
                oversampling=ctx.oversampling,
                fftfreq_max=ctx.fftfreq_max,
                interpolation=ctx.interpolation,
            )
            if ctx.input_dim == 2:
                gw = gw[0]
            grad_weights = gw.to(device=ctx.weights.device, dtype=ctx.weights.dtype)

        grad_dir = None
        grad_shift_3d = None
        if needs[2] or needs[3]:
            gd, gs3 = run_line_3d_pose_grad(
                g,
                ctx.directions,
                ctx.lines,
                oversampling=ctx.oversampling,
                fftfreq_max=ctx.fftfreq_max,
                interpolation=ctx.interpolation,
                insert=True,
                shifts_3d=ctx.shifts_3d,
            )
            if needs[2]:
                grad_dir = reduce_to(gd, ctx.directions)
            if needs[3] and ctx.shifts_3d is not None and gs3 is not None:
                grad_shift_3d = reduce_to(gs3, ctx.shifts_3d)
        return (grad_lines, grad_weights, grad_dir, grad_shift_3d, None, None, None)
