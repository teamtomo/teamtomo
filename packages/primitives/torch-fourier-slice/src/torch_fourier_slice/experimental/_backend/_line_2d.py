"""2D central-line backend: 2D image <-> 1D central lines.

The graph reads nodes from crop FTs, not volumes.

The ``run_*`` ops validate inputs, build device buffers and dispatch to the CPU
or GPU Mojo kernel; ``ExtractLines2D`` / ``InsertLines2D`` wire them into
autograd.
"""

from __future__ import annotations

import torch

from ._common import reduce_to
from ._device import prepare_launch
from ._loader import device_session, kernels
from ._validation import (
    KernelParams,
    interp_code,
    prep_directions_2d,
    prep_shifts_2d,
    validate_image_2d,
)


def run_extract_lines_2d(
    image_rfft: torch.Tensor,
    directions: torch.Tensor,
    output_length: int | None,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
    shifts_2d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract 1D central lines from 2D rfft images (CPU or GPU per device).

    ``image_rfft`` is ``(bv, h, w)`` (DC at origin, square/even). ``directions``
    are yx unit vectors ``(bv_dir, bp, 2)``; ``shifts_2d`` optional yx image
    translations ``(bv_shift, bp, 2)``. Returns complex ``(bv, bp, w_out)``.
    """
    device = image_rfft.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    image_rfft, bv, h, _w = validate_image_2d(image_rfft)
    dir_t, _bv_dir, bp = prep_directions_2d(directions, bv, tgt)
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, tgt)

    line_sidelength = h if output_length is None else int(output_length)
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
        has_shifts_2d=int(has_shifts_2d),
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        interp=interp_code(interpolation),
        ewald_curvature=0.0,
        has_shifts_3d=0,
    )
    line_r = torch.zeros(bv, bp, line_half, 2, dtype=torch.float32, device=tgt)
    img_r = torch.view_as_real(
        image_rfft.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    bufs = (img_r, dir_t, shifts_2d_t, line_r)
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        kernels().extract_central_lines_rfft_2d_gpu(
            device_session(), bufs, params, addrs
        )
    else:
        kernels().extract_central_lines_rfft_2d(bufs, params)
    return torch.view_as_complex(line_r)


def run_insert_lines_2d(
    lines: torch.Tensor,
    directions: torch.Tensor,
    image_shape: tuple[int, int],
    *,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fftfreq_max: float | None = None,
    interpolation: str = "linear",
    friedel_double: bool = False,
    shifts_2d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert complex ``lines`` ``(bv, bp, w)`` into a 2D rfft image.

    ``image_shape`` is ``(h, w_rfft)`` (no batch). ``shifts_2d`` applies the
    conjugate phase ramp (adjoint of the forward shift). Returns ``(image,
    weight_image)``.
    """
    device = lines.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if lines.dim() == 2:
        lines = lines.unsqueeze(0)
    if lines.dim() != 3 or not lines.is_complex():
        raise ValueError("lines must be complex (bv, bp, w)")
    bv, bp, lw = lines.shape
    h, w_rfft = image_shape

    dir_t, _bv_dir, bp_dir = prep_directions_2d(directions, bv, tgt)
    if bp_dir != bp:
        raise ValueError("directions pose count must match lines")
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, tgt)

    lines_r = torch.view_as_real(
        lines.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    vol_r = torch.zeros(bv, h, w_rfft, 2, dtype=torch.float32, device=tgt)
    wvol = torch.zeros(bv, h, w_rfft, dtype=torch.float32, device=tgt)
    if weights is not None:
        w_in = weights.to(device=tgt, dtype=torch.float32).contiguous()
    else:
        w_in = torch.zeros(bv, bp, lw, dtype=torch.float32, device=tgt)  # unused

    line_sidelength = 2 * (lw - 1)
    radius = (
        line_sidelength / 2.0
        if fftfreq_max is None
        else float(fftfreq_max) * line_sidelength
    )
    params = KernelParams(
        oversampling=float(oversampling),
        radius_cutoff_sq=float(radius * radius),
        has_shifts_2d=int(has_shifts_2d),
        has_weights=int(weights is not None),
        friedel_double=int(bool(friedel_double)),
        skip_redundant=0,
        interp=interp_code(interpolation),
        ewald_curvature=0.0,
        has_shifts_3d=0,
    )
    bufs = (lines_r, w_in, dir_t, shifts_2d_t, vol_r, wvol)
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        kernels().insert_central_lines_rfft_2d_gpu(
            device_session(), bufs, params, addrs
        )
    else:
        kernels().insert_central_lines_rfft_2d(bufs, params)

    out_vol = torch.view_as_complex(vol_r)
    out_weights = wvol if weights is not None else None
    return out_vol, out_weights


def run_line_2d_pose_grad(
    image: torch.Tensor,
    directions: torch.Tensor,
    line_pixels: torch.Tensor,
    *,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
    insert: bool,
    shifts_2d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gradient w.r.t. 2D line directions and shifts (pose backward kernel).

    ``image`` is the field whose spatial gradient is sampled (the ``img`` for the
    extraction, or the image-gradient for the insertion); ``line_pixels``
    is the per-node cotangent. Returns ``(grad_dir (bv_dir, bp, 2), grad_shift
    (bv_shift, bp, 2) or None)``.
    """
    device = line_pixels.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if image.dim() == 2:
        image = image.unsqueeze(0)
    bv = image.shape[0]
    dir_t, bv_dir, bp = prep_directions_2d(directions, bv, tgt)
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, tgt)
    bv_shift_2d = shifts_2d_t.shape[0]
    if line_pixels.dim() == 2:
        line_pixels = line_pixels.unsqueeze(0)
    line_sidelength = 2 * (int(line_pixels.shape[2]) - 1)

    img_r = torch.view_as_real(
        image.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    pix_r = torch.view_as_real(
        line_pixels.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    grad_dir = torch.zeros(bv_dir, bp, 2, dtype=torch.float32, device=tgt)
    grad_shift = torch.zeros(bv_shift_2d, bp, 2, dtype=torch.float32, device=tgt)

    radius = (
        line_sidelength / 2.0
        if fftfreq_max is None
        else float(fftfreq_max) * line_sidelength
    )
    params = KernelParams(
        oversampling=float(oversampling),
        radius_cutoff_sq=float(radius * radius),
        has_shifts_2d=int(has_shifts_2d),
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        interp=interp_code(interpolation),
        ewald_curvature=0.0,
        has_shifts_3d=0,
    )
    bufs = (img_r, dir_t, shifts_2d_t, pix_r, grad_dir, grad_shift)
    if insert:
        fn = (
            kernels().insert_central_lines_rfft_2d_pose_grad_gpu
            if use_gpu
            else kernels().insert_central_lines_rfft_2d_pose_grad
        )
    else:
        fn = (
            kernels().extract_central_lines_rfft_2d_pose_grad_gpu
            if use_gpu
            else kernels().extract_central_lines_rfft_2d_pose_grad
        )
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        fn(device_session(), bufs, params, addrs)
    else:
        fn(bufs, params)
    return grad_dir, (grad_shift if has_shifts_2d else None)


def run_line_2d_weight_grad(
    grad_weight_img: torch.Tensor,
    directions: torch.Tensor,
    line_sidelength: int,
    *,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
) -> torch.Tensor:
    """Gradient w.r.t. 2D line insertion weights: gather ``grad_weight_img``.

    Adjoint of the (Hermitian-doubled) line weight splat. Returns real
    ``(bv, bp, w)``.
    """
    device = grad_weight_img.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if grad_weight_img.dim() == 2:
        grad_weight_img = grad_weight_img.unsqueeze(0)
    bv = grad_weight_img.shape[0]
    dir_t, _bv_dir, bp = prep_directions_2d(directions, bv, tgt)
    line_half = line_sidelength // 2 + 1

    gwimg = grad_weight_img.to(device=tgt, dtype=torch.float32).contiguous()
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
    bufs = (gwimg, dir_t, grad_weight)
    fn = (
        kernels().insert_central_lines_rfft_2d_weight_grad_gpu
        if use_gpu
        else kernels().insert_central_lines_rfft_2d_weight_grad
    )
    if use_gpu:
        addrs = prepare_launch(device, bufs)
        fn(device_session(), bufs, params, addrs)
    else:
        fn(bufs, params)
    return grad_weight


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


class ExtractLines2D(torch.autograd.Function):
    """Differentiable 2D->1D central-line extraction.

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
        fftfreq_max: float | None,
        interpolation: str,
    ) -> torch.Tensor:
        line = run_extract_lines_2d(
            image_rfft,
            directions,
            output_length,
            oversampling,
            fftfreq_max,
            interpolation,
            shifts_2d,
        )
        img3 = image_rfft.unsqueeze(0) if image_rfft.dim() == 2 else image_rfft
        ctx.image_shape = (int(img3.shape[-2]), int(img3.shape[-1]))
        ctx.image_rfft = image_rfft
        ctx.directions = directions
        ctx.shifts_2d = shifts_2d
        ctx.oversampling = oversampling
        ctx.fftfreq_max = fftfreq_max
        ctx.interpolation = interpolation
        ctx.input_dim = image_rfft.dim()
        return line

    @staticmethod
    def backward(ctx, grad_line: torch.Tensor):
        needs = ctx.needs_input_grad
        grad_line = grad_line.contiguous()
        grad_img = None
        if needs[0]:
            grad_img, _ = run_insert_lines_2d(
                grad_line,
                ctx.directions,
                ctx.image_shape,
                oversampling=ctx.oversampling,
                fftfreq_max=ctx.fftfreq_max,
                interpolation=ctx.interpolation,
                friedel_double=False,
                shifts_2d=ctx.shifts_2d,
            )
            if ctx.input_dim == 2:
                grad_img = grad_img[0]

        grad_dir = None
        grad_shift = None
        if needs[1] or needs[2]:
            gd, gs = run_line_2d_pose_grad(
                ctx.image_rfft,
                ctx.directions,
                grad_line,
                oversampling=ctx.oversampling,
                fftfreq_max=ctx.fftfreq_max,
                interpolation=ctx.interpolation,
                insert=False,
                shifts_2d=ctx.shifts_2d,
            )
            if needs[1]:
                grad_dir = reduce_to(gd, ctx.directions)
            if needs[2] and ctx.shifts_2d is not None and gs is not None:
                grad_shift = reduce_to(gs, ctx.shifts_2d)
        return (grad_img, grad_dir, grad_shift, None, None, None, None)


class InsertLines2D(torch.autograd.Function):
    """Differentiable 1D->2D central-line insertion (Hermitian, kx=0 double-insert).

    Differentiable w.r.t. the input ``lines`` (adjoint 2D line extraction
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
        fftfreq_max: float | None,
        interpolation: str,
    ):
        lines3 = lines.unsqueeze(0) if lines.dim() == 2 else lines
        line_sidelength = 2 * (int(lines3.shape[2]) - 1)
        image_shape = (line_sidelength, line_sidelength // 2 + 1)
        img, wimg = run_insert_lines_2d(
            lines,
            directions,
            image_shape,
            weights=weights,
            oversampling=oversampling,
            fftfreq_max=fftfreq_max,
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
        ctx.fftfreq_max = fftfreq_max
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
            grad_lines = run_extract_lines_2d(
                g,
                ctx.directions,
                ctx.line_sidelength,
                ctx.oversampling,
                ctx.fftfreq_max,
                ctx.interpolation,
                ctx.shifts_2d,
            ).clone()
            if ctx.input_dim == 2:
                grad_lines = grad_lines[0]

        grad_weights = None
        if needs[1] and ctx.weights is not None and grad_weight is not None:
            gw = run_line_2d_weight_grad(
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
        grad_shift = None
        if needs[2] or needs[3]:
            gd, gs = run_line_2d_pose_grad(
                g,
                ctx.directions,
                ctx.lines,
                oversampling=ctx.oversampling,
                fftfreq_max=ctx.fftfreq_max,
                interpolation=ctx.interpolation,
                insert=True,
                shifts_2d=ctx.shifts_2d,
            )
            if needs[2]:
                grad_dir = reduce_to(gd, ctx.directions)
            if needs[3] and ctx.shifts_2d is not None and gs is not None:
                grad_shift = reduce_to(gs, ctx.shifts_2d)
        return (grad_lines, grad_weights, grad_dir, grad_shift, None, None, None)
