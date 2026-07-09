"""GPU plumbing: the per-pixel kernels and their launchers.

Each kernel runs one thread per rfft output pixel; `FourierSliceParams` isn't
`DevicePassable`, so the kernels take its (primitive) fields as scalars and
rebuild it on the device (only the per-pixel math is shared with the CPU path).
The kernels read and write torch device memory in place -- the Python caller
passes raw device addresses (see `projectors.mojo` / `experimental/_gpu.py`), so
there is no host<->device staging here.
"""

from std.math import ceildiv
from std.gpu import global_idx
from std.gpu.host import DeviceContext

from _common import BLOCK, FP, FourierSliceParams
from _pixel import _project_pixel, _scatter_pixel
from _pose_grad import (
    _backproject_pose_grad_pixel,
    _forward_pose_grad_pixel,
    _weight_grad_pixel,
)


# ---------------------------------------------------------------------------
# Kernels (one thread per rfft pixel; rebuild `p` from primitive scalars)
# ---------------------------------------------------------------------------


def _project_gpu_kernel(
    rec: FP,
    rot: FP,
    shifts_2d: FP,
    shifts_3d: FP,
    proj: FP,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    bv_shift_2d: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_2d: Int,
    interp: Int,
    ewald_curvature: Float32,
    has_shifts_3d: Int,
    bv_shift_3d: Int,
):
    var idx = global_idx.x
    if idx >= total:
        return
    var p = FourierSliceParams(
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        bv_shift_2d,
        oversampling,
        radius_cutoff_sq,
        has_shifts_2d,
        interp,
        0,
        0,
        0,
        ewald_curvature,
        has_shifts_3d,
        bv_shift_3d,
    )
    var psh = p.proj_sidelength_half()
    var x = idx % psh
    var t = idx // psh
    var y = t % proj_sidelength
    var vp = t // proj_sidelength
    _project_pixel(rec, rot, shifts_2d, shifts_3d, proj, vp // bp, vp % bp, y, x, p)


def _scatter_gpu_kernel(
    inp: FP,
    weights: FP,
    rot: FP,
    shifts_2d: FP,
    shifts_3d: FP,
    vol: FP,
    wvol: FP,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    bv_shift_2d: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_2d: Int,
    interp: Int,
    has_weights: Int,
    friedel_double: Int,
    skip_redundant: Int,
    ewald_curvature: Float32,
    has_shifts_3d: Int,
    bv_shift_3d: Int,
):
    var idx = global_idx.x
    if idx >= total:
        return
    var p = FourierSliceParams(
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        bv_shift_2d,
        oversampling,
        radius_cutoff_sq,
        has_shifts_2d,
        interp,
        has_weights,
        friedel_double,
        skip_redundant,
        ewald_curvature,
        has_shifts_3d,
        bv_shift_3d,
    )
    var psh = p.proj_sidelength_half()
    var x = idx % psh
    var t = idx // psh
    var y = t % proj_sidelength
    var vp = t // proj_sidelength
    _scatter_pixel(
        inp, weights, rot, shifts_2d, shifts_3d, vol, wvol, vp // bp, vp % bp, y, x, p
    )


def _forward_pose_grad_kernel(
    rec: FP,
    rot: FP,
    shifts_2d: FP,
    shifts_3d: FP,
    grad_proj: FP,
    grad_rot: FP,
    grad_shift: FP,
    grad_shift_3d: FP,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    bv_shift_2d: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_2d: Int,
    interp: Int,
    ewald_curvature: Float32,
    has_shifts_3d: Int,
    bv_shift_3d: Int,
):
    var idx = global_idx.x
    if idx >= total:
        return
    var p = FourierSliceParams(
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        bv_shift_2d,
        oversampling,
        radius_cutoff_sq,
        has_shifts_2d,
        interp,
        0,
        0,
        0,
        ewald_curvature,
        has_shifts_3d,
        bv_shift_3d,
    )
    var psh = p.proj_sidelength_half()
    var x = idx % psh
    var t = idx // psh
    var y = t % proj_sidelength
    var vp = t // proj_sidelength
    _forward_pose_grad_pixel(
        rec,
        rot,
        shifts_2d,
        shifts_3d,
        grad_proj,
        grad_rot,
        grad_shift,
        grad_shift_3d,
        vp // bp,
        vp % bp,
        y,
        x,
        p,
    )


def _backproject_pose_grad_kernel(
    grad_rec: FP,
    rot: FP,
    shifts_2d: FP,
    shifts_3d: FP,
    proj: FP,
    grad_rot: FP,
    grad_shift: FP,
    grad_shift_3d: FP,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    bv_shift_2d: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_2d: Int,
    interp: Int,
    ewald_curvature: Float32,
    has_shifts_3d: Int,
    bv_shift_3d: Int,
):
    var idx = global_idx.x
    if idx >= total:
        return
    var p = FourierSliceParams(
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        bv_shift_2d,
        oversampling,
        radius_cutoff_sq,
        has_shifts_2d,
        interp,
        0,
        0,
        0,
        ewald_curvature,
        has_shifts_3d,
        bv_shift_3d,
    )
    var psh = p.proj_sidelength_half()
    var x = idx % psh
    var t = idx // psh
    var y = t % proj_sidelength
    var vp = t // proj_sidelength
    _backproject_pose_grad_pixel(
        grad_rec,
        rot,
        shifts_2d,
        shifts_3d,
        proj,
        grad_rot,
        grad_shift,
        grad_shift_3d,
        vp // bp,
        vp % bp,
        y,
        x,
        p,
    )


def _weight_grad_kernel(
    gwvol: FP,
    rot: FP,
    grad_weight: FP,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    bv_shift_2d: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    interp: Int,
    friedel_double: Int,
    ewald_curvature: Float32,
):
    var idx = global_idx.x
    if idx >= total:
        return
    var p = FourierSliceParams(
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        bv_shift_2d,
        oversampling,
        radius_cutoff_sq,
        0,
        interp,
        0,
        friedel_double,
        0,
        ewald_curvature,
        0,
        0,
    )
    var psh = p.proj_sidelength_half()
    var x = idx % psh
    var t = idx // psh
    var y = t % proj_sidelength
    var vp = t // proj_sidelength
    _weight_grad_pixel(gwvol, rot, grad_weight, vp // bp, vp % bp, y, x, p)


# ---------------------------------------------------------------------------
# Launchers (unpack `p` to device scalars)
# ---------------------------------------------------------------------------


@always_inline
def _launch_project(
    ctx: DeviceContext,
    rec: FP,
    rot: FP,
    shifts_2d: FP,
    shifts_3d: FP,
    proj: FP,
    total: Int,
    p: FourierSliceParams,
) raises:
    ctx.enqueue_function[_project_gpu_kernel](
        rec,
        rot,
        shifts_2d,
        shifts_3d,
        proj,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.bv_shift_2d,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_2d,
        p.interp,
        p.ewald_curvature,
        p.has_shifts_3d,
        p.bv_shift_3d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_scatter(
    ctx: DeviceContext,
    inp: FP,
    weights: FP,
    rot: FP,
    shifts_2d: FP,
    shifts_3d: FP,
    vol: FP,
    wvol: FP,
    total: Int,
    p: FourierSliceParams,
) raises:
    ctx.enqueue_function[_scatter_gpu_kernel](
        inp,
        weights,
        rot,
        shifts_2d,
        shifts_3d,
        vol,
        wvol,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.bv_shift_2d,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_2d,
        p.interp,
        p.has_weights,
        p.friedel_double,
        p.skip_redundant,
        p.ewald_curvature,
        p.has_shifts_3d,
        p.bv_shift_3d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_forward_pose_grad(
    ctx: DeviceContext,
    rec: FP,
    rot: FP,
    shifts_2d: FP,
    shifts_3d: FP,
    grad_proj: FP,
    grad_rot: FP,
    grad_shift: FP,
    grad_shift_3d: FP,
    total: Int,
    p: FourierSliceParams,
) raises:
    ctx.enqueue_function[_forward_pose_grad_kernel](
        rec,
        rot,
        shifts_2d,
        shifts_3d,
        grad_proj,
        grad_rot,
        grad_shift,
        grad_shift_3d,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.bv_shift_2d,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_2d,
        p.interp,
        p.ewald_curvature,
        p.has_shifts_3d,
        p.bv_shift_3d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_backproject_pose_grad(
    ctx: DeviceContext,
    grad_rec: FP,
    rot: FP,
    shifts_2d: FP,
    shifts_3d: FP,
    proj: FP,
    grad_rot: FP,
    grad_shift: FP,
    grad_shift_3d: FP,
    total: Int,
    p: FourierSliceParams,
) raises:
    ctx.enqueue_function[_backproject_pose_grad_kernel](
        grad_rec,
        rot,
        shifts_2d,
        shifts_3d,
        proj,
        grad_rot,
        grad_shift,
        grad_shift_3d,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.bv_shift_2d,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_2d,
        p.interp,
        p.ewald_curvature,
        p.has_shifts_3d,
        p.bv_shift_3d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_weight_grad(
    ctx: DeviceContext,
    gwvol: FP,
    rot: FP,
    grad_weight: FP,
    total: Int,
    p: FourierSliceParams,
) raises:
    ctx.enqueue_function[_weight_grad_kernel](
        gwvol,
        rot,
        grad_weight,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.bv_shift_2d,
        p.oversampling,
        p.radius_cutoff_sq,
        p.interp,
        p.friedel_double,
        p.ewald_curvature,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )
