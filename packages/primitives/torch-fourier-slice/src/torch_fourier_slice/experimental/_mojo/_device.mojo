"""GPU plumbing: the per-pixel kernels and their launchers.

Each kernel runs one thread per rfft output pixel; `FourierSliceParams` isn't
`DevicePassable`, so the kernels take its (primitive) fields as scalars and
rebuild it on the device (only the per-pixel math is shared with the CPU path).
The kernels read and write torch device memory in place -- the Python caller
passes raw device addresses (see `fourier_slice_kernels.mojo` / `experimental/_gpu.py`), so
there is no host<->device staging here.
"""

from std.math import ceildiv
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.memory import OpaquePointer

from _common import (
    BLOCK,
    BackprojectGradBuffers,
    BackprojectLine2DGradBuffers,
    BackprojectLineGradBuffers,
    Float32Ptr,
    ForwardGradBuffers,
    ForwardLine2DGradBuffers,
    ForwardLineGradBuffers,
    FourierSliceParams,
    ProjectBuffers,
    ProjectLine2DBuffers,
    ProjectLineBuffers,
    ScatterBuffers,
    ScatterLine2DBuffers,
    ScatterLineBuffers,
    WeightGradBuffers,
    WeightLine2DGradBuffers,
    WeightLineGradBuffers,
)
from _line import _project_line_pixel, _scatter_line_pixel
from _line2d import _project_line2d_pixel, _scatter_line2d_pixel
from _line2d_grad import (
    _backproject_line2d_pose_grad_pixel,
    _forward_line2d_pose_grad_pixel,
    _weight_line2d_grad_pixel,
)
from _line_grad import (
    _backproject_line_pose_grad_pixel,
    _forward_line_pose_grad_pixel,
    _weight_line_grad_pixel,
)
from _pixel import _project_pixel, _scatter_pixel
from _pose_grad import (
    _backproject_pose_grad_pixel,
    _forward_pose_grad_pixel,
    _weight_grad_pixel,
)


# ---------------------------------------------------------------------------
# Kernels (one thread per rfft pixel; rebuild `p` from primitive scalars)
# ---------------------------------------------------------------------------


def _project_gpu_kernel[
    interp: Int
](
    rec: Float32Ptr,
    rot: Float32Ptr,
    shifts_2d: Float32Ptr,
    shifts_3d: Float32Ptr,
    proj: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    bv_shift_2d: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_2d: Int,
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
    _project_pixel[interp](
        rec, rot, shifts_2d, shifts_3d, proj, vp // bp, vp % bp, y, x, p
    )


def _scatter_gpu_kernel[
    interp: Int
](
    inp: Float32Ptr,
    weights: Float32Ptr,
    rot: Float32Ptr,
    shifts_2d: Float32Ptr,
    shifts_3d: Float32Ptr,
    vol: Float32Ptr,
    wvol: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    bv_shift_2d: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_2d: Int,
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
    _scatter_pixel[interp](
        inp,
        weights,
        rot,
        shifts_2d,
        shifts_3d,
        vol,
        wvol,
        vp // bp,
        vp % bp,
        y,
        x,
        p,
    )


def _project_line_gpu_kernel[
    interp: Int
](
    rec: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    line: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
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
        1,  # bv_shift_2d (unused)
        oversampling,
        radius_cutoff_sq,
        0,  # has_shifts_2d (unused: a line has no image plane)
        interp,
        0,  # has_weights
        0,  # friedel_double
        0,  # skip_redundant
        0.0,  # ewald_curvature (unused for a 1D line)
        has_shifts_3d,
        bv_shift_3d,
    )
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _project_line_pixel[interp](
        rec, direction, shifts_3d, line, vp // bp, vp % bp, x, p
    )


def _scatter_line_gpu_kernel[
    interp: Int
](
    inp: Float32Ptr,
    weights: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    vol: Float32Ptr,
    wvol: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_weights: Int,
    friedel_double: Int,
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
        1,  # bv_shift_2d (unused)
        oversampling,
        radius_cutoff_sq,
        0,  # has_shifts_2d (unused)
        interp,
        has_weights,
        friedel_double,
        0,  # skip_redundant (a line has no redundant half to skip)
        0.0,  # ewald_curvature (unused for a 1D line)
        has_shifts_3d,
        bv_shift_3d,
    )
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _scatter_line_pixel[interp](
        inp, weights, direction, shifts_3d, vol, wvol, vp // bp, vp % bp, x, p
    )


def _project_line2d_gpu_kernel[
    interp: Int
](
    img: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    line: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_2d: Int,
    bv_shift_2d: Int,
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
        0,  # has_weights
        0,  # friedel_double
        0,  # skip_redundant
        0.0,  # ewald_curvature
        0,  # has_shifts_3d
        1,  # bv_shift_3d
    )
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _project_line2d_pixel[interp](
        img, direction, shifts_2d, line, vp // bp, vp % bp, x, p
    )


def _scatter_line2d_gpu_kernel[
    interp: Int
](
    inp: Float32Ptr,
    weights: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    vol: Float32Ptr,
    wvol: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_weights: Int,
    friedel_double: Int,
    has_shifts_2d: Int,
    bv_shift_2d: Int,
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
        0,
        0.0,
        0,
        1,
    )
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _scatter_line2d_pixel[interp](
        inp, weights, direction, shifts_2d, vol, wvol, vp // bp, vp % bp, x, p
    )


def _line2d_grad_params[
    interp: Int
](
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    friedel_double: Int,
    has_shifts_2d: Int,
    bv_shift_2d: Int,
) -> FourierSliceParams:
    """Rebuild a 2D line grad kernel's `FourierSliceParams` (unused fields zeroed).
    """
    return FourierSliceParams(
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
        friedel_double,
        0,
        0.0,
        0,
        1,
    )


def _forward_line2d_pose_grad_kernel[
    interp: Int
](
    img: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    grad_line: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_2d: Int,
    bv_shift_2d: Int,
):
    var idx = global_idx.x
    if idx >= total:
        return
    var p = _line2d_grad_params[interp](
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        oversampling,
        radius_cutoff_sq,
        0,
        has_shifts_2d,
        bv_shift_2d,
    )
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _forward_line2d_pose_grad_pixel[interp](
        img,
        direction,
        shifts_2d,
        grad_line,
        grad_dir,
        grad_shift,
        vp // bp,
        vp % bp,
        x,
        p,
    )


def _backproject_line2d_pose_grad_kernel[
    interp: Int
](
    grad_img: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    lines: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_2d: Int,
    bv_shift_2d: Int,
):
    var idx = global_idx.x
    if idx >= total:
        return
    var p = _line2d_grad_params[interp](
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        oversampling,
        radius_cutoff_sq,
        0,
        has_shifts_2d,
        bv_shift_2d,
    )
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _backproject_line2d_pose_grad_pixel[interp](
        grad_img,
        direction,
        shifts_2d,
        lines,
        grad_dir,
        grad_shift,
        vp // bp,
        vp % bp,
        x,
        p,
    )


def _weight_line2d_grad_kernel[
    interp: Int
](
    gwimg: Float32Ptr,
    direction: Float32Ptr,
    grad_weight: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    friedel_double: Int,
):
    var idx = global_idx.x
    if idx >= total:
        return
    var p = _line2d_grad_params[interp](
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        oversampling,
        radius_cutoff_sq,
        friedel_double,
        0,  # has_shifts_2d (weight grad has no shift)
        1,  # bv_shift_2d
    )
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _weight_line2d_grad_pixel[interp](
        gwimg, direction, grad_weight, vp // bp, vp % bp, x, p
    )


def _line_grad_params[
    interp: Int
](
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_weights: Int,
    friedel_double: Int,
    has_shifts_3d: Int,
    bv_shift_3d: Int,
) -> FourierSliceParams:
    """Rebuild a line kernel's `FourierSliceParams` (unused slice fields zeroed).
    """
    return FourierSliceParams(
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        1,  # bv_shift_2d (unused)
        oversampling,
        radius_cutoff_sq,
        0,  # has_shifts_2d (unused)
        interp,
        has_weights,
        friedel_double,
        0,  # skip_redundant (a line has no redundant half)
        0.0,  # ewald_curvature (unused for a 1D line)
        has_shifts_3d,
        bv_shift_3d,
    )


def _forward_line_pose_grad_kernel[
    interp: Int
](
    rec: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    grad_line: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift_3d: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_3d: Int,
    bv_shift_3d: Int,
):
    var idx = global_idx.x
    if idx >= total:
        return
    var p = _line_grad_params[interp](
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        oversampling,
        radius_cutoff_sq,
        0,
        0,
        has_shifts_3d,
        bv_shift_3d,
    )
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _forward_line_pose_grad_pixel[interp](
        rec,
        direction,
        shifts_3d,
        grad_line,
        grad_dir,
        grad_shift_3d,
        vp // bp,
        vp % bp,
        x,
        p,
    )


def _backproject_line_pose_grad_kernel[
    interp: Int
](
    grad_rec: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    lines: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift_3d: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_3d: Int,
    bv_shift_3d: Int,
):
    var idx = global_idx.x
    if idx >= total:
        return
    var p = _line_grad_params[interp](
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        oversampling,
        radius_cutoff_sq,
        0,
        0,
        has_shifts_3d,
        bv_shift_3d,
    )
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _backproject_line_pose_grad_pixel[interp](
        grad_rec,
        direction,
        shifts_3d,
        lines,
        grad_dir,
        grad_shift_3d,
        vp // bp,
        vp % bp,
        x,
        p,
    )


def _weight_line_grad_kernel[
    interp: Int
](
    gwvol: Float32Ptr,
    direction: Float32Ptr,
    grad_weight: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    friedel_double: Int,
):
    var idx = global_idx.x
    if idx >= total:
        return
    var p = _line_grad_params[interp](
        bp,
        sidelength,
        proj_sidelength,
        bv_rot,
        oversampling,
        radius_cutoff_sq,
        0,
        friedel_double,
        0,
        1,
    )
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _weight_line_grad_pixel[interp](
        gwvol, direction, grad_weight, vp // bp, vp % bp, x, p
    )


def _forward_pose_grad_kernel[
    interp: Int
](
    rec: Float32Ptr,
    rot: Float32Ptr,
    shifts_2d: Float32Ptr,
    shifts_3d: Float32Ptr,
    grad_proj: Float32Ptr,
    grad_rot: Float32Ptr,
    grad_shift: Float32Ptr,
    grad_shift_3d: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    bv_shift_2d: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_2d: Int,
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
    _forward_pose_grad_pixel[interp](
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


def _backproject_pose_grad_kernel[
    interp: Int
](
    grad_rec: Float32Ptr,
    rot: Float32Ptr,
    shifts_2d: Float32Ptr,
    shifts_3d: Float32Ptr,
    proj: Float32Ptr,
    grad_rot: Float32Ptr,
    grad_shift: Float32Ptr,
    grad_shift_3d: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    bv_shift_2d: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
    has_shifts_2d: Int,
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
    _backproject_pose_grad_pixel[interp](
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


def _weight_grad_kernel[
    interp: Int
](
    gwvol: Float32Ptr,
    rot: Float32Ptr,
    grad_weight: Float32Ptr,
    total: Int,
    bp: Int,
    sidelength: Int,
    proj_sidelength: Int,
    bv_rot: Int,
    bv_shift_2d: Int,
    oversampling: Float32,
    radius_cutoff_sq: Float32,
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
    _weight_grad_pixel[interp](
        gwvol, rot, grad_weight, vp // bp, vp % bp, y, x, p
    )


# ---------------------------------------------------------------------------
# Launchers (unpack `p` to device scalars)
#
# `stream_addr` selects the GPU stream the kernel is enqueued on:
#   != 0 : a foreign (torch) stream address (CUDA CUstream). Enqueuing on it
#          orders the kernel with the surrounding torch ops directly, so no full
#          device sync is needed -- the caller relies on torch's own stream.
#   == 0 : the DeviceContext's own stream (the Metal path; the caller syncs the
#          context afterwards, since Metal has no external-stream handoff).
# `ctx.stream()` and `create_external_stream(...)` are the same stream type, so
# one enqueue path serves both.
# ---------------------------------------------------------------------------


@always_inline
def _launch_project[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ProjectBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        # CUDA: enqueue on torch's stream (Metal has no external-stream API).
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_project_gpu_kernel[interp]]()
        stream.enqueue_function(
            compiled,
            buffers.rec,
            buffers.rot,
            buffers.shifts_2d,
            buffers.shifts_3d,
            buffers.proj,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.bv_shift_2d,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_shifts_2d,
            p.ewald_curvature,
            p.has_shifts_3d,
            p.bv_shift_3d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_project_gpu_kernel[interp]](
        buffers.rec,
        buffers.rot,
        buffers.shifts_2d,
        buffers.shifts_3d,
        buffers.proj,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.bv_shift_2d,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_2d,
        p.ewald_curvature,
        p.has_shifts_3d,
        p.bv_shift_3d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_scatter[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ScatterBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_scatter_gpu_kernel[interp]]()
        stream.enqueue_function(
            compiled,
            buffers.inp,
            buffers.weights,
            buffers.rot,
            buffers.shifts_2d,
            buffers.shifts_3d,
            buffers.vol,
            buffers.wvol,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.bv_shift_2d,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_shifts_2d,
            p.has_weights,
            p.friedel_double,
            p.skip_redundant,
            p.ewald_curvature,
            p.has_shifts_3d,
            p.bv_shift_3d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_scatter_gpu_kernel[interp]](
        buffers.inp,
        buffers.weights,
        buffers.rot,
        buffers.shifts_2d,
        buffers.shifts_3d,
        buffers.vol,
        buffers.wvol,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.bv_shift_2d,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_2d,
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
def _launch_project_line[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ProjectLineBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_project_line_gpu_kernel[interp]]()
        stream.enqueue_function(
            compiled,
            buffers.rec,
            buffers.direction,
            buffers.shifts_3d,
            buffers.line,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_shifts_3d,
            p.bv_shift_3d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_project_line_gpu_kernel[interp]](
        buffers.rec,
        buffers.direction,
        buffers.shifts_3d,
        buffers.line,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_3d,
        p.bv_shift_3d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_scatter_line[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ScatterLineBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_scatter_line_gpu_kernel[interp]]()
        stream.enqueue_function(
            compiled,
            buffers.inp,
            buffers.weights,
            buffers.direction,
            buffers.shifts_3d,
            buffers.vol,
            buffers.wvol,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_weights,
            p.friedel_double,
            p.has_shifts_3d,
            p.bv_shift_3d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_scatter_line_gpu_kernel[interp]](
        buffers.inp,
        buffers.weights,
        buffers.direction,
        buffers.shifts_3d,
        buffers.vol,
        buffers.wvol,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_weights,
        p.friedel_double,
        p.has_shifts_3d,
        p.bv_shift_3d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_project_line2d[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ProjectLine2DBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _project_line2d_gpu_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.img,
            buffers.direction,
            buffers.shifts_2d,
            buffers.line,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_shifts_2d,
            p.bv_shift_2d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_project_line2d_gpu_kernel[interp]](
        buffers.img,
        buffers.direction,
        buffers.shifts_2d,
        buffers.line,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_2d,
        p.bv_shift_2d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_scatter_line2d[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ScatterLine2DBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _scatter_line2d_gpu_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.inp,
            buffers.weights,
            buffers.direction,
            buffers.shifts_2d,
            buffers.vol,
            buffers.wvol,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_weights,
            p.friedel_double,
            p.has_shifts_2d,
            p.bv_shift_2d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_scatter_line2d_gpu_kernel[interp]](
        buffers.inp,
        buffers.weights,
        buffers.direction,
        buffers.shifts_2d,
        buffers.vol,
        buffers.wvol,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_weights,
        p.friedel_double,
        p.has_shifts_2d,
        p.bv_shift_2d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_forward_line2d_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ForwardLine2DGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _forward_line2d_pose_grad_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.img,
            buffers.direction,
            buffers.shifts_2d,
            buffers.grad_line,
            buffers.grad_dir,
            buffers.grad_shift,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_shifts_2d,
            p.bv_shift_2d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_forward_line2d_pose_grad_kernel[interp]](
        buffers.img,
        buffers.direction,
        buffers.shifts_2d,
        buffers.grad_line,
        buffers.grad_dir,
        buffers.grad_shift,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_2d,
        p.bv_shift_2d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_backproject_line2d_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: BackprojectLine2DGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _backproject_line2d_pose_grad_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.grad_img,
            buffers.direction,
            buffers.shifts_2d,
            buffers.lines,
            buffers.grad_dir,
            buffers.grad_shift,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_shifts_2d,
            p.bv_shift_2d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_backproject_line2d_pose_grad_kernel[interp]](
        buffers.grad_img,
        buffers.direction,
        buffers.shifts_2d,
        buffers.lines,
        buffers.grad_dir,
        buffers.grad_shift,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_2d,
        p.bv_shift_2d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_weight_line2d_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: WeightLine2DGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _weight_line2d_grad_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.gwimg,
            buffers.direction,
            buffers.grad_weight,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.oversampling,
            p.radius_cutoff_sq,
            p.friedel_double,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_weight_line2d_grad_kernel[interp]](
        buffers.gwimg,
        buffers.direction,
        buffers.grad_weight,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.oversampling,
        p.radius_cutoff_sq,
        p.friedel_double,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_forward_line_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ForwardLineGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _forward_line_pose_grad_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.rec,
            buffers.direction,
            buffers.shifts_3d,
            buffers.grad_line,
            buffers.grad_dir,
            buffers.grad_shift_3d,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_shifts_3d,
            p.bv_shift_3d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_forward_line_pose_grad_kernel[interp]](
        buffers.rec,
        buffers.direction,
        buffers.shifts_3d,
        buffers.grad_line,
        buffers.grad_dir,
        buffers.grad_shift_3d,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_3d,
        p.bv_shift_3d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_backproject_line_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: BackprojectLineGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _backproject_line_pose_grad_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.grad_rec,
            buffers.direction,
            buffers.shifts_3d,
            buffers.lines,
            buffers.grad_dir,
            buffers.grad_shift_3d,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_shifts_3d,
            p.bv_shift_3d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_backproject_line_pose_grad_kernel[interp]](
        buffers.grad_rec,
        buffers.direction,
        buffers.shifts_3d,
        buffers.lines,
        buffers.grad_dir,
        buffers.grad_shift_3d,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_3d,
        p.bv_shift_3d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_weight_line_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: WeightLineGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_weight_line_grad_kernel[interp]]()
        stream.enqueue_function(
            compiled,
            buffers.gwvol,
            buffers.direction,
            buffers.grad_weight,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.oversampling,
            p.radius_cutoff_sq,
            p.friedel_double,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_weight_line_grad_kernel[interp]](
        buffers.gwvol,
        buffers.direction,
        buffers.grad_weight,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.oversampling,
        p.radius_cutoff_sq,
        p.friedel_double,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_forward_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ForwardGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_forward_pose_grad_kernel[interp]]()
        stream.enqueue_function(
            compiled,
            buffers.rec,
            buffers.rot,
            buffers.shifts_2d,
            buffers.shifts_3d,
            buffers.grad_proj,
            buffers.grad_rot,
            buffers.grad_shift,
            buffers.grad_shift_3d,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.bv_shift_2d,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_shifts_2d,
            p.ewald_curvature,
            p.has_shifts_3d,
            p.bv_shift_3d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_forward_pose_grad_kernel[interp]](
        buffers.rec,
        buffers.rot,
        buffers.shifts_2d,
        buffers.shifts_3d,
        buffers.grad_proj,
        buffers.grad_rot,
        buffers.grad_shift,
        buffers.grad_shift_3d,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.bv_shift_2d,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_2d,
        p.ewald_curvature,
        p.has_shifts_3d,
        p.bv_shift_3d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_backproject_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: BackprojectGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _backproject_pose_grad_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.grad_rec,
            buffers.rot,
            buffers.shifts_2d,
            buffers.shifts_3d,
            buffers.proj,
            buffers.grad_rot,
            buffers.grad_shift,
            buffers.grad_shift_3d,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.bv_shift_2d,
            p.oversampling,
            p.radius_cutoff_sq,
            p.has_shifts_2d,
            p.ewald_curvature,
            p.has_shifts_3d,
            p.bv_shift_3d,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_backproject_pose_grad_kernel[interp]](
        buffers.grad_rec,
        buffers.rot,
        buffers.shifts_2d,
        buffers.shifts_3d,
        buffers.proj,
        buffers.grad_rot,
        buffers.grad_shift,
        buffers.grad_shift_3d,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.bv_shift_2d,
        p.oversampling,
        p.radius_cutoff_sq,
        p.has_shifts_2d,
        p.ewald_curvature,
        p.has_shifts_3d,
        p.bv_shift_3d,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_weight_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: WeightGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_weight_grad_kernel[interp]]()
        stream.enqueue_function(
            compiled,
            buffers.gwvol,
            buffers.rot,
            buffers.grad_weight,
            total,
            p.bp,
            p.sidelength,
            p.proj_sidelength,
            p.bv_rot,
            p.bv_shift_2d,
            p.oversampling,
            p.radius_cutoff_sq,
            p.friedel_double,
            p.ewald_curvature,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_weight_grad_kernel[interp]](
        buffers.gwvol,
        buffers.rot,
        buffers.grad_weight,
        total,
        p.bp,
        p.sidelength,
        p.proj_sidelength,
        p.bv_rot,
        p.bv_shift_2d,
        p.oversampling,
        p.radius_cutoff_sq,
        p.friedel_double,
        p.ewald_curvature,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )
