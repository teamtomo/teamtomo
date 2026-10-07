"""GPU plumbing: the per-pixel kernels and their launchers.

Each kernel runs one thread per rfft output pixel. `FourierSliceParams` itself
isn't `DevicePassable` (`Int` isn't a fixed-width kernel-argument type), so
each launcher packs it into one `DeviceParams` (see `_common.mojo`) and every
kernel takes that as its single params argument, unpacking it back to a
`FourierSliceParams` on entry -- the per-pixel math is shared with the CPU
path unchanged. The kernels read and write torch device memory in place --
the Python caller passes raw device addresses (see `fourier_slice_kernels.mojo`
/ `_backend/_device.py`), so there is no host<->device staging here.
"""

from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.math import ceildiv
from std.memory import OpaquePointer

from max.gpu.host import DeviceContext

from _common import (
    BLOCK,
    SCATTER_BLOCK,
    SCATTER_COARSEN_LINEAR,
    _grad_add,
    _line_2d_pose_grad_offsets,
    _line_3d_pose_grad_offsets,
    _slice_3d_pose_grad_offsets,
    _scatter_coarsen,
    _warp_pose_uniform,
    InsertSlice3DPoseGradBuffers,
    InsertLine2DPoseGradBuffers,
    InsertLine3DPoseGradBuffers,
    DeviceParams,
    Float32Ptr,
    ExtractSlice3DPoseGradBuffers,
    ExtractLine2DPoseGradBuffers,
    ExtractLine3DPoseGradBuffers,
    FourierSliceParams,
    ExtractSlice3DBuffers,
    ExtractLine2DBuffers,
    ExtractLine3DBuffers,
    InsertSlice3DBuffers,
    InsertLine2DBuffers,
    InsertLine3DBuffers,
    InsertSlice3DWeightGradBuffers,
    InsertLine2DWeightGradBuffers,
    InsertLine3DWeightGradBuffers,
)
from _line_3d import _extract_line_3d_pixel, _insert_line_3d_pixel
from _line_2d import _extract_line_2d_pixel, _insert_line_2d_pixel
from _line_2d_grad import (
    _insert_line_2d_pose_grad_pixel,
    _extract_line_2d_pose_grad_pixel,
    _insert_line_2d_weight_grad_pixel,
)
from _line_3d_grad import (
    _insert_line_3d_pose_grad_pixel,
    _extract_line_3d_pose_grad_pixel,
    _insert_line_3d_weight_grad_pixel,
)
from _slice_3d import _extract_slice_3d_pixel, _insert_slice_3d_pixel
from _slice_3d_grad import (
    _insert_slice_3d_pose_grad_pixel,
    _extract_slice_3d_pose_grad_pixel,
    _insert_slice_3d_weight_grad_pixel,
)


# ---------------------------------------------------------------------------
# Kernels (one thread per rfft pixel; unpack `dp` back to `p` on entry)
# ---------------------------------------------------------------------------


def _extract_slice_3d_gpu_kernel[
    interp: Int
](
    rec: Float32Ptr,
    rot: Float32Ptr,
    shifts_2d: Float32Ptr,
    shifts_3d: Float32Ptr,
    proj: Float32Ptr,
    dp: DeviceParams,
):
    var idx = global_idx.x
    if idx >= Int(dp.total):
        return
    var p = dp.to_params(interp)
    var psh = p.proj_sidelength_half()
    var x = idx % psh
    var t = idx // psh
    var y = t % p.proj_sidelength
    var vp = t // p.proj_sidelength
    _extract_slice_3d_pixel[interp](
        rec, rot, shifts_2d, shifts_3d, proj, vp // p.bp, vp % p.bp, y, x, p
    )


def _insert_slice_3d_gpu_kernel[
    interp: Int, coarsen: Int
](
    inp: Float32Ptr,
    weights: Float32Ptr,
    rot: Float32Ptr,
    shifts_2d: Float32Ptr,
    shifts_3d: Float32Ptr,
    vol: Float32Ptr,
    wvol: Float32Ptr,
    dp: DeviceParams,
):
    # Atomic-scatter kernel: each thread handles `coarsen` consecutive-in-warp
    # pixels (block_base + k*block_dim + tid) rather than one, so the
    # (cheap, shared) `FourierSliceParams`/`proj_sidelength_half` setup below
    # is amortised across several scatters instead of redone per pixel, and
    # the unrolled `comptime for` exposes ILP across independent atomic adds.
    var p = dp.to_params(interp)
    var total = Int(dp.total)
    var psh = p.proj_sidelength_half()
    var block_base = block_idx.x * block_dim.x * coarsen
    var tid = thread_idx.x

    comptime for k in range(coarsen):
        var idx = block_base + k * block_dim.x + tid
        if idx < total:
            var x = idx % psh
            var t = idx // psh
            var y = t % p.proj_sidelength
            var vp = t // p.proj_sidelength
            _insert_slice_3d_pixel[interp](
                inp,
                weights,
                rot,
                shifts_2d,
                shifts_3d,
                vol,
                wvol,
                vp // p.bp,
                vp % p.bp,
                y,
                x,
                p,
            )


def _extract_line_3d_gpu_kernel[
    interp: Int
](
    rec: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    line: Float32Ptr,
    dp: DeviceParams,
):
    var idx = global_idx.x
    if idx >= Int(dp.total):
        return
    var p = dp.to_params(interp)
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _extract_line_3d_pixel[interp](
        rec, direction, shifts_3d, line, vp // p.bp, vp % p.bp, x, p
    )


def _insert_line_3d_gpu_kernel[
    interp: Int, coarsen: Int
](
    inp: Float32Ptr,
    weights: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    vol: Float32Ptr,
    wvol: Float32Ptr,
    dp: DeviceParams,
):
    var p = dp.to_params(interp)
    var total = Int(dp.total)
    var lsh = p.proj_sidelength_half()
    var block_base = block_idx.x * block_dim.x * coarsen
    var tid = thread_idx.x

    comptime for k in range(coarsen):
        var idx = block_base + k * block_dim.x + tid
        if idx < total:
            var x = idx % lsh
            var vp = idx // lsh
            _insert_line_3d_pixel[interp](
                inp,
                weights,
                direction,
                shifts_3d,
                vol,
                wvol,
                vp // p.bp,
                vp % p.bp,
                x,
                p,
            )


def _extract_line_2d_gpu_kernel[
    interp: Int
](
    img: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    line: Float32Ptr,
    dp: DeviceParams,
):
    var idx = global_idx.x
    if idx >= Int(dp.total):
        return
    var p = dp.to_params(interp)
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _extract_line_2d_pixel[interp](
        img, direction, shifts_2d, line, vp // p.bp, vp % p.bp, x, p
    )


def _insert_line_2d_gpu_kernel[
    interp: Int, coarsen: Int
](
    inp: Float32Ptr,
    weights: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    vol: Float32Ptr,
    wvol: Float32Ptr,
    dp: DeviceParams,
):
    var p = dp.to_params(interp)
    var total = Int(dp.total)
    var lsh = p.proj_sidelength_half()
    var block_base = block_idx.x * block_dim.x * coarsen
    var tid = thread_idx.x

    comptime for k in range(coarsen):
        var idx = block_base + k * block_dim.x + tid
        if idx < total:
            var x = idx % lsh
            var vp = idx // lsh
            _insert_line_2d_pixel[interp](
                inp,
                weights,
                direction,
                shifts_2d,
                vol,
                wvol,
                vp // p.bp,
                vp % p.bp,
                x,
                p,
            )


def _extract_line_2d_pose_grad_kernel[
    interp: Int
](
    img: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    grad_line: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift: Float32Ptr,
    dp: DeviceParams,
):
    # Same per-pose atomic-contention fix as _extract_slice_3d_pose_grad_kernel: reduce
    # across a warp before one atomic add per warp; clamp-and-mask instead of
    # early-return for out-of-bounds threads to keep the warp uniform.
    var idx = global_idx.x
    var total = Int(dp.total)
    var in_bounds = idx < total
    var idx_safe = idx if in_bounds else total - 1
    var p = dp.to_params(interp)
    var lsh = p.proj_sidelength_half()
    var x = idx_safe % lsh
    var vp = idx_safe // lsh
    var i_bv = vp // p.bp
    var i_bp = vp % p.bp
    var contrib = _extract_line_2d_pose_grad_pixel[interp](
        img, direction, shifts_2d, grad_line, i_bv, i_bp, x, p
    )
    if not in_bounds:
        contrib = SIMD[DType.float32, 4](0)
    var uniform = _warp_pose_uniform(vp)
    var dbase, sbase = _line_2d_pose_grad_offsets(i_bv, i_bp, p)
    _grad_add(grad_dir, dbase + 0, contrib[0], uniform)
    _grad_add(grad_dir, dbase + 1, contrib[1], uniform)
    if p.has_shifts_2d != 0:
        _grad_add(grad_shift, sbase + 0, contrib[2], uniform)
        _grad_add(grad_shift, sbase + 1, contrib[3], uniform)


def _insert_line_2d_pose_grad_kernel[
    interp: Int
](
    grad_img: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    lines: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift: Float32Ptr,
    dp: DeviceParams,
):
    # See _extract_line_2d_pose_grad_kernel above for the reduction rationale.
    var idx = global_idx.x
    var total = Int(dp.total)
    var in_bounds = idx < total
    var idx_safe = idx if in_bounds else total - 1
    var p = dp.to_params(interp)
    var lsh = p.proj_sidelength_half()
    var x = idx_safe % lsh
    var vp = idx_safe // lsh
    var i_bv = vp // p.bp
    var i_bp = vp % p.bp
    var contrib = _insert_line_2d_pose_grad_pixel[interp](
        grad_img, direction, shifts_2d, lines, i_bv, i_bp, x, p
    )
    if not in_bounds:
        contrib = SIMD[DType.float32, 4](0)
    var uniform = _warp_pose_uniform(vp)
    var dbase, sbase = _line_2d_pose_grad_offsets(i_bv, i_bp, p)
    _grad_add(grad_dir, dbase + 0, contrib[0], uniform)
    _grad_add(grad_dir, dbase + 1, contrib[1], uniform)
    if p.has_shifts_2d != 0:
        _grad_add(grad_shift, sbase + 0, contrib[2], uniform)
        _grad_add(grad_shift, sbase + 1, contrib[3], uniform)


def _insert_line_2d_weight_grad_kernel[
    interp: Int
](
    gwimg: Float32Ptr,
    direction: Float32Ptr,
    grad_weight: Float32Ptr,
    dp: DeviceParams,
):
    var idx = global_idx.x
    if idx >= Int(dp.total):
        return
    var p = dp.to_params(interp)
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _insert_line_2d_weight_grad_pixel[interp](
        gwimg, direction, grad_weight, vp // p.bp, vp % p.bp, x, p
    )


def _extract_line_3d_pose_grad_kernel[
    interp: Int
](
    rec: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    grad_line: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift_3d: Float32Ptr,
    dp: DeviceParams,
):
    # Same per-pose atomic-contention fix as _extract_slice_3d_pose_grad_kernel: reduce
    # across a warp before one atomic add per warp; clamp-and-mask instead of
    # early-return for out-of-bounds threads to keep the warp uniform.
    var idx = global_idx.x
    var total = Int(dp.total)
    var in_bounds = idx < total
    var idx_safe = idx if in_bounds else total - 1
    var p = dp.to_params(interp)
    var lsh = p.proj_sidelength_half()
    var x = idx_safe % lsh
    var vp = idx_safe // lsh
    var i_bv = vp // p.bp
    var i_bp = vp % p.bp
    var contrib = _extract_line_3d_pose_grad_pixel[interp](
        rec, direction, shifts_3d, grad_line, i_bv, i_bp, x, p
    )
    if not in_bounds:
        contrib = SIMD[DType.float32, 8](0)
    var uniform = _warp_pose_uniform(vp)
    var dbase, s3base = _line_3d_pose_grad_offsets(i_bv, i_bp, p)
    _grad_add(grad_dir, dbase + 0, contrib[0], uniform)
    _grad_add(grad_dir, dbase + 1, contrib[1], uniform)
    _grad_add(grad_dir, dbase + 2, contrib[2], uniform)
    if p.has_shifts_3d != 0:
        _grad_add(grad_shift_3d, s3base + 0, contrib[3], uniform)
        _grad_add(grad_shift_3d, s3base + 1, contrib[4], uniform)
        _grad_add(grad_shift_3d, s3base + 2, contrib[5], uniform)


def _insert_line_3d_pose_grad_kernel[
    interp: Int
](
    grad_rec: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    lines: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift_3d: Float32Ptr,
    dp: DeviceParams,
):
    # See _extract_line_3d_pose_grad_kernel above for the reduction rationale.
    var idx = global_idx.x
    var total = Int(dp.total)
    var in_bounds = idx < total
    var idx_safe = idx if in_bounds else total - 1
    var p = dp.to_params(interp)
    var lsh = p.proj_sidelength_half()
    var x = idx_safe % lsh
    var vp = idx_safe // lsh
    var i_bv = vp // p.bp
    var i_bp = vp % p.bp
    var contrib = _insert_line_3d_pose_grad_pixel[interp](
        grad_rec, direction, shifts_3d, lines, i_bv, i_bp, x, p
    )
    if not in_bounds:
        contrib = SIMD[DType.float32, 8](0)
    var uniform = _warp_pose_uniform(vp)
    var dbase, s3base = _line_3d_pose_grad_offsets(i_bv, i_bp, p)
    _grad_add(grad_dir, dbase + 0, contrib[0], uniform)
    _grad_add(grad_dir, dbase + 1, contrib[1], uniform)
    _grad_add(grad_dir, dbase + 2, contrib[2], uniform)
    if p.has_shifts_3d != 0:
        _grad_add(grad_shift_3d, s3base + 0, contrib[3], uniform)
        _grad_add(grad_shift_3d, s3base + 1, contrib[4], uniform)
        _grad_add(grad_shift_3d, s3base + 2, contrib[5], uniform)


def _insert_line_3d_weight_grad_kernel[
    interp: Int
](
    gwvol: Float32Ptr,
    direction: Float32Ptr,
    grad_weight: Float32Ptr,
    dp: DeviceParams,
):
    var idx = global_idx.x
    if idx >= Int(dp.total):
        return
    var p = dp.to_params(interp)
    var lsh = p.proj_sidelength_half()
    var x = idx % lsh
    var vp = idx // lsh
    _insert_line_3d_weight_grad_pixel[interp](
        gwvol, direction, grad_weight, vp // p.bp, vp % p.bp, x, p
    )


def _extract_slice_3d_pose_grad_kernel[
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
    dp: DeviceParams,
):
    # Every pixel of a pose targets the SAME ~14-scalar gradient accumulator (far
    # more contended than the volume/projection scatter's spread-out targets), so
    # this kernel reduces across a warp before one atomic add per warp -- see the
    # module comment on `_grad_add` in _common.mojo. That requires every lane of
    # the warp to uniformly reach the reduction, so an out-of-bounds thread clamps
    # its pixel index instead of returning early, and its contribution is zeroed
    # out afterward rather than skipped.
    var idx = global_idx.x
    var total = Int(dp.total)
    var in_bounds = idx < total
    var idx_safe = idx if in_bounds else total - 1
    var p = dp.to_params(interp)
    var psh = p.proj_sidelength_half()
    var x = idx_safe % psh
    var t = idx_safe // psh
    var y = t % p.proj_sidelength
    var vp = t // p.proj_sidelength
    var i_bv = vp // p.bp
    var i_bp = vp % p.bp
    var contrib = _extract_slice_3d_pose_grad_pixel[interp](
        rec, rot, shifts_2d, shifts_3d, grad_proj, i_bv, i_bp, y, x, p
    )
    if not in_bounds:
        contrib = SIMD[DType.float32, 16](0)
    var uniform = _warp_pose_uniform(vp)
    var rbase, sbase, s3base = _slice_3d_pose_grad_offsets(i_bv, i_bp, p)
    _grad_add(grad_rot, rbase + 1, contrib[1], uniform)
    _grad_add(grad_rot, rbase + 2, contrib[2], uniform)
    _grad_add(grad_rot, rbase + 4, contrib[4], uniform)
    _grad_add(grad_rot, rbase + 5, contrib[5], uniform)
    _grad_add(grad_rot, rbase + 7, contrib[7], uniform)
    _grad_add(grad_rot, rbase + 8, contrib[8], uniform)
    if p.ewald_curvature != 0.0:
        _grad_add(grad_rot, rbase + 0, contrib[0], uniform)
        _grad_add(grad_rot, rbase + 3, contrib[3], uniform)
        _grad_add(grad_rot, rbase + 6, contrib[6], uniform)
    if p.has_shifts_2d != 0:
        _grad_add(grad_shift, sbase + 0, contrib[9], uniform)
        _grad_add(grad_shift, sbase + 1, contrib[10], uniform)
    if p.has_shifts_3d != 0:
        _grad_add(grad_shift_3d, s3base + 0, contrib[11], uniform)
        _grad_add(grad_shift_3d, s3base + 1, contrib[12], uniform)
        _grad_add(grad_shift_3d, s3base + 2, contrib[13], uniform)


def _insert_slice_3d_pose_grad_kernel[
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
    dp: DeviceParams,
):
    # See the comment in _extract_slice_3d_pose_grad_kernel: same per-pose atomic-contention
    # fix, with the same clamp-and-mask instead of early-return for out-of-bounds
    # threads to keep the warp uniformly reaching the reduction.
    var idx = global_idx.x
    var total = Int(dp.total)
    var in_bounds = idx < total
    var idx_safe = idx if in_bounds else total - 1
    var p = dp.to_params(interp)
    var psh = p.proj_sidelength_half()
    var x = idx_safe % psh
    var t = idx_safe // psh
    var y = t % p.proj_sidelength
    var vp = t // p.proj_sidelength
    var i_bv = vp // p.bp
    var i_bp = vp % p.bp
    var contrib = _insert_slice_3d_pose_grad_pixel[interp](
        grad_rec, rot, shifts_2d, shifts_3d, proj, i_bv, i_bp, y, x, p
    )
    if not in_bounds:
        contrib = SIMD[DType.float32, 16](0)
    var uniform = _warp_pose_uniform(vp)
    var rbase, sbase, s3base = _slice_3d_pose_grad_offsets(i_bv, i_bp, p)
    _grad_add(grad_rot, rbase + 1, contrib[1], uniform)
    _grad_add(grad_rot, rbase + 2, contrib[2], uniform)
    _grad_add(grad_rot, rbase + 4, contrib[4], uniform)
    _grad_add(grad_rot, rbase + 5, contrib[5], uniform)
    _grad_add(grad_rot, rbase + 7, contrib[7], uniform)
    _grad_add(grad_rot, rbase + 8, contrib[8], uniform)
    if p.ewald_curvature != 0.0:
        _grad_add(grad_rot, rbase + 0, contrib[0], uniform)
        _grad_add(grad_rot, rbase + 3, contrib[3], uniform)
        _grad_add(grad_rot, rbase + 6, contrib[6], uniform)
    if p.has_shifts_2d != 0:
        _grad_add(grad_shift, sbase + 0, contrib[9], uniform)
        _grad_add(grad_shift, sbase + 1, contrib[10], uniform)
    if p.has_shifts_3d != 0:
        _grad_add(grad_shift_3d, s3base + 0, contrib[11], uniform)
        _grad_add(grad_shift_3d, s3base + 1, contrib[12], uniform)
        _grad_add(grad_shift_3d, s3base + 2, contrib[13], uniform)


def _insert_slice_3d_weight_grad_kernel[
    interp: Int
](
    gwvol: Float32Ptr,
    rot: Float32Ptr,
    grad_weight: Float32Ptr,
    dp: DeviceParams,
):
    var idx = global_idx.x
    if idx >= Int(dp.total):
        return
    var p = dp.to_params(interp)
    var psh = p.proj_sidelength_half()
    var x = idx % psh
    var t = idx // psh
    var y = t % p.proj_sidelength
    var vp = t // p.proj_sidelength
    _insert_slice_3d_weight_grad_pixel[interp](
        gwvol, rot, grad_weight, vp // p.bp, vp % p.bp, y, x, p
    )


# ---------------------------------------------------------------------------
# Launchers (pack `p` + `total` into one `DeviceParams` kernel argument)
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
def _launch_extract_slice_3d[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ExtractSlice3DBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        # CUDA: enqueue on torch's stream (Metal has no external-stream API).
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_extract_slice_3d_gpu_kernel[interp]]()
        stream.enqueue_function(
            compiled,
            buffers.rec,
            buffers.rot,
            buffers.shifts_2d,
            buffers.shifts_3d,
            buffers.proj,
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_extract_slice_3d_gpu_kernel[interp]](
        buffers.rec,
        buffers.rot,
        buffers.shifts_2d,
        buffers.shifts_3d,
        buffers.proj,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_insert_slice_3d[
    interp: Int
](
    ctx: DeviceContext,
    buffers: InsertSlice3DBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    comptime coarsen = _scatter_coarsen[interp]()
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _insert_slice_3d_gpu_kernel[interp, coarsen]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.inp,
            buffers.weights,
            buffers.rot,
            buffers.shifts_2d,
            buffers.shifts_3d,
            buffers.vol,
            buffers.wvol,
            dp,
            grid_dim=ceildiv(total, SCATTER_BLOCK * coarsen),
            block_dim=SCATTER_BLOCK,
        )
        return
    ctx.enqueue_function[_insert_slice_3d_gpu_kernel[interp, coarsen]](
        buffers.inp,
        buffers.weights,
        buffers.rot,
        buffers.shifts_2d,
        buffers.shifts_3d,
        buffers.vol,
        buffers.wvol,
        dp,
        grid_dim=ceildiv(total, SCATTER_BLOCK * coarsen),
        block_dim=SCATTER_BLOCK,
    )


@always_inline
def _launch_extract_line_3d[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ExtractLine3DBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_extract_line_3d_gpu_kernel[interp]]()
        stream.enqueue_function(
            compiled,
            buffers.rec,
            buffers.direction,
            buffers.shifts_3d,
            buffers.line,
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_extract_line_3d_gpu_kernel[interp]](
        buffers.rec,
        buffers.direction,
        buffers.shifts_3d,
        buffers.line,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_insert_line_3d[
    interp: Int
](
    ctx: DeviceContext,
    buffers: InsertLine3DBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    comptime coarsen = _scatter_coarsen[interp]()
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _insert_line_3d_gpu_kernel[interp, coarsen]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.inp,
            buffers.weights,
            buffers.direction,
            buffers.shifts_3d,
            buffers.vol,
            buffers.wvol,
            dp,
            grid_dim=ceildiv(total, SCATTER_BLOCK * coarsen),
            block_dim=SCATTER_BLOCK,
        )
        return
    ctx.enqueue_function[_insert_line_3d_gpu_kernel[interp, coarsen]](
        buffers.inp,
        buffers.weights,
        buffers.direction,
        buffers.shifts_3d,
        buffers.vol,
        buffers.wvol,
        dp,
        grid_dim=ceildiv(total, SCATTER_BLOCK * coarsen),
        block_dim=SCATTER_BLOCK,
    )


@always_inline
def _launch_extract_line_2d[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ExtractLine2DBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _extract_line_2d_gpu_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.img,
            buffers.direction,
            buffers.shifts_2d,
            buffers.line,
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_extract_line_2d_gpu_kernel[interp]](
        buffers.img,
        buffers.direction,
        buffers.shifts_2d,
        buffers.line,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_insert_line_2d[
    interp: Int
](
    ctx: DeviceContext,
    buffers: InsertLine2DBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    # Bicubic 2D-line splat is only 4x4 = 16 corners/pixel -- measured slower with
    # coarsening (see the SCATTER_COARSEN comment in _common.mojo), unlike tricubic
    # 3D's 64 corners. Always use the uncoarsened setting here, regardless of interp.
    comptime coarsen = SCATTER_COARSEN_LINEAR
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _insert_line_2d_gpu_kernel[interp, coarsen]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.inp,
            buffers.weights,
            buffers.direction,
            buffers.shifts_2d,
            buffers.vol,
            buffers.wvol,
            dp,
            grid_dim=ceildiv(total, SCATTER_BLOCK * coarsen),
            block_dim=SCATTER_BLOCK,
        )
        return
    ctx.enqueue_function[_insert_line_2d_gpu_kernel[interp, coarsen]](
        buffers.inp,
        buffers.weights,
        buffers.direction,
        buffers.shifts_2d,
        buffers.vol,
        buffers.wvol,
        dp,
        grid_dim=ceildiv(total, SCATTER_BLOCK * coarsen),
        block_dim=SCATTER_BLOCK,
    )


@always_inline
def _launch_extract_line_2d_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ExtractLine2DPoseGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _extract_line_2d_pose_grad_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.img,
            buffers.direction,
            buffers.shifts_2d,
            buffers.grad_line,
            buffers.grad_dir,
            buffers.grad_shift,
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_extract_line_2d_pose_grad_kernel[interp]](
        buffers.img,
        buffers.direction,
        buffers.shifts_2d,
        buffers.grad_line,
        buffers.grad_dir,
        buffers.grad_shift,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_insert_line_2d_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: InsertLine2DPoseGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _insert_line_2d_pose_grad_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.grad_img,
            buffers.direction,
            buffers.shifts_2d,
            buffers.lines,
            buffers.grad_dir,
            buffers.grad_shift,
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_insert_line_2d_pose_grad_kernel[interp]](
        buffers.grad_img,
        buffers.direction,
        buffers.shifts_2d,
        buffers.lines,
        buffers.grad_dir,
        buffers.grad_shift,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_insert_line_2d_weight_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: InsertLine2DWeightGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _insert_line_2d_weight_grad_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.gwimg,
            buffers.direction,
            buffers.grad_weight,
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_insert_line_2d_weight_grad_kernel[interp]](
        buffers.gwimg,
        buffers.direction,
        buffers.grad_weight,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_extract_line_3d_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ExtractLine3DPoseGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _extract_line_3d_pose_grad_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.rec,
            buffers.direction,
            buffers.shifts_3d,
            buffers.grad_line,
            buffers.grad_dir,
            buffers.grad_shift_3d,
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_extract_line_3d_pose_grad_kernel[interp]](
        buffers.rec,
        buffers.direction,
        buffers.shifts_3d,
        buffers.grad_line,
        buffers.grad_dir,
        buffers.grad_shift_3d,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_insert_line_3d_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: InsertLine3DPoseGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _insert_line_3d_pose_grad_kernel[interp]
        ]()
        stream.enqueue_function(
            compiled,
            buffers.grad_rec,
            buffers.direction,
            buffers.shifts_3d,
            buffers.lines,
            buffers.grad_dir,
            buffers.grad_shift_3d,
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_insert_line_3d_pose_grad_kernel[interp]](
        buffers.grad_rec,
        buffers.direction,
        buffers.shifts_3d,
        buffers.lines,
        buffers.grad_dir,
        buffers.grad_shift_3d,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_insert_line_3d_weight_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: InsertLine3DWeightGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_insert_line_3d_weight_grad_kernel[interp]]()
        stream.enqueue_function(
            compiled,
            buffers.gwvol,
            buffers.direction,
            buffers.grad_weight,
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_insert_line_3d_weight_grad_kernel[interp]](
        buffers.gwvol,
        buffers.direction,
        buffers.grad_weight,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_extract_slice_3d_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: ExtractSlice3DPoseGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_extract_slice_3d_pose_grad_kernel[interp]]()
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
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_extract_slice_3d_pose_grad_kernel[interp]](
        buffers.rec,
        buffers.rot,
        buffers.shifts_2d,
        buffers.shifts_3d,
        buffers.grad_proj,
        buffers.grad_rot,
        buffers.grad_shift,
        buffers.grad_shift_3d,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_insert_slice_3d_pose_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: InsertSlice3DPoseGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _insert_slice_3d_pose_grad_kernel[interp]
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
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_insert_slice_3d_pose_grad_kernel[interp]](
        buffers.grad_rec,
        buffers.rot,
        buffers.shifts_2d,
        buffers.shifts_3d,
        buffers.proj,
        buffers.grad_rot,
        buffers.grad_shift,
        buffers.grad_shift_3d,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_insert_slice_3d_weight_grad[
    interp: Int
](
    ctx: DeviceContext,
    buffers: InsertSlice3DWeightGradBuffers,
    total: Int,
    p: FourierSliceParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(total)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_insert_slice_3d_weight_grad_kernel[interp]]()
        stream.enqueue_function(
            compiled,
            buffers.gwvol,
            buffers.rot,
            buffers.grad_weight,
            dp,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_insert_slice_3d_weight_grad_kernel[interp]](
        buffers.gwvol,
        buffers.rot,
        buffers.grad_weight,
        dp,
        grid_dim=ceildiv(total, BLOCK),
        block_dim=BLOCK,
    )
