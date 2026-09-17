"""GPU kernels (one thread per sample) and their launchers.

The per-sample math is `_interp.mojo`, shared with the CPU module unchanged.
`InterpParams` is not `DevicePassable` (`Int` has no fixed width), so each
launcher packs it into a `DeviceParams` and the kernel unpacks it on entry.
Kernels read and write torch device memory in place: the Python caller passes
raw device addresses (see `_mojo_backend/_gpu.py`), so nothing is staged.

`stream_addr` selects the stream a kernel is enqueued on: a non-zero value is
torch's own CUDA stream (ordering with surrounding torch ops is then the
stream's job); zero means the `DeviceContext`'s own stream (Metal), and the
entry point synchronises the context before returning.
"""

from std.gpu import global_idx
from std.math import ceildiv
from std.memory import OpaquePointer

from max.gpu.host import DeviceContext

from _common import BLOCK, DeviceParams, F32Ptr, InterpParams
from _interp import (
    _insert_backward_one,
    _insert_one,
    _sample_backward_one,
    _sample_one,
)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _fill_zero_kernel(dst: F32Ptr, dp: DeviceParams):
    """dst[0 : dp.total] = 0 -- lets a scatter kernel start from an uninitialised
    buffer without a separate torch fill (and its queue round trip)."""
    var i = global_idx.x
    if i < Int(dp.total):
        dst[unsafe_offset=i] = 0.0


def _sample_forward_kernel[
    ndim: Int, interp: Int
](img: F32Ptr, coords: F32Ptr, dst: F32Ptr, dp: DeviceParams):
    var s = global_idx.x
    if s >= Int(dp.total):
        return
    var p = dp.to_params(ndim, interp)
    _sample_one[ndim, interp](img, coords, dst, s, p)


def _sample_backward_kernel[
    ndim: Int, interp: Int
](
    img: F32Ptr,
    coords: F32Ptr,
    gout: F32Ptr,
    gimg: F32Ptr,
    gcoords: F32Ptr,
    dp: DeviceParams,
):
    var s = global_idx.x
    if s >= Int(dp.total):
        return
    var p = dp.to_params(ndim, interp)
    _sample_backward_one[ndim, interp](img, coords, gout, gimg, gcoords, s, p)


def _insert_forward_kernel[
    ndim: Int, interp: Int
](values: F32Ptr, coords: F32Ptr, img: F32Ptr, wimg: F32Ptr, dp: DeviceParams):
    var s = global_idx.x
    if s >= Int(dp.total):
        return
    var p = dp.to_params(ndim, interp)
    _insert_one[ndim, interp](values, coords, img, wimg, s, p)


def _insert_backward_kernel[
    ndim: Int, interp: Int
](
    values: F32Ptr,
    coords: F32Ptr,
    gimg: F32Ptr,
    gwimg: F32Ptr,
    gvalues: F32Ptr,
    gcoords: F32Ptr,
    dp: DeviceParams,
):
    var s = global_idx.x
    if s >= Int(dp.total):
        return
    var p = dp.to_params(ndim, interp)
    _insert_backward_one[ndim, interp](
        values, coords, gimg, gwimg, gvalues, gcoords, s, p
    )


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


@always_inline
def _launch_fill_zero(
    ctx: DeviceContext, dst: F32Ptr, count: Int, p: InterpParams, stream_addr: Int
) raises:
    """Zero `count` floats at `dst`, ordered before later launches on the same stream."""
    var dp = p.to_device(count)
    var grid = ceildiv(count, BLOCK)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_fill_zero_kernel]()
        stream.enqueue_function(compiled, dst, dp, grid_dim=grid, block_dim=BLOCK)
        return
    ctx.enqueue_function[_fill_zero_kernel](dst, dp, grid_dim=grid, block_dim=BLOCK)


@always_inline
def _launch_sample_forward[
    ndim: Int, interp: Int
](
    ctx: DeviceContext,
    img: F32Ptr,
    coords: F32Ptr,
    dst: F32Ptr,
    p: InterpParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(p.n)
    var grid = ceildiv(p.n, BLOCK)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_sample_forward_kernel[ndim, interp]]()
        stream.enqueue_function(
            compiled, img, coords, dst, dp, grid_dim=grid, block_dim=BLOCK
        )
        return
    ctx.enqueue_function[_sample_forward_kernel[ndim, interp]](
        img, coords, dst, dp, grid_dim=grid, block_dim=BLOCK
    )


@always_inline
def _launch_sample_backward[
    ndim: Int, interp: Int
](
    ctx: DeviceContext,
    img: F32Ptr,
    coords: F32Ptr,
    gout: F32Ptr,
    gimg: F32Ptr,
    gcoords: F32Ptr,
    p: InterpParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(p.n)
    var grid = ceildiv(p.n, BLOCK)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _sample_backward_kernel[ndim, interp]
        ]()
        stream.enqueue_function(
            compiled,
            img,
            coords,
            gout,
            gimg,
            gcoords,
            dp,
            grid_dim=grid,
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_sample_backward_kernel[ndim, interp]](
        img, coords, gout, gimg, gcoords, dp, grid_dim=grid, block_dim=BLOCK
    )


@always_inline
def _launch_insert_forward[
    ndim: Int, interp: Int
](
    ctx: DeviceContext,
    values: F32Ptr,
    coords: F32Ptr,
    img: F32Ptr,
    wimg: F32Ptr,
    p: InterpParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(p.n)
    var grid = ceildiv(p.n, BLOCK)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[_insert_forward_kernel[ndim, interp]]()
        stream.enqueue_function(
            compiled, values, coords, img, wimg, dp, grid_dim=grid, block_dim=BLOCK
        )
        return
    ctx.enqueue_function[_insert_forward_kernel[ndim, interp]](
        values, coords, img, wimg, dp, grid_dim=grid, block_dim=BLOCK
    )


@always_inline
def _launch_insert_backward[
    ndim: Int, interp: Int
](
    ctx: DeviceContext,
    values: F32Ptr,
    coords: F32Ptr,
    gimg: F32Ptr,
    gwimg: F32Ptr,
    gvalues: F32Ptr,
    gcoords: F32Ptr,
    p: InterpParams,
    stream_addr: Int,
) raises:
    var dp = p.to_device(p.n)
    var grid = ceildiv(p.n, BLOCK)
    if stream_addr != 0:
        var stream = ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        var compiled = ctx.compile_function[
            _insert_backward_kernel[ndim, interp]
        ]()
        stream.enqueue_function(
            compiled,
            values,
            coords,
            gimg,
            gwimg,
            gvalues,
            gcoords,
            dp,
            grid_dim=grid,
            block_dim=BLOCK,
        )
        return
    ctx.enqueue_function[_insert_backward_kernel[ndim, interp]](
        values,
        coords,
        gimg,
        gwimg,
        gvalues,
        gcoords,
        dp,
        grid_dim=grid,
        block_dim=BLOCK,
    )
