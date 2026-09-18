"""GPU kernels (one thread per sample) and their launchers.

The per-sample math is `_interp.mojo`, shared with the CPU module unchanged.
`InterpParams` is not `DevicePassable` (`Int` has no fixed width), so each
launcher packs it into a `DeviceParams` and the kernel unpacks it on entry.
Kernels read and write torch device memory in place: the Python caller passes
raw device addresses (see `_mojo_backend/_gpu.py`), so nothing is staged.

Every launcher enqueues on the `DeviceStream` it is handed -- torch's own CUDA
stream (wrapped once per `DeviceSession`, see `image_interpolation_gpu.mojo`)
so ordering with the surrounding torch ops is the stream's job, or the
`DeviceContext`'s own stream on Metal, where the entry point synchronises the
context before returning.

The backward kernels are specialised at compile time on which gradients they
produce: the coordinate gradient needs `ndim` weight derivatives per tap and
roughly triples the register footprint, and a kernel's register allocation
covers every path in its body, so a runtime branch would cost occupancy in
both. The voxel width (1 real / 2 complex, `_interp.mojo`'s `W`) is NOT
specialised, deliberately: kernels built for a fixed `W` came out with the
same registers and occupancy but a lower L1 hit rate and ran 15-35% slower on
every multi-tap gather (ncu, RTX 6000 Ada: 3D trilinear complex gather 60 vs
52 us, its insert-backward twin 60 vs 44 us, bicubic 113 vs 103 us) -- ptxas
schedules the tap loads differently for the smaller body. The uniform runtime
branch on `p.inner` is free, so the cores take it.
"""

from std.gpu import global_idx
from std.math import ceildiv

from max.gpu.host import DeviceContext, DeviceStream

from _common import BLOCK, DeviceParams, F32Ptr, InterpParams
from _interp import (
    _insert_backward_impl,
    _insert_one,
    _sample_backward_impl,
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
    ndim: Int, interp: Int, grad_image: Bool, grad_coords: Bool
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
    _sample_backward_impl[ndim, interp, True, grad_image, grad_coords](
        img, coords, gout, gimg, gcoords, s, p
    )


def _insert_forward_kernel[
    ndim: Int, interp: Int
](values: F32Ptr, coords: F32Ptr, img: F32Ptr, wimg: F32Ptr, dp: DeviceParams):
    var s = global_idx.x
    if s >= Int(dp.total):
        return
    var p = dp.to_params(ndim, interp)
    _insert_one[ndim, interp](values, coords, img, wimg, s, p)


def _insert_backward_kernel[
    ndim: Int, interp: Int, grad_values: Bool, grad_coords: Bool
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
    _insert_backward_impl[ndim, interp, grad_values, grad_coords](
        values, coords, gimg, gwimg, gvalues, gcoords, s, p
    )


# ---------------------------------------------------------------------------
# Launchers: pick the specialisation from the runtime params, then enqueue
# ---------------------------------------------------------------------------


@always_inline
def _launch_fill_zero(
    ctx: DeviceContext,
    stream: DeviceStream,
    dst: F32Ptr,
    count: Int,
    p: InterpParams,
) raises:
    """Zero `count` floats at `dst`, ordered before later launches on `stream`.
    """
    var dp = p.to_device(count)
    var compiled = ctx.compile_function[_fill_zero_kernel]()
    stream.enqueue_function(
        compiled, dst, dp, grid_dim=ceildiv(count, BLOCK), block_dim=BLOCK
    )


@always_inline
def _launch_sample_forward[
    ndim: Int, interp: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    img: F32Ptr,
    coords: F32Ptr,
    dst: F32Ptr,
    p: InterpParams,
) raises:
    var dp = p.to_device(p.n)
    var compiled = ctx.compile_function[_sample_forward_kernel[ndim, interp]]()
    stream.enqueue_function(
        compiled,
        img,
        coords,
        dst,
        dp,
        grid_dim=ceildiv(p.n, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _enqueue_sample_backward[
    ndim: Int, interp: Int, grad_image: Bool, grad_coords: Bool
](
    ctx: DeviceContext,
    stream: DeviceStream,
    img: F32Ptr,
    coords: F32Ptr,
    gout: F32Ptr,
    gimg: F32Ptr,
    gcoords: F32Ptr,
    dp: DeviceParams,
) raises:
    var compiled = ctx.compile_function[
        _sample_backward_kernel[ndim, interp, grad_image, grad_coords]
    ]()
    stream.enqueue_function(
        compiled,
        img,
        coords,
        gout,
        gimg,
        gcoords,
        dp,
        grid_dim=ceildiv(Int(dp.total), BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_sample_backward[
    ndim: Int, interp: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    img: F32Ptr,
    coords: F32Ptr,
    gout: F32Ptr,
    gimg: F32Ptr,
    gcoords: F32Ptr,
    p: InterpParams,
) raises:
    var dp = p.to_device(p.n)
    if p.need_grad_coords != 0:
        if p.need_grad_image != 0:
            _enqueue_sample_backward[ndim, interp, True, True](
                ctx, stream, img, coords, gout, gimg, gcoords, dp
            )
        else:
            _enqueue_sample_backward[ndim, interp, False, True](
                ctx, stream, img, coords, gout, gimg, gcoords, dp
            )
    else:
        _enqueue_sample_backward[ndim, interp, True, False](
            ctx, stream, img, coords, gout, gimg, gcoords, dp
        )


@always_inline
def _launch_insert_forward[
    ndim: Int, interp: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    values: F32Ptr,
    coords: F32Ptr,
    img: F32Ptr,
    wimg: F32Ptr,
    p: InterpParams,
) raises:
    var dp = p.to_device(p.n)
    var compiled = ctx.compile_function[_insert_forward_kernel[ndim, interp]]()
    stream.enqueue_function(
        compiled,
        values,
        coords,
        img,
        wimg,
        dp,
        grid_dim=ceildiv(p.n, BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _enqueue_insert_backward[
    ndim: Int, interp: Int, grad_values: Bool, grad_coords: Bool
](
    ctx: DeviceContext,
    stream: DeviceStream,
    values: F32Ptr,
    coords: F32Ptr,
    gimg: F32Ptr,
    gwimg: F32Ptr,
    gvalues: F32Ptr,
    gcoords: F32Ptr,
    dp: DeviceParams,
) raises:
    var compiled = ctx.compile_function[
        _insert_backward_kernel[ndim, interp, grad_values, grad_coords]
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
        grid_dim=ceildiv(Int(dp.total), BLOCK),
        block_dim=BLOCK,
    )


@always_inline
def _launch_insert_backward[
    ndim: Int, interp: Int
](
    ctx: DeviceContext,
    stream: DeviceStream,
    values: F32Ptr,
    coords: F32Ptr,
    gimg: F32Ptr,
    gwimg: F32Ptr,
    gvalues: F32Ptr,
    gcoords: F32Ptr,
    p: InterpParams,
) raises:
    var dp = p.to_device(p.n)
    if p.need_grad_coords != 0:
        if p.need_grad_values != 0:
            _enqueue_insert_backward[ndim, interp, True, True](
                ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, dp
            )
        else:
            _enqueue_insert_backward[ndim, interp, False, True](
                ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, dp
            )
    else:
        _enqueue_insert_backward[ndim, interp, True, False](
            ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, dp
        )
