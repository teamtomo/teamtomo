"""GPU entry points of the Mojo image-interpolation kernels (Python module).

A separate extension module from the CPU one so that a missing / broken GPU
compiler (e.g. no Metal Toolchain) leaves the CPU kernels usable. The kernels
themselves are in `_device.mojo`; the per-sample math in `_interp.mojo` is
shared with the CPU module.

Every entry point takes `(session, bufs, params, addrs)`: `session` is the
process-wide `DeviceSession`, `bufs` the tuple of torch device tensors (used
only for their identity here -- the kernels read/write their memory in place),
`params` a `KernelParams` NamedTuple, and `addrs` the raw device address of each
buffer in `bufs` order followed by torch's stream address (0 on Metal). Output
buffers are allocated (and, where accumulated into, zeroed) by the caller.

A non-zero stream address is torch's current CUDA stream: kernels are enqueued
on it, so they are ordered with the surrounding torch ops without any device
synchronisation. The `DeviceStream` wrapping it is created once and cached on
the session -- wrapping it per call was measured (nsys) to issue a
`cuStreamSynchronize` per launch when the wrapper is torn down, which
serialises the host with the GPU and leaves the device idle between kernels.
A zero address (Metal) runs on the context's own stream and the entry point
synchronises before returning.
"""

from std.os import abort
from std.memory import OpaquePointer
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext, DeviceStream

from _common import (
    CUBIC,
    LINEAR,
    NEAREST,
    F32Ptr,
    InterpParams,
    _dptr,
    _read_params,
)
from _device import (
    _launch_fill_zero,
    _launch_insert_backward,
    _launch_insert_forward,
    _launch_sample_backward,
    _launch_sample_forward,
)


struct DeviceSession(Movable, Writable):
    """Process-wide GPU context holder: one `DeviceContext` for all calls.

    Constructing a fresh `DeviceContext` per call leaks the underlying Metal
    command queue and crashes long loops; Python caches one session instead.
    Also caches the `DeviceStream` wrapping torch's current stream (see the
    module docstring for why that matters).
    """

    var ctx: DeviceContext
    var stream_addr: Int
    var stream: Optional[DeviceStream]

    def write_to(self, mut writer: Some[Writer]):
        writer.write("DeviceSession()")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("DeviceSession()")

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.stream_addr = 0
        self.stream = None

    @staticmethod
    def py_init(
        out self: DeviceSession, args: PythonObject, kwargs: PythonObject
    ) raises:
        self = DeviceSession()

    def ensure_stream(mut self, stream_addr: Int) raises:
        """Make `self.stream` wrap the (non-zero) external stream `stream_addr`.
        """
        if self.stream and self.stream_addr == stream_addr:
            return
        self.stream = self.ctx.create_external_stream(
            OpaquePointer[MutAnyOrigin](unsafe_from_address=stream_addr)
        )
        self.stream_addr = stream_addr


# ---------------------------------------------------------------------------
# (ndim, interp) -> kernel specialisation
# ---------------------------------------------------------------------------


def _dispatch_sample_forward(
    ctx: DeviceContext,
    stream: DeviceStream,
    p: InterpParams,
    img: F32Ptr,
    coords: F32Ptr,
    dst: F32Ptr,
) raises:
    if p.ndim == 3:
        if p.interp == NEAREST:
            _launch_sample_forward[3, NEAREST](ctx, stream, img, coords, dst, p)
        elif p.interp == LINEAR:
            _launch_sample_forward[3, LINEAR](ctx, stream, img, coords, dst, p)
        else:
            _launch_sample_forward[3, CUBIC](ctx, stream, img, coords, dst, p)
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _launch_sample_forward[2, NEAREST](ctx, stream, img, coords, dst, p)
        elif p.interp == LINEAR:
            _launch_sample_forward[2, LINEAR](ctx, stream, img, coords, dst, p)
        else:
            _launch_sample_forward[2, CUBIC](ctx, stream, img, coords, dst, p)
    else:
        if p.interp == NEAREST:
            _launch_sample_forward[1, NEAREST](ctx, stream, img, coords, dst, p)
        elif p.interp == LINEAR:
            _launch_sample_forward[1, LINEAR](ctx, stream, img, coords, dst, p)
        else:
            _launch_sample_forward[1, CUBIC](ctx, stream, img, coords, dst, p)


def _dispatch_sample_backward(
    ctx: DeviceContext,
    stream: DeviceStream,
    p: InterpParams,
    img: F32Ptr,
    coords: F32Ptr,
    gout: F32Ptr,
    gimg: F32Ptr,
    gcoords: F32Ptr,
) raises:
    if p.need_grad_image != 0 and p.zero_grad_image != 0:
        _launch_fill_zero(
            ctx, stream, gimg, p.c * p.spatial_size() * p.inner, p
        )
    if p.ndim == 3:
        if p.interp == NEAREST:
            _launch_sample_backward[3, NEAREST](
                ctx, stream, img, coords, gout, gimg, gcoords, p
            )
        elif p.interp == LINEAR:
            _launch_sample_backward[3, LINEAR](
                ctx, stream, img, coords, gout, gimg, gcoords, p
            )
        else:
            _launch_sample_backward[3, CUBIC](
                ctx, stream, img, coords, gout, gimg, gcoords, p
            )
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _launch_sample_backward[2, NEAREST](
                ctx, stream, img, coords, gout, gimg, gcoords, p
            )
        elif p.interp == LINEAR:
            _launch_sample_backward[2, LINEAR](
                ctx, stream, img, coords, gout, gimg, gcoords, p
            )
        else:
            _launch_sample_backward[2, CUBIC](
                ctx, stream, img, coords, gout, gimg, gcoords, p
            )
    else:
        if p.interp == NEAREST:
            _launch_sample_backward[1, NEAREST](
                ctx, stream, img, coords, gout, gimg, gcoords, p
            )
        elif p.interp == LINEAR:
            _launch_sample_backward[1, LINEAR](
                ctx, stream, img, coords, gout, gimg, gcoords, p
            )
        else:
            _launch_sample_backward[1, CUBIC](
                ctx, stream, img, coords, gout, gimg, gcoords, p
            )


def _dispatch_insert_forward(
    ctx: DeviceContext,
    stream: DeviceStream,
    p: InterpParams,
    values: F32Ptr,
    coords: F32Ptr,
    img: F32Ptr,
    wimg: F32Ptr,
) raises:
    if p.ndim == 3:
        if p.interp == NEAREST:
            _launch_insert_forward[3, NEAREST](
                ctx, stream, values, coords, img, wimg, p
            )
        elif p.interp == LINEAR:
            _launch_insert_forward[3, LINEAR](
                ctx, stream, values, coords, img, wimg, p
            )
        else:
            _launch_insert_forward[3, CUBIC](
                ctx, stream, values, coords, img, wimg, p
            )
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _launch_insert_forward[2, NEAREST](
                ctx, stream, values, coords, img, wimg, p
            )
        elif p.interp == LINEAR:
            _launch_insert_forward[2, LINEAR](
                ctx, stream, values, coords, img, wimg, p
            )
        else:
            _launch_insert_forward[2, CUBIC](
                ctx, stream, values, coords, img, wimg, p
            )
    else:
        if p.interp == NEAREST:
            _launch_insert_forward[1, NEAREST](
                ctx, stream, values, coords, img, wimg, p
            )
        elif p.interp == LINEAR:
            _launch_insert_forward[1, LINEAR](
                ctx, stream, values, coords, img, wimg, p
            )
        else:
            _launch_insert_forward[1, CUBIC](
                ctx, stream, values, coords, img, wimg, p
            )


def _dispatch_insert_backward(
    ctx: DeviceContext,
    stream: DeviceStream,
    p: InterpParams,
    values: F32Ptr,
    coords: F32Ptr,
    gimg: F32Ptr,
    gwimg: F32Ptr,
    gvalues: F32Ptr,
    gcoords: F32Ptr,
) raises:
    if p.ndim == 3:
        if p.interp == NEAREST:
            _launch_insert_backward[3, NEAREST](
                ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, p
            )
        elif p.interp == LINEAR:
            _launch_insert_backward[3, LINEAR](
                ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, p
            )
        else:
            _launch_insert_backward[3, CUBIC](
                ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, p
            )
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _launch_insert_backward[2, NEAREST](
                ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, p
            )
        elif p.interp == LINEAR:
            _launch_insert_backward[2, LINEAR](
                ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, p
            )
        else:
            _launch_insert_backward[2, CUBIC](
                ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, p
            )
    else:
        if p.interp == NEAREST:
            _launch_insert_backward[1, NEAREST](
                ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, p
            )
        elif p.interp == LINEAR:
            _launch_insert_backward[1, LINEAR](
                ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, p
            )
        else:
            _launch_insert_backward[1, CUBIC](
                ctx, stream, values, coords, gimg, gwimg, gvalues, gcoords, p
            )


# ---------------------------------------------------------------------------
# Python entry points
# ---------------------------------------------------------------------------


def sample_forward_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """GPU `sample_forward`; bufs = (image, coords, samples)."""
    var p = _read_params(params_obj)
    if p.n == 0:
        return PythonObject(0)
    var sess = session_obj.downcast_value_ptr[DeviceSession]()
    var ctx = sess[].ctx
    var img = _dptr(addrs_obj[0])
    var coords = _dptr(addrs_obj[1])
    var dst = _dptr(addrs_obj[2])
    var sa = Int(py=addrs_obj[3])
    if sa != 0:
        sess[].ensure_stream(sa)
        _dispatch_sample_forward(
            ctx, sess[].stream.value(), p, img, coords, dst
        )
    else:
        _dispatch_sample_forward(ctx, ctx.stream(), p, img, coords, dst)
        ctx.synchronize()
    return PythonObject(0)


def sample_backward_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """GPU `sample_backward`; bufs = (image, coords, grad_samples, grad_image, grad_coords).
    """
    var p = _read_params(params_obj)
    if p.n == 0:
        return PythonObject(0)
    var sess = session_obj.downcast_value_ptr[DeviceSession]()
    var ctx = sess[].ctx
    var img = _dptr(addrs_obj[0])
    var coords = _dptr(addrs_obj[1])
    var gout = _dptr(addrs_obj[2])
    var gimg = _dptr(addrs_obj[3])
    var gcoords = _dptr(addrs_obj[4])
    var sa = Int(py=addrs_obj[5])
    if sa != 0:
        sess[].ensure_stream(sa)
        _dispatch_sample_backward(
            ctx, sess[].stream.value(), p, img, coords, gout, gimg, gcoords
        )
    else:
        _dispatch_sample_backward(
            ctx, ctx.stream(), p, img, coords, gout, gimg, gcoords
        )
        ctx.synchronize()
    return PythonObject(0)


def insert_forward_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """GPU `insert_forward`; bufs = (values, coords, image, weights)."""
    var p = _read_params(params_obj)
    if p.n == 0:
        return PythonObject(0)
    var sess = session_obj.downcast_value_ptr[DeviceSession]()
    var ctx = sess[].ctx
    var values = _dptr(addrs_obj[0])
    var coords = _dptr(addrs_obj[1])
    var img = _dptr(addrs_obj[2])
    var wimg = _dptr(addrs_obj[3])
    var sa = Int(py=addrs_obj[4])
    if sa != 0:
        sess[].ensure_stream(sa)
        _dispatch_insert_forward(
            ctx, sess[].stream.value(), p, values, coords, img, wimg
        )
    else:
        _dispatch_insert_forward(
            ctx, ctx.stream(), p, values, coords, img, wimg
        )
        ctx.synchronize()
    return PythonObject(0)


def insert_backward_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """GPU `insert_backward`.

    bufs = (values, coords, grad_image, grad_weights, grad_values, grad_coords).
    """
    var p = _read_params(params_obj)
    if p.n == 0:
        return PythonObject(0)
    var sess = session_obj.downcast_value_ptr[DeviceSession]()
    var ctx = sess[].ctx
    var values = _dptr(addrs_obj[0])
    var coords = _dptr(addrs_obj[1])
    var gimg = _dptr(addrs_obj[2])
    var gwimg = _dptr(addrs_obj[3])
    var gvalues = _dptr(addrs_obj[4])
    var gcoords = _dptr(addrs_obj[5])
    var sa = Int(py=addrs_obj[6])
    if sa != 0:
        sess[].ensure_stream(sa)
        _dispatch_insert_backward(
            ctx,
            sess[].stream.value(),
            p,
            values,
            coords,
            gimg,
            gwimg,
            gvalues,
            gcoords,
        )
    else:
        _dispatch_insert_backward(
            ctx, ctx.stream(), p, values, coords, gimg, gwimg, gvalues, gcoords
        )
        ctx.synchronize()
    return PythonObject(0)


@export
def PyInit_image_interpolation_gpu() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("image_interpolation_gpu")
        _ = m.add_type[DeviceSession]("DeviceSession").def_py_init[
            DeviceSession.py_init
        ]()
        m.def_function[sample_forward_gpu](
            "sample_forward_gpu",
            docstring="Interpolate an image at coordinates (GPU).",
        )
        m.def_function[sample_backward_gpu](
            "sample_backward_gpu", docstring="Adjoint of sample_forward (GPU)."
        )
        m.def_function[insert_forward_gpu](
            "insert_forward_gpu", docstring="Splat values into an image (GPU)."
        )
        m.def_function[insert_backward_gpu](
            "insert_backward_gpu", docstring="Adjoint of insert_forward (GPU)."
        )
        return m.finalize()
    except e:
        abort(String("failed to create image_interpolation_gpu module: ", e))
