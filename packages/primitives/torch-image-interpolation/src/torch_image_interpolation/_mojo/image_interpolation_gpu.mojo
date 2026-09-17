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
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from max.gpu.host import DeviceContext

from _common import CUBIC, LINEAR, NEAREST, _dptr, _read_params
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
    """

    var ctx: DeviceContext

    def write_to(self, mut writer: Some[Writer]):
        writer.write("DeviceSession()")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("DeviceSession()")

    def __init__(out self) raises:
        self.ctx = DeviceContext()

    @staticmethod
    def py_init(
        out self: DeviceSession, args: PythonObject, kwargs: PythonObject
    ) raises:
        self = DeviceSession()


@always_inline
def _session_ctx(session_obj: PythonObject) raises -> DeviceContext:
    return session_obj.downcast_value_ptr[DeviceSession]()[].ctx


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
    var ctx = _session_ctx(session_obj)
    var img = _dptr(addrs_obj[0])
    var coords = _dptr(addrs_obj[1])
    var dst = _dptr(addrs_obj[2])
    var sa = Int(py=addrs_obj[3])
    if p.ndim == 3:
        if p.interp == NEAREST:
            _launch_sample_forward[3, NEAREST](ctx, img, coords, dst, p, sa)
        elif p.interp == LINEAR:
            _launch_sample_forward[3, LINEAR](ctx, img, coords, dst, p, sa)
        else:
            _launch_sample_forward[3, CUBIC](ctx, img, coords, dst, p, sa)
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _launch_sample_forward[2, NEAREST](ctx, img, coords, dst, p, sa)
        elif p.interp == LINEAR:
            _launch_sample_forward[2, LINEAR](ctx, img, coords, dst, p, sa)
        else:
            _launch_sample_forward[2, CUBIC](ctx, img, coords, dst, p, sa)
    else:
        if p.interp == NEAREST:
            _launch_sample_forward[1, NEAREST](ctx, img, coords, dst, p, sa)
        elif p.interp == LINEAR:
            _launch_sample_forward[1, LINEAR](ctx, img, coords, dst, p, sa)
        else:
            _launch_sample_forward[1, CUBIC](ctx, img, coords, dst, p, sa)
    if sa == 0:
        ctx.synchronize()
    return PythonObject(0)


def sample_backward_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """GPU `sample_backward`; bufs = (image, coords, grad_samples, grad_image, grad_coords)."""
    var p = _read_params(params_obj)
    if p.n == 0:
        return PythonObject(0)
    var ctx = _session_ctx(session_obj)
    var img = _dptr(addrs_obj[0])
    var coords = _dptr(addrs_obj[1])
    var gout = _dptr(addrs_obj[2])
    var gimg = _dptr(addrs_obj[3])
    var gcoords = _dptr(addrs_obj[4])
    var sa = Int(py=addrs_obj[5])
    if p.need_grad_image != 0 and p.zero_grad_image != 0:
        _launch_fill_zero(ctx, gimg, p.c * p.spatial_size() * p.inner, p, sa)
    if p.ndim == 3:
        if p.interp == NEAREST:
            _launch_sample_backward[3, NEAREST](ctx, img, coords, gout, gimg, gcoords, p, sa)
        elif p.interp == LINEAR:
            _launch_sample_backward[3, LINEAR](ctx, img, coords, gout, gimg, gcoords, p, sa)
        else:
            _launch_sample_backward[3, CUBIC](ctx, img, coords, gout, gimg, gcoords, p, sa)
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _launch_sample_backward[2, NEAREST](ctx, img, coords, gout, gimg, gcoords, p, sa)
        elif p.interp == LINEAR:
            _launch_sample_backward[2, LINEAR](ctx, img, coords, gout, gimg, gcoords, p, sa)
        else:
            _launch_sample_backward[2, CUBIC](ctx, img, coords, gout, gimg, gcoords, p, sa)
    else:
        if p.interp == NEAREST:
            _launch_sample_backward[1, NEAREST](ctx, img, coords, gout, gimg, gcoords, p, sa)
        elif p.interp == LINEAR:
            _launch_sample_backward[1, LINEAR](ctx, img, coords, gout, gimg, gcoords, p, sa)
        else:
            _launch_sample_backward[1, CUBIC](ctx, img, coords, gout, gimg, gcoords, p, sa)
    if sa == 0:
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
    var ctx = _session_ctx(session_obj)
    var values = _dptr(addrs_obj[0])
    var coords = _dptr(addrs_obj[1])
    var img = _dptr(addrs_obj[2])
    var wimg = _dptr(addrs_obj[3])
    var sa = Int(py=addrs_obj[4])
    if p.ndim == 3:
        if p.interp == NEAREST:
            _launch_insert_forward[3, NEAREST](ctx, values, coords, img, wimg, p, sa)
        elif p.interp == LINEAR:
            _launch_insert_forward[3, LINEAR](ctx, values, coords, img, wimg, p, sa)
        else:
            _launch_insert_forward[3, CUBIC](ctx, values, coords, img, wimg, p, sa)
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _launch_insert_forward[2, NEAREST](ctx, values, coords, img, wimg, p, sa)
        elif p.interp == LINEAR:
            _launch_insert_forward[2, LINEAR](ctx, values, coords, img, wimg, p, sa)
        else:
            _launch_insert_forward[2, CUBIC](ctx, values, coords, img, wimg, p, sa)
    else:
        if p.interp == NEAREST:
            _launch_insert_forward[1, NEAREST](ctx, values, coords, img, wimg, p, sa)
        elif p.interp == LINEAR:
            _launch_insert_forward[1, LINEAR](ctx, values, coords, img, wimg, p, sa)
        else:
            _launch_insert_forward[1, CUBIC](ctx, values, coords, img, wimg, p, sa)
    if sa == 0:
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
    var ctx = _session_ctx(session_obj)
    var values = _dptr(addrs_obj[0])
    var coords = _dptr(addrs_obj[1])
    var gimg = _dptr(addrs_obj[2])
    var gwimg = _dptr(addrs_obj[3])
    var gvalues = _dptr(addrs_obj[4])
    var gcoords = _dptr(addrs_obj[5])
    var sa = Int(py=addrs_obj[6])
    if p.ndim == 3:
        if p.interp == NEAREST:
            _launch_insert_backward[3, NEAREST](ctx, values, coords, gimg, gwimg, gvalues, gcoords, p, sa)
        elif p.interp == LINEAR:
            _launch_insert_backward[3, LINEAR](ctx, values, coords, gimg, gwimg, gvalues, gcoords, p, sa)
        else:
            _launch_insert_backward[3, CUBIC](ctx, values, coords, gimg, gwimg, gvalues, gcoords, p, sa)
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _launch_insert_backward[2, NEAREST](ctx, values, coords, gimg, gwimg, gvalues, gcoords, p, sa)
        elif p.interp == LINEAR:
            _launch_insert_backward[2, LINEAR](ctx, values, coords, gimg, gwimg, gvalues, gcoords, p, sa)
        else:
            _launch_insert_backward[2, CUBIC](ctx, values, coords, gimg, gwimg, gvalues, gcoords, p, sa)
    else:
        if p.interp == NEAREST:
            _launch_insert_backward[1, NEAREST](ctx, values, coords, gimg, gwimg, gvalues, gcoords, p, sa)
        elif p.interp == LINEAR:
            _launch_insert_backward[1, LINEAR](ctx, values, coords, gimg, gwimg, gvalues, gcoords, p, sa)
        else:
            _launch_insert_backward[1, CUBIC](ctx, values, coords, gimg, gwimg, gvalues, gcoords, p, sa)
    if sa == 0:
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
            "sample_forward_gpu", docstring="Interpolate an image at coordinates (GPU)."
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
