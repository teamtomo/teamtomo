"""Mojo CPU + GPU kernels for Fourier-space projection and backprojection.

This is the Python entry module of a single extension module. The
Fourier-space math is written **once** and shared
by two execution strategies; the implementation is split into grouped files:

    _common.mojo    types, constants, FourierSliceParams, small geometry/complex helpers
    _gather.mojo    sample + interpolate (forward projection)
    _scatter.mojo   atomic accumulate + splat (backprojection)
    _pixel.mojo     the per-pixel ops shared by CPU loops and GPU threads
    _device.mojo    GPU kernels + launchers (and legacy host<->device transfer)
    projectors.mojo (this file) the Python-facing entry points

- **CPU**: `parallelize` over projections (one thread per pose), atomic scatter.
- **GPU**: one `DeviceContext` thread per rfft pixel, reading and writing the
  memory behind torch device tensors **in place** -- no host round-trip. The
  Python caller places every buffer (inputs and pre-zeroed outputs) on the
  compute device and passes their raw device addresses in the `addrs` tuple;
  each GPU entry point rebuilds device pointers via `_dptr` and synchronises
  its command queue before returning. (The CPU entry points still read host
  memory via `_ptr`.)

Data is exchanged through raw pointers into contiguous tensors viewed as real
(`torch.view_as_real`: complex64 -> trailing dim 2) -- on the host for the CPU
kernels, on the device for the GPU kernels. All shapes/scalars are read once
into a `FourierSliceParams` value so the hot loops never call back into Python.

Tensor layouts (C-contiguous, rfft with DC at the origin):
    volume      float32 [bv, d, h, w, 2]       (complex; h = sidelength, w = h//2+1)
    slices      float32 [bv, bp, h, w, 2]      (complex)
    weights     float32 [bv, bp, h, w]         (real, scatter only)
    rotations   float32 [bv_rot, bp, 3, 3]     (zyx convention)
    shifts_2d      float32 [bv_shift_2d, bp, 2]
"""

from std.algorithm import parallelize
from std.os import abort
from std.gpu.host import DeviceContext
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import num_physical_cores

from _common import FP, FourierSliceParams, _dptr, _ptr
from _device import (
    _launch_backproject_pose_grad,
    _launch_forward_pose_grad,
    _launch_project,
    _launch_scatter,
    _launch_weight_grad,
)
from _pixel import _project_pixel, _scatter_pixel
from _pose_grad import (
    _backproject_pose_grad_pixel,
    _forward_pose_grad_pixel,
    _weight_grad_pixel,
)


@export
def PyInit_projectors() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("projectors")
        m.def_function[extract_central_slices_rfft_3d](
            "extract_central_slices_rfft_3d",
            docstring="Forward 3D->2D projection (CPU).",
        )
        m.def_function[extract_central_slices_rfft_3d_gpu](
            "extract_central_slices_rfft_3d_gpu",
            docstring="Forward 3D->2D projection (GPU).",
        )
        m.def_function[insert_central_slices_rfft_3d](
            "insert_central_slices_rfft_3d",
            docstring="Scatter 2D slices into a 3D volume (CPU).",
        )
        m.def_function[insert_central_slices_rfft_3d_gpu](
            "insert_central_slices_rfft_3d_gpu",
            docstring="Scatter 2D slices into a 3D volume (GPU).",
        )
        _ = m.add_type[DeviceSession]("DeviceSession").def_py_init[
            DeviceSession.py_init
        ]()
        m.def_function[extract_central_slices_rfft_3d_pose_grad](
            "extract_central_slices_rfft_3d_pose_grad",
            docstring="Forward-projection rotation/shift gradients (CPU).",
        )
        m.def_function[extract_central_slices_rfft_3d_pose_grad_gpu](
            "extract_central_slices_rfft_3d_pose_grad_gpu",
            docstring="Forward-projection rotation/shift gradients (GPU).",
        )
        m.def_function[insert_central_slices_rfft_3d_pose_grad](
            "insert_central_slices_rfft_3d_pose_grad",
            docstring="Backprojection rotation/shift gradients (CPU).",
        )
        m.def_function[insert_central_slices_rfft_3d_pose_grad_gpu](
            "insert_central_slices_rfft_3d_pose_grad_gpu",
            docstring="Backprojection rotation/shift gradients (GPU).",
        )
        m.def_function[insert_central_slices_rfft_3d_weight_grad](
            "insert_central_slices_rfft_3d_weight_grad",
            docstring="Backprojection weight gradients (CPU).",
        )
        m.def_function[insert_central_slices_rfft_3d_weight_grad_gpu](
            "insert_central_slices_rfft_3d_weight_grad_gpu",
            docstring="Backprojection weight gradients (GPU).",
        )
        return m.finalize()
    except e:
        abort(String("failed to create projectors module: ", e))


# ===========================================================================
# CPU entry points (parallelize over poses)
# ===========================================================================


def extract_central_slices_rfft_3d(
    rec_obj: PythonObject,
    rot_obj: PythonObject,
    shifts_2d_obj: PythonObject,
    shifts_3d_obj: PythonObject,
    proj_obj: PythonObject,
    params_obj: PythonObject,
) raises -> PythonObject:
    """Forward project on the CPU; writes `proj_obj` (pre-zeroed by the caller)."""
    var rec = _ptr(rec_obj)
    var rot = _ptr(rot_obj)
    var shifts_2d = _ptr(shifts_2d_obj)
    var shifts_3d = _ptr(shifts_3d_obj)
    var proj = _ptr(proj_obj)
    var p = _forward_params(
        rec_obj, rot_obj, shifts_2d_obj, shifts_3d_obj, proj_obj, params_obj
    )
    var bv = Int(py=rec_obj.shape[0])

    @parameter
    def worker(vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for y in range(p.proj_sidelength):
            for x in range(p.proj_sidelength_half()):
                _project_pixel(
                    rec, rot, shifts_2d, shifts_3d, proj, i_bv, i_bp, y, x, p
                )

    parallelize[worker](bv * p.bp, num_physical_cores())
    return PythonObject(0)


def insert_central_slices_rfft_3d(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Scatter on the CPU; writes vol (and wvol if weighted), pre-zeroed by caller.

    bufs   : (slices, weights, rot, shifts_2d, vol, wvol) viewed as real.
    params_obj : (oversampling, radius_cutoff_sq, has_shifts_2d, has_weights,
              friedel_double, skip_redundant, interp)
    """
    var inp = _ptr(bufs[0])
    var weights = _ptr(bufs[1])
    var rot = _ptr(bufs[2])
    var shifts_2d = _ptr(bufs[3])
    var shifts_3d = _ptr(bufs[4])
    var vol = _ptr(bufs[5])
    var wvol = _ptr(bufs[6])
    var p = _scatter_params(bufs, params_obj)
    var bv = Int(py=bufs[0].shape[0])

    @parameter
    def worker(vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for y in range(p.proj_sidelength):
            for x in range(p.proj_sidelength_half()):
                _scatter_pixel(
                    inp,
                    weights,
                    rot,
                    shifts_2d,
                    shifts_3d,
                    vol,
                    wvol,
                    i_bv,
                    i_bp,
                    y,
                    x,
                    p,
                )

    parallelize[worker](bv * p.bp, num_physical_cores())
    return PythonObject(0)


def extract_central_slices_rfft_3d_pose_grad(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Forward-projection rotation/shift gradients on the CPU.

    bufs : (rec, rot, shifts_2d, grad_proj, grad_rot, grad_shift); grad_* pre-zeroed.
    """
    var rec = _ptr(bufs[0])
    var rot = _ptr(bufs[1])
    var shifts_2d = _ptr(bufs[2])
    var shifts_3d = _ptr(bufs[3])
    var grad_proj = _ptr(bufs[4])
    var grad_rot = _ptr(bufs[5])
    var grad_shift = _ptr(bufs[6])
    var grad_shift_3d = _ptr(bufs[7])
    var p = _pose_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])

    @parameter
    def worker(vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for y in range(p.proj_sidelength):
            for x in range(p.proj_sidelength_half()):
                _forward_pose_grad_pixel(
                    rec,
                    rot,
                    shifts_2d,
                    shifts_3d,
                    grad_proj,
                    grad_rot,
                    grad_shift,
                    grad_shift_3d,
                    i_bv,
                    i_bp,
                    y,
                    x,
                    p,
                )

    parallelize[worker](bv * p.bp, num_physical_cores())
    return PythonObject(0)


def insert_central_slices_rfft_3d_pose_grad(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Backprojection rotation/shift gradients on the CPU.

    bufs : (grad_rec, rot, shifts_2d, proj, grad_rot, grad_shift); grad_* pre-zeroed.
    """
    var grad_rec = _ptr(bufs[0])
    var rot = _ptr(bufs[1])
    var shifts_2d = _ptr(bufs[2])
    var shifts_3d = _ptr(bufs[3])
    var proj = _ptr(bufs[4])
    var grad_rot = _ptr(bufs[5])
    var grad_shift = _ptr(bufs[6])
    var grad_shift_3d = _ptr(bufs[7])
    var p = _pose_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])

    @parameter
    def worker(vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for y in range(p.proj_sidelength):
            for x in range(p.proj_sidelength_half()):
                _backproject_pose_grad_pixel(
                    grad_rec,
                    rot,
                    shifts_2d,
                    shifts_3d,
                    proj,
                    grad_rot,
                    grad_shift,
                    grad_shift_3d,
                    i_bv,
                    i_bp,
                    y,
                    x,
                    p,
                )

    parallelize[worker](bv * p.bp, num_physical_cores())
    return PythonObject(0)


def insert_central_slices_rfft_3d_weight_grad(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Backprojection weight gradients on the CPU.

    bufs : (grad_weight_vol, rot, shifts_2d, grad_weight); grad_weight pre-zeroed.
    """
    var gwvol = _ptr(bufs[0])
    var rot = _ptr(bufs[1])
    var grad_weight = _ptr(bufs[4])
    var p = _pose_grad_params(bufs, params_obj, 1)
    var bv = Int(py=bufs[0].shape[0])

    @parameter
    def worker(vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for y in range(p.proj_sidelength):
            for x in range(p.proj_sidelength_half()):
                _weight_grad_pixel(gwvol, rot, grad_weight, i_bv, i_bp, y, x, p)

    parallelize[worker](bv * p.bp, num_physical_cores())
    return PythonObject(0)


# ===========================================================================
# Shared GPU context
# ===========================================================================


struct DeviceSession(Movable, Writable):
    """Process-wide GPU context holder: one ``DeviceContext`` for all calls.

    Constructing a fresh ``DeviceContext`` per kernel call leaks the underlying
    Metal command queue; the queues accumulate until Metal's limit is hit and
    long training loops crash after ~1000 steps. Python holds one cached
    ``DeviceSession`` and passes it into every GPU entry point so a single
    command queue is reused for the whole process.
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
    """Borrow the shared ``DeviceContext`` from a Python ``DeviceSession``."""
    return session_obj.downcast_value_ptr[DeviceSession]()[].ctx


# ===========================================================================
# GPU entry points
# ===========================================================================


def extract_central_slices_rfft_3d_gpu(
    session_obj: PythonObject,
    rec_obj: PythonObject,
    rot_obj: PythonObject,
    shifts_2d_obj: PythonObject,
    shifts_3d_obj: PythonObject,
    proj_obj: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Forward project on the GPU, reading/writing torch device memory directly.

    `addrs_obj` carries the raw device virtual addresses of, in order,
    (rec, rot, shifts_2d, shifts_3d, proj). The tensor objects are used only for
    their shapes. `proj` is pre-zeroed on the device by the caller (radius-cut
    pixels stay 0, matching the CPU path).
    """
    var p = _forward_params(
        rec_obj, rot_obj, shifts_2d_obj, shifts_3d_obj, proj_obj, params_obj
    )
    var bv = Int(py=rec_obj.shape[0])
    var total = bv * p.bp * p.proj_sidelength * p.proj_sidelength_half()

    var ctx = _session_ctx(session_obj)
    _launch_project(
        ctx,
        _dptr(addrs_obj[0]),
        _dptr(addrs_obj[1]),
        _dptr(addrs_obj[2]),
        _dptr(addrs_obj[3]),
        _dptr(addrs_obj[4]),
        total,
        p,
    )
    ctx.synchronize()  # writes land before Python reads the output tensor
    return PythonObject(0)


def insert_central_slices_rfft_3d_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Scatter on the GPU, reading/writing torch device memory directly.

    `addrs_obj` holds device VAs for bufs = (slices, weights, rot, shifts_2d,
    shifts_3d, vol, wvol) in order. `vol`/`wvol` are pre-zeroed on the device by
    the caller (the kernel atomically accumulates into them).
    """
    var p = _scatter_params(bufs, params_obj)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength * p.proj_sidelength_half()

    var ctx = _session_ctx(session_obj)
    _launch_scatter(
        ctx,
        _dptr(addrs_obj[0]),
        _dptr(addrs_obj[1]),
        _dptr(addrs_obj[2]),
        _dptr(addrs_obj[3]),
        _dptr(addrs_obj[4]),
        _dptr(addrs_obj[5]),
        _dptr(addrs_obj[6]),
        total,
        p,
    )
    ctx.synchronize()  # writes land before Python reads vol/wvol
    return PythonObject(0)


def extract_central_slices_rfft_3d_pose_grad_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Forward-projection rotation/shift gradients on the GPU (zero-copy).

    `addrs_obj` holds device VAs for bufs = (rec, rot, shifts_2d, shifts_3d,
    grad_proj, grad_rot, grad_shift, grad_shift_3d). The three grad_* outputs
    are pre-zeroed on the device by the caller (atomically accumulated).
    """
    var p = _pose_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength * p.proj_sidelength_half()

    var ctx = _session_ctx(session_obj)
    _launch_forward_pose_grad(
        ctx,
        _dptr(addrs_obj[0]),
        _dptr(addrs_obj[1]),
        _dptr(addrs_obj[2]),
        _dptr(addrs_obj[3]),
        _dptr(addrs_obj[4]),
        _dptr(addrs_obj[5]),
        _dptr(addrs_obj[6]),
        _dptr(addrs_obj[7]),
        total,
        p,
    )
    ctx.synchronize()  # writes land before Python reads the grad tensors
    return PythonObject(0)


def insert_central_slices_rfft_3d_pose_grad_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Backprojection rotation/shift gradients on the GPU (zero-copy).

    `addrs_obj` holds device VAs for bufs = (grad_rec, rot, shifts_2d,
    shifts_3d, proj, grad_rot, grad_shift, grad_shift_3d). The three grad_*
    outputs are pre-zeroed on the device by the caller (atomically accumulated).
    """
    var p = _pose_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength * p.proj_sidelength_half()

    var ctx = _session_ctx(session_obj)
    _launch_backproject_pose_grad(
        ctx,
        _dptr(addrs_obj[0]),
        _dptr(addrs_obj[1]),
        _dptr(addrs_obj[2]),
        _dptr(addrs_obj[3]),
        _dptr(addrs_obj[4]),
        _dptr(addrs_obj[5]),
        _dptr(addrs_obj[6]),
        _dptr(addrs_obj[7]),
        total,
        p,
    )
    ctx.synchronize()  # writes land before Python reads the grad tensors
    return PythonObject(0)


def insert_central_slices_rfft_3d_weight_grad_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Backprojection weight gradients on the GPU (zero-copy).

    `addrs_obj` holds device VAs for the used buffers gwvol=bufs[0], rot=bufs[1]
    and grad_weight=bufs[4] (at addr indices 0, 1, 4 to match the buffer order;
    indices 2/3 are the unused shift placeholders). `grad_weight` is pre-zeroed
    on the device by the caller.
    """
    var p = _pose_grad_params(bufs, params_obj, 1)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength * p.proj_sidelength_half()

    var ctx = _session_ctx(session_obj)
    _launch_weight_grad(
        ctx, _dptr(addrs_obj[0]), _dptr(addrs_obj[1]), _dptr(addrs_obj[4]), total, p
    )
    ctx.synchronize()  # writes land before Python reads grad_weight
    return PythonObject(0)


# ===========================================================================
# FourierSliceParams construction from Python args
# ===========================================================================


@always_inline
def _forward_params(
    rec_obj: PythonObject,
    rot_obj: PythonObject,
    shifts_2d_obj: PythonObject,
    shifts_3d_obj: PythonObject,
    proj_obj: PythonObject,
    params_obj: PythonObject,
) raises -> FourierSliceParams:
    return _forward_params_for_sidelength(
        Int(py=rec_obj.shape[2]),
        rot_obj,
        shifts_2d_obj,
        shifts_3d_obj,
        proj_obj,
        params_obj,
    )


@always_inline
def _forward_params_for_sidelength(
    sidelength: Int,
    rot_obj: PythonObject,
    shifts_2d_obj: PythonObject,
    shifts_3d_obj: PythonObject,
    proj_obj: PythonObject,
    params_obj: PythonObject,
) raises -> FourierSliceParams:
    return FourierSliceParams(
        bp=Int(py=rot_obj.shape[1]),
        sidelength=sidelength,
        proj_sidelength=Int(py=proj_obj.shape[2]),
        bv_rot=Int(py=rot_obj.shape[0]),
        bv_shift_2d=Int(py=shifts_2d_obj.shape[0]),
        oversampling=Float32(py=params_obj[0]),
        radius_cutoff_sq=Float32(py=params_obj[1]),
        has_shifts_2d=Int(py=params_obj[2]),
        interp=Int(py=params_obj[3]),
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        ewald_curvature=Float32(py=params_obj[4]),
        has_shifts_3d=Int(py=params_obj[5]),
        bv_shift_3d=Int(py=shifts_3d_obj.shape[0]),
    )


@always_inline
def _pose_grad_params(
    bufs: PythonObject, params_obj: PythonObject, friedel_double: Int
) raises -> FourierSliceParams:
    """Params for the pose/weight gradient kernels.

    bufs[0] is the volume (rec / grad_rec / grad_weight_vol), bufs[1] rotations,
    bufs[2] shifts_2d (2D), bufs[3] shifts_3d, bufs[4] the projection-shaped tensor.
    params_obj is the forward tuple
    (oversampling, radius_cutoff_sq, has_shifts_2d, interp, ewald, has_shifts_3d).
    """
    return FourierSliceParams(
        bp=Int(py=bufs[1].shape[1]),
        sidelength=Int(py=bufs[0].shape[2]),
        proj_sidelength=Int(py=bufs[4].shape[2]),
        bv_rot=Int(py=bufs[1].shape[0]),
        bv_shift_2d=Int(py=bufs[2].shape[0]),
        oversampling=Float32(py=params_obj[0]),
        radius_cutoff_sq=Float32(py=params_obj[1]),
        has_shifts_2d=Int(py=params_obj[2]),
        interp=Int(py=params_obj[3]),
        has_weights=0,
        friedel_double=friedel_double,
        skip_redundant=0,
        ewald_curvature=Float32(py=params_obj[4]),
        has_shifts_3d=Int(py=params_obj[5]),
        bv_shift_3d=Int(py=bufs[3].shape[0]),
    )


@always_inline
def _scatter_params(
    bufs: PythonObject, params_obj: PythonObject
) raises -> FourierSliceParams:
    return FourierSliceParams(
        bp=Int(py=bufs[0].shape[1]),
        sidelength=Int(py=bufs[5].shape[2]),
        proj_sidelength=Int(py=bufs[0].shape[2]),
        bv_rot=Int(py=bufs[2].shape[0]),
        bv_shift_2d=Int(py=bufs[3].shape[0]),
        oversampling=Float32(py=params_obj[0]),
        radius_cutoff_sq=Float32(py=params_obj[1]),
        has_shifts_2d=Int(py=params_obj[2]),
        interp=Int(py=params_obj[6]),
        has_weights=Int(py=params_obj[3]),
        friedel_double=Int(py=params_obj[4]),
        skip_redundant=Int(py=params_obj[5]),
        ewald_curvature=Float32(py=params_obj[7]),
        has_shifts_3d=Int(py=params_obj[8]),
        bv_shift_3d=Int(py=bufs[4].shape[0]),
    )
