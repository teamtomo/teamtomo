"""Mojo CPU + GPU kernels for Fourier-space extraction and insertion.

This is the Python entry module of a single extension module. The Fourier-space
math is written **once** and shared by two execution strategies; the
implementation is split into grouped files:

    _common.mojo      types, constants, FourierSliceParams, geometry/complex helpers
    _gather.mojo      sample + interpolate (extraction)
    _gather_grad.mojo interpolate + analytical spatial gradient (pose gradients)
    _scatter.mojo     atomic accumulate + splat (insertion)
    _pixel.mojo       per-pixel slice ops shared by CPU loops and GPU threads
    _pose_grad.mojo   per-pixel rotation / shift / weight gradient ops
    _line.mojo        per-pixel 3D <-> 1D central-line ops
    _line_grad.mojo   per-pixel 3D line direction / shift / weight gradient ops
    _line2d.mojo      per-pixel 2D <-> 1D central-line ops
    _line2d_grad.mojo per-pixel 2D line direction / shift / weight gradient ops
    _device.mojo      GPU kernels + launchers
    fourier_slice_kernels.mojo  (this file) the Python-facing entry points

- **CPU**: `parallelize` over projections (one thread per pose), atomic scatter.
- **GPU**: one `DeviceContext` thread per rfft pixel, reading and writing the
  memory behind torch device tensors **in place** -- no host round-trip. The
  Python caller places every buffer (inputs and pre-zeroed outputs) on the
  compute device and passes their raw device addresses in the `addrs` tuple;
  each GPU entry point rebuilds device pointers via `_dptr` and launches. When
  the caller also passes a non-zero stream address (CUDA), the launch is
  enqueued on that stream and ordering is the stream's job; when it is zero
  (Metal), the entry point synchronises its own command queue before returning.
  (The CPU entry points read host memory via `_ptr`.)

Data is exchanged through raw pointers into contiguous tensors viewed as real
(`torch.view_as_real`: complex64 -> trailing dim 2) -- on the host for the CPU
kernels, on the device for the GPU kernels. All shapes/scalars are read once
into a `FourierSliceParams` value so the hot loops never call back into Python.

Tensor layouts (C-contiguous, rfft with DC at the origin):
    volume      float32 [bv, d, h, w, 2]        (complex; h = sidelength, w = h//2+1)
    slices      float32 [bv, bp, h, w, 2]       (complex)
    lines       float32 [bv, bp, w, 2]          (complex half-line)
    weights     float32 [bv, bp, h, w]          (real, insertion only)
    rotations   float32 [bv_rot, bp, 3, 3]      (zyx convention)
    directions  float32 [bv_rot, bp, 3]         (zyx unit vectors; lines)
    shifts_2d   float32 [bv_shift_2d, bp, 2]
    shifts_3d   float32 [bv_shift_3d, bp, 3]
"""

from std.algorithm import parallelize
from std.os import abort
from std.gpu.host import DeviceContext
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import num_physical_cores

from _common import (
    CUBIC,
    LINEAR,
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
    _dptr,
    _ptr,
)
from _device import (
    _launch_backproject_line2d_pose_grad,
    _launch_backproject_line_pose_grad,
    _launch_backproject_pose_grad,
    _launch_forward_line2d_pose_grad,
    _launch_forward_line_pose_grad,
    _launch_forward_pose_grad,
    _launch_project,
    _launch_project_line,
    _launch_project_line2d,
    _launch_scatter,
    _launch_scatter_line,
    _launch_scatter_line2d,
    _launch_weight_grad,
    _launch_weight_line2d_grad,
    _launch_weight_line_grad,
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


@export
def PyInit_fourier_slice_kernels() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("fourier_slice_kernels")
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
        m.def_function[extract_central_line_rfft_3d](
            "extract_central_line_rfft_3d",
            docstring="Forward 3D->1D central-line projection (CPU).",
        )
        m.def_function[extract_central_line_rfft_3d_gpu](
            "extract_central_line_rfft_3d_gpu",
            docstring="Forward 3D->1D central-line projection (GPU).",
        )
        m.def_function[insert_central_line_rfft_3d](
            "insert_central_line_rfft_3d",
            docstring="Scatter 1D central lines into a 3D volume (CPU).",
        )
        m.def_function[insert_central_line_rfft_3d_gpu](
            "insert_central_line_rfft_3d_gpu",
            docstring="Scatter 1D central lines into a 3D volume (GPU).",
        )
        m.def_function[extract_central_line_rfft_3d_pose_grad](
            "extract_central_line_rfft_3d_pose_grad",
            docstring="Forward-line direction/3D-shift gradients (CPU).",
        )
        m.def_function[extract_central_line_rfft_3d_pose_grad_gpu](
            "extract_central_line_rfft_3d_pose_grad_gpu",
            docstring="Forward-line direction/3D-shift gradients (GPU).",
        )
        m.def_function[insert_central_line_rfft_3d_pose_grad](
            "insert_central_line_rfft_3d_pose_grad",
            docstring="Line-insertion direction/3D-shift gradients (CPU).",
        )
        m.def_function[insert_central_line_rfft_3d_pose_grad_gpu](
            "insert_central_line_rfft_3d_pose_grad_gpu",
            docstring="Line-insertion direction/3D-shift gradients (GPU).",
        )
        m.def_function[insert_central_line_rfft_3d_weight_grad](
            "insert_central_line_rfft_3d_weight_grad",
            docstring="Line-insertion weight gradients (CPU).",
        )
        m.def_function[insert_central_line_rfft_3d_weight_grad_gpu](
            "insert_central_line_rfft_3d_weight_grad_gpu",
            docstring="Line-insertion weight gradients (GPU).",
        )
        m.def_function[extract_central_line_rfft_2d](
            "extract_central_line_rfft_2d",
            docstring="Forward 2D->1D central-line projection (CPU).",
        )
        m.def_function[extract_central_line_rfft_2d_gpu](
            "extract_central_line_rfft_2d_gpu",
            docstring="Forward 2D->1D central-line projection (GPU).",
        )
        m.def_function[insert_central_line_rfft_2d](
            "insert_central_line_rfft_2d",
            docstring="Scatter 1D central lines into a 2D image (CPU).",
        )
        m.def_function[insert_central_line_rfft_2d_gpu](
            "insert_central_line_rfft_2d_gpu",
            docstring="Scatter 1D central lines into a 2D image (GPU).",
        )
        m.def_function[extract_central_line_rfft_2d_pose_grad](
            "extract_central_line_rfft_2d_pose_grad",
            docstring="Forward 2D-line direction gradients (CPU).",
        )
        m.def_function[extract_central_line_rfft_2d_pose_grad_gpu](
            "extract_central_line_rfft_2d_pose_grad_gpu",
            docstring="Forward 2D-line direction gradients (GPU).",
        )
        m.def_function[insert_central_line_rfft_2d_pose_grad](
            "insert_central_line_rfft_2d_pose_grad",
            docstring="2D-line insertion direction gradients (CPU).",
        )
        m.def_function[insert_central_line_rfft_2d_pose_grad_gpu](
            "insert_central_line_rfft_2d_pose_grad_gpu",
            docstring="2D-line insertion direction gradients (GPU).",
        )
        m.def_function[insert_central_line_rfft_2d_weight_grad](
            "insert_central_line_rfft_2d_weight_grad",
            docstring="2D-line insertion weight gradients (CPU).",
        )
        m.def_function[insert_central_line_rfft_2d_weight_grad_gpu](
            "insert_central_line_rfft_2d_weight_grad_gpu",
            docstring="2D-line insertion weight gradients (GPU).",
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
        abort(String("failed to create fourier_slice_kernels module: ", e))


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
    """Forward project on the CPU; writes `proj_obj` (pre-zeroed by the caller).
    """
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
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for y in range(p.proj_sidelength):
            for x in range(p.proj_sidelength_half()):
                _project_pixel[interp](
                    rec, rot, shifts_2d, shifts_3d, proj, i_bv, i_bp, y, x, p
                )

    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
    return PythonObject(0)


def insert_central_slices_rfft_3d(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Scatter on the CPU; writes vol (and wvol if weighted), pre-zeroed by caller.

    bufs   : (slices, weights, rot, shifts_2d, vol, wvol) viewed as real.
    params_obj : a `KernelParams` (read by field name; see `_validation.py`).
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
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for y in range(p.proj_sidelength):
            for x in range(p.proj_sidelength_half()):
                _scatter_pixel[interp](
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

    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
    return PythonObject(0)


# ===========================================================================
# Central-line CPU entry points (3D <-> 1D; loop over the line's x only)
# ===========================================================================


def extract_central_line_rfft_3d(
    rec_obj: PythonObject,
    direction_obj: PythonObject,
    shifts_3d_obj: PythonObject,
    line_obj: PythonObject,
    params_obj: PythonObject,
) raises -> PythonObject:
    """Forward project 1D central lines on the CPU; writes `line_obj` (pre-zeroed).
    """
    var rec = _ptr(rec_obj)
    var direction = _ptr(direction_obj)
    var shifts_3d = _ptr(shifts_3d_obj)
    var line = _ptr(line_obj)
    var p = _forward_line_params(
        rec_obj, direction_obj, shifts_3d_obj, line_obj, params_obj
    )
    var bv = Int(py=rec_obj.shape[0])
    var lsh = p.proj_sidelength_half()

    @parameter
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for x in range(lsh):
            _project_line_pixel[interp](
                rec, direction, shifts_3d, line, i_bv, i_bp, x, p
            )

    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
    return PythonObject(0)


def insert_central_line_rfft_3d(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Scatter 1D central lines into a 3D volume on the CPU (pre-zeroed by caller).

    bufs : (lines, weights, direction, shifts_3d, vol, wvol) viewed as real.
    """
    var inp = _ptr(bufs[0])
    var weights = _ptr(bufs[1])
    var direction = _ptr(bufs[2])
    var shifts_3d = _ptr(bufs[3])
    var vol = _ptr(bufs[4])
    var wvol = _ptr(bufs[5])
    var p = _scatter_line_params(bufs, params_obj)
    var bv = Int(py=bufs[0].shape[0])
    var lsh = p.proj_sidelength_half()

    @parameter
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for x in range(lsh):
            _scatter_line_pixel[interp](
                inp, weights, direction, shifts_3d, vol, wvol, i_bv, i_bp, x, p
            )

    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
    return PythonObject(0)


# ===========================================================================
# 2D->1D central-line CPU entry points
# ===========================================================================


def extract_central_line_rfft_2d(
    img_obj: PythonObject,
    direction_obj: PythonObject,
    shifts_2d_obj: PythonObject,
    line_obj: PythonObject,
    params_obj: PythonObject,
) raises -> PythonObject:
    """Forward project 1D central lines from 2D images on the CPU (pre-zeroed line).
    """
    var img = _ptr(img_obj)
    var direction = _ptr(direction_obj)
    var shifts_2d = _ptr(shifts_2d_obj)
    var line = _ptr(line_obj)
    var p = _forward_line2d_params(
        img_obj, direction_obj, shifts_2d_obj, line_obj, params_obj
    )
    var bv = Int(py=img_obj.shape[0])
    var lsh = p.proj_sidelength_half()

    @parameter
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for x in range(lsh):
            _project_line2d_pixel[interp](
                img, direction, shifts_2d, line, i_bv, i_bp, x, p
            )

    if p.interp == CUBIC:
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
    return PythonObject(0)


def insert_central_line_rfft_2d(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Scatter 1D central lines into a 2D image on the CPU (pre-zeroed by caller).

    bufs : (lines, weights, direction, shifts_2d, vol, wvol) viewed as real.
    """
    var inp = _ptr(bufs[0])
    var weights = _ptr(bufs[1])
    var direction = _ptr(bufs[2])
    var shifts_2d = _ptr(bufs[3])
    var vol = _ptr(bufs[4])
    var wvol = _ptr(bufs[5])
    var p = _scatter_line2d_params(bufs, params_obj)
    var bv = Int(py=bufs[0].shape[0])
    var lsh = p.proj_sidelength_half()

    @parameter
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for x in range(lsh):
            _scatter_line2d_pixel[interp](
                inp, weights, direction, shifts_2d, vol, wvol, i_bv, i_bp, x, p
            )

    if p.interp == CUBIC:
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
    return PythonObject(0)


# ===========================================================================
# 2D->1D central-line pose / weight gradient CPU entry points
# ===========================================================================


def extract_central_line_rfft_2d_pose_grad(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Forward 2D-line direction gradients on the CPU.

    bufs : (img, direction, shifts_2d, grad_line, grad_dir, grad_shift);
    grad_* pre-zeroed by caller.
    """
    var img = _ptr(bufs[0])
    var direction = _ptr(bufs[1])
    var shifts_2d = _ptr(bufs[2])
    var grad_line = _ptr(bufs[3])
    var grad_dir = _ptr(bufs[4])
    var grad_shift = _ptr(bufs[5])
    var p = _line2d_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var lsh = p.proj_sidelength_half()

    @parameter
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for x in range(lsh):
            _forward_line2d_pose_grad_pixel[interp](
                img,
                direction,
                shifts_2d,
                grad_line,
                grad_dir,
                grad_shift,
                i_bv,
                i_bp,
                x,
                p,
            )

    if p.interp == CUBIC:
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
    return PythonObject(0)


def insert_central_line_rfft_2d_pose_grad(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """2D-line insertion direction gradients on the CPU.

    bufs : (grad_img, direction, shifts_2d, lines, grad_dir, grad_shift);
    grad_* pre-zeroed by caller.
    """
    var grad_img = _ptr(bufs[0])
    var direction = _ptr(bufs[1])
    var shifts_2d = _ptr(bufs[2])
    var lines = _ptr(bufs[3])
    var grad_dir = _ptr(bufs[4])
    var grad_shift = _ptr(bufs[5])
    var p = _line2d_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var lsh = p.proj_sidelength_half()

    @parameter
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for x in range(lsh):
            _backproject_line2d_pose_grad_pixel[interp](
                grad_img,
                direction,
                shifts_2d,
                lines,
                grad_dir,
                grad_shift,
                i_bv,
                i_bp,
                x,
                p,
            )

    if p.interp == CUBIC:
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
    return PythonObject(0)


def insert_central_line_rfft_2d_weight_grad(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """2D-line insertion weight gradients on the CPU.

    bufs : (gwimg, direction, grad_weight); grad_weight pre-zeroed by caller.
    """
    var gwimg = _ptr(bufs[0])
    var direction = _ptr(bufs[1])
    var grad_weight = _ptr(bufs[2])
    var p = _line2d_grad_params(bufs, params_obj, 1)
    var bv = Int(py=bufs[0].shape[0])
    var lsh = p.proj_sidelength_half()

    @parameter
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for x in range(lsh):
            _weight_line2d_grad_pixel[interp](
                gwimg, direction, grad_weight, i_bv, i_bp, x, p
            )

    if p.interp == CUBIC:
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
    return PythonObject(0)


# ===========================================================================
# Central-line pose / weight gradient CPU entry points
# ===========================================================================


def extract_central_line_rfft_3d_pose_grad(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Forward-line direction/3D-shift gradients on the CPU.

    bufs : (rec, direction, shifts_3d, grad_line, grad_dir, grad_shift_3d);
    grad_* pre-zeroed by the caller.
    """
    var rec = _ptr(bufs[0])
    var direction = _ptr(bufs[1])
    var shifts_3d = _ptr(bufs[2])
    var grad_line = _ptr(bufs[3])
    var grad_dir = _ptr(bufs[4])
    var grad_shift_3d = _ptr(bufs[5])
    var p = _line_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var lsh = p.proj_sidelength_half()

    @parameter
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for x in range(lsh):
            _forward_line_pose_grad_pixel[interp](
                rec,
                direction,
                shifts_3d,
                grad_line,
                grad_dir,
                grad_shift_3d,
                i_bv,
                i_bp,
                x,
                p,
            )

    if p.interp == CUBIC:
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
    return PythonObject(0)


def insert_central_line_rfft_3d_pose_grad(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Line-insertion direction/3D-shift gradients on the CPU.

    bufs : (grad_rec, direction, shifts_3d, lines, grad_dir, grad_shift_3d);
    grad_* pre-zeroed by the caller.
    """
    var grad_rec = _ptr(bufs[0])
    var direction = _ptr(bufs[1])
    var shifts_3d = _ptr(bufs[2])
    var lines = _ptr(bufs[3])
    var grad_dir = _ptr(bufs[4])
    var grad_shift_3d = _ptr(bufs[5])
    var p = _line_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var lsh = p.proj_sidelength_half()

    @parameter
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for x in range(lsh):
            _backproject_line_pose_grad_pixel[interp](
                grad_rec,
                direction,
                shifts_3d,
                lines,
                grad_dir,
                grad_shift_3d,
                i_bv,
                i_bp,
                x,
                p,
            )

    if p.interp == CUBIC:
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
    return PythonObject(0)


def insert_central_line_rfft_3d_weight_grad(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Line-insertion weight gradients on the CPU.

    bufs : (gwvol, direction, grad_weight); grad_weight pre-zeroed by the caller.
    """
    var gwvol = _ptr(bufs[0])
    var direction = _ptr(bufs[1])
    var grad_weight = _ptr(bufs[2])
    var p = _line_grad_params(bufs, params_obj, 1)
    var bv = Int(py=bufs[0].shape[0])
    var lsh = p.proj_sidelength_half()

    @parameter
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for x in range(lsh):
            _weight_line_grad_pixel[interp](
                gwvol, direction, grad_weight, i_bv, i_bp, x, p
            )

    if p.interp == CUBIC:
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
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
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for y in range(p.proj_sidelength):
            for x in range(p.proj_sidelength_half()):
                _forward_pose_grad_pixel[interp](
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

    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
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
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for y in range(p.proj_sidelength):
            for x in range(p.proj_sidelength_half()):
                _backproject_pose_grad_pixel[interp](
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

    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
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
    def worker[interp: Int](vp: Int):
        var i_bv = vp // p.bp
        var i_bp = vp % p.bp
        for y in range(p.proj_sidelength):
            for x in range(p.proj_sidelength_half()):
                _weight_grad_pixel[interp](
                    gwvol, rot, grad_weight, i_bv, i_bp, y, x, p
                )

    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        parallelize[worker[CUBIC]](bv * p.bp, num_physical_cores())
    else:
        parallelize[worker[LINEAR]](bv * p.bp, num_physical_cores())
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
    (rec, rot, shifts_2d, shifts_3d, proj), then the foreign (torch) GPU stream
    address as the trailing element (0 to enqueue on the context's own stream;
    see `_launch_project`). The tensor objects are used only for their shapes.
    `proj` is pre-zeroed on the device by the caller (radius-cut pixels stay 0,
    matching the CPU path). We sync the context only on the own-stream path.
    """
    var p = _forward_params(
        rec_obj, rot_obj, shifts_2d_obj, shifts_3d_obj, proj_obj, params_obj
    )
    var bv = Int(py=rec_obj.shape[0])
    var total = bv * p.bp * p.proj_sidelength * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[5])  # trailing element after 5 buffers

    var ctx = _session_ctx(session_obj)
    var buffers = ProjectBuffers(
        rec=_dptr(addrs_obj[0]),
        rot=_dptr(addrs_obj[1]),
        shifts_2d=_dptr(addrs_obj[2]),
        shifts_3d=_dptr(addrs_obj[3]),
        proj=_dptr(addrs_obj[4]),
    )
    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        _launch_project[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_project[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()  # own-stream path: writes land before Python reads
    return PythonObject(0)


def insert_central_slices_rfft_3d_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Scatter on the GPU, reading/writing torch device memory directly.

    `addrs_obj` holds device VAs for bufs = (slices, weights, rot, shifts_2d,
    shifts_3d, vol, wvol) then the foreign stream address as the trailing
    element (0 for the own-stream path; see `extract_central_slices_rfft_3d_gpu`).
    `vol`/`wvol` are pre-zeroed on the device by the caller (atomic accumulate).
    """
    var p = _scatter_params(bufs, params_obj)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[7])  # trailing element after 7 buffers

    var ctx = _session_ctx(session_obj)
    var buffers = ScatterBuffers(
        inp=_dptr(addrs_obj[0]),
        weights=_dptr(addrs_obj[1]),
        rot=_dptr(addrs_obj[2]),
        shifts_2d=_dptr(addrs_obj[3]),
        shifts_3d=_dptr(addrs_obj[4]),
        vol=_dptr(addrs_obj[5]),
        wvol=_dptr(addrs_obj[6]),
    )
    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        _launch_scatter[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_scatter[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()  # own-stream path: writes land before Python reads
    return PythonObject(0)


# ===========================================================================
# Central-line GPU entry points (zero-copy)
# ===========================================================================


def extract_central_line_rfft_3d_gpu(
    session_obj: PythonObject,
    rec_obj: PythonObject,
    direction_obj: PythonObject,
    shifts_3d_obj: PythonObject,
    line_obj: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Forward project 1D central lines on the GPU, reading/writing torch memory.

    `addrs_obj` carries device VAs of (rec, direction, shifts_3d, line) then the
    foreign stream address as the trailing element (0 for the own-stream path).
    `line` is pre-zeroed on the device by the caller (radius-cut pixels stay 0).
    """
    var p = _forward_line_params(
        rec_obj, direction_obj, shifts_3d_obj, line_obj, params_obj
    )
    var bv = Int(py=rec_obj.shape[0])
    var total = bv * p.bp * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[4])  # trailing element after 4 buffers

    var ctx = _session_ctx(session_obj)
    var buffers = ProjectLineBuffers(
        rec=_dptr(addrs_obj[0]),
        direction=_dptr(addrs_obj[1]),
        shifts_3d=_dptr(addrs_obj[2]),
        line=_dptr(addrs_obj[3]),
    )
    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        _launch_project_line[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_project_line[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()  # own-stream path: writes land before Python reads
    return PythonObject(0)


def insert_central_line_rfft_3d_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Scatter 1D central lines into a 3D volume on the GPU, in place.

    `addrs_obj` holds device VAs for bufs = (lines, weights, direction, shifts_3d,
    vol, wvol) then the foreign stream address as the trailing element (index 6; 0
    for the own-stream path). `vol`/`wvol` are pre-zeroed on the device by caller.
    """
    var p = _scatter_line_params(bufs, params_obj)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[6])  # trailing element after 6 buffers

    var ctx = _session_ctx(session_obj)
    var buffers = ScatterLineBuffers(
        inp=_dptr(addrs_obj[0]),
        weights=_dptr(addrs_obj[1]),
        direction=_dptr(addrs_obj[2]),
        shifts_3d=_dptr(addrs_obj[3]),
        vol=_dptr(addrs_obj[4]),
        wvol=_dptr(addrs_obj[5]),
    )
    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        _launch_scatter_line[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_scatter_line[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()  # own-stream path: writes land before Python reads
    return PythonObject(0)


def extract_central_line_rfft_2d_gpu(
    session_obj: PythonObject,
    img_obj: PythonObject,
    direction_obj: PythonObject,
    shifts_2d_obj: PythonObject,
    line_obj: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Forward project 1D central lines from 2D images on the GPU (zero-copy).

    `addrs_obj` holds device VAs of (img, direction, shifts_2d, line) then the
    foreign stream address (index 4). `line` is pre-zeroed by caller.
    """
    var p = _forward_line2d_params(
        img_obj, direction_obj, shifts_2d_obj, line_obj, params_obj
    )
    var bv = Int(py=img_obj.shape[0])
    var total = bv * p.bp * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[4])

    var ctx = _session_ctx(session_obj)
    var buffers = ProjectLine2DBuffers(
        img=_dptr(addrs_obj[0]),
        direction=_dptr(addrs_obj[1]),
        shifts_2d=_dptr(addrs_obj[2]),
        line=_dptr(addrs_obj[3]),
    )
    if p.interp == CUBIC:
        _launch_project_line2d[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_project_line2d[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()
    return PythonObject(0)


def insert_central_line_rfft_2d_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Scatter 1D central lines into a 2D image on the GPU (zero-copy).

    `addrs_obj` holds device VAs for bufs = (lines, weights, direction, shifts_2d,
    vol, wvol) then the foreign stream address (index 6). `vol`/`wvol` pre-zeroed.
    """
    var p = _scatter_line2d_params(bufs, params_obj)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[6])

    var ctx = _session_ctx(session_obj)
    var buffers = ScatterLine2DBuffers(
        inp=_dptr(addrs_obj[0]),
        weights=_dptr(addrs_obj[1]),
        direction=_dptr(addrs_obj[2]),
        shifts_2d=_dptr(addrs_obj[3]),
        vol=_dptr(addrs_obj[4]),
        wvol=_dptr(addrs_obj[5]),
    )
    if p.interp == CUBIC:
        _launch_scatter_line2d[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_scatter_line2d[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()
    return PythonObject(0)


def extract_central_line_rfft_2d_pose_grad_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Forward 2D-line direction gradients on the GPU (zero-copy).

    `addrs_obj`: (img, direction, shifts_2d, grad_line, grad_dir, grad_shift) then
    the foreign stream address (index 6). `grad_*` pre-zeroed by the caller.
    """
    var p = _line2d_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[6])

    var ctx = _session_ctx(session_obj)
    var buffers = ForwardLine2DGradBuffers(
        img=_dptr(addrs_obj[0]),
        direction=_dptr(addrs_obj[1]),
        shifts_2d=_dptr(addrs_obj[2]),
        grad_line=_dptr(addrs_obj[3]),
        grad_dir=_dptr(addrs_obj[4]),
        grad_shift=_dptr(addrs_obj[5]),
    )
    if p.interp == CUBIC:
        _launch_forward_line2d_pose_grad[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_forward_line2d_pose_grad[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()
    return PythonObject(0)


def insert_central_line_rfft_2d_pose_grad_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """2D-line insertion direction gradients on the GPU (zero-copy).

    `addrs_obj`: (grad_img, direction, shifts_2d, lines, grad_dir, grad_shift) then
    the foreign stream address (index 6). `grad_*` pre-zeroed by the caller.
    """
    var p = _line2d_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[6])

    var ctx = _session_ctx(session_obj)
    var buffers = BackprojectLine2DGradBuffers(
        grad_img=_dptr(addrs_obj[0]),
        direction=_dptr(addrs_obj[1]),
        shifts_2d=_dptr(addrs_obj[2]),
        lines=_dptr(addrs_obj[3]),
        grad_dir=_dptr(addrs_obj[4]),
        grad_shift=_dptr(addrs_obj[5]),
    )
    if p.interp == CUBIC:
        _launch_backproject_line2d_pose_grad[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_backproject_line2d_pose_grad[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()
    return PythonObject(0)


def insert_central_line_rfft_2d_weight_grad_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """2D-line insertion weight gradients on the GPU (zero-copy).

    `addrs_obj`: (gwimg, direction, grad_weight) then the foreign stream address
    (index 3). `grad_weight` is pre-zeroed by the caller.
    """
    var p = _line2d_grad_params(bufs, params_obj, 1)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[3])

    var ctx = _session_ctx(session_obj)
    var buffers = WeightLine2DGradBuffers(
        gwimg=_dptr(addrs_obj[0]),
        direction=_dptr(addrs_obj[1]),
        grad_weight=_dptr(addrs_obj[2]),
    )
    if p.interp == CUBIC:
        _launch_weight_line2d_grad[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_weight_line2d_grad[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()
    return PythonObject(0)


def extract_central_line_rfft_3d_pose_grad_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Forward-line direction/3D-shift gradients on the GPU (zero-copy).

    `addrs_obj` holds device VAs for bufs = (rec, direction, shifts_3d, grad_line,
    grad_dir, grad_shift_3d) then the foreign stream address (index 6). The two
    grad_* outputs are pre-zeroed on the device by the caller.
    """
    var p = _line_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[6])

    var ctx = _session_ctx(session_obj)
    var buffers = ForwardLineGradBuffers(
        rec=_dptr(addrs_obj[0]),
        direction=_dptr(addrs_obj[1]),
        shifts_3d=_dptr(addrs_obj[2]),
        grad_line=_dptr(addrs_obj[3]),
        grad_dir=_dptr(addrs_obj[4]),
        grad_shift_3d=_dptr(addrs_obj[5]),
    )
    if p.interp == CUBIC:
        _launch_forward_line_pose_grad[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_forward_line_pose_grad[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()
    return PythonObject(0)


def insert_central_line_rfft_3d_pose_grad_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Line-insertion direction/3D-shift gradients on the GPU (zero-copy).

    `addrs_obj` holds device VAs for bufs = (grad_rec, direction, shifts_3d,
    lines, grad_dir, grad_shift_3d) then the foreign stream address (index 6). The
    two grad_* outputs are pre-zeroed on the device by the caller.
    """
    var p = _line_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[6])

    var ctx = _session_ctx(session_obj)
    var buffers = BackprojectLineGradBuffers(
        grad_rec=_dptr(addrs_obj[0]),
        direction=_dptr(addrs_obj[1]),
        shifts_3d=_dptr(addrs_obj[2]),
        lines=_dptr(addrs_obj[3]),
        grad_dir=_dptr(addrs_obj[4]),
        grad_shift_3d=_dptr(addrs_obj[5]),
    )
    if p.interp == CUBIC:
        _launch_backproject_line_pose_grad[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_backproject_line_pose_grad[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()
    return PythonObject(0)


def insert_central_line_rfft_3d_weight_grad_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Line-insertion weight gradients on the GPU (zero-copy).

    `addrs_obj` holds device VAs for bufs = (gwvol, direction, grad_weight) then
    the foreign stream address (index 3). `grad_weight` is pre-zeroed by caller.
    """
    var p = _line_grad_params(bufs, params_obj, 1)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[3])

    var ctx = _session_ctx(session_obj)
    var buffers = WeightLineGradBuffers(
        gwvol=_dptr(addrs_obj[0]),
        direction=_dptr(addrs_obj[1]),
        grad_weight=_dptr(addrs_obj[2]),
    )
    if p.interp == CUBIC:
        _launch_weight_line_grad[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_weight_line_grad[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()
    return PythonObject(0)


def extract_central_slices_rfft_3d_pose_grad_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Forward-projection rotation/shift gradients on the GPU (zero-copy).

    `addrs_obj` holds device VAs for bufs = (rec, rot, shifts_2d, shifts_3d,
    grad_proj, grad_rot, grad_shift, grad_shift_3d) then the foreign stream
    address as the trailing element (0 for the own-stream path). The three
    grad_* outputs are pre-zeroed on the device by the caller (atomic accumulate).
    """
    var p = _pose_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[8])  # trailing element after 8 buffers

    var ctx = _session_ctx(session_obj)
    var buffers = ForwardGradBuffers(
        rec=_dptr(addrs_obj[0]),
        rot=_dptr(addrs_obj[1]),
        shifts_2d=_dptr(addrs_obj[2]),
        shifts_3d=_dptr(addrs_obj[3]),
        grad_proj=_dptr(addrs_obj[4]),
        grad_rot=_dptr(addrs_obj[5]),
        grad_shift=_dptr(addrs_obj[6]),
        grad_shift_3d=_dptr(addrs_obj[7]),
    )
    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        _launch_forward_pose_grad[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_forward_pose_grad[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()  # own-stream path: writes land before Python reads
    return PythonObject(0)


def insert_central_slices_rfft_3d_pose_grad_gpu(
    session_obj: PythonObject,
    bufs: PythonObject,
    params_obj: PythonObject,
    addrs_obj: PythonObject,
) raises -> PythonObject:
    """Backprojection rotation/shift gradients on the GPU (zero-copy).

    `addrs_obj` holds device VAs for bufs = (grad_rec, rot, shifts_2d,
    shifts_3d, proj, grad_rot, grad_shift, grad_shift_3d) then the foreign
    stream address as the trailing element (0 for the own-stream path). The
    three grad_* outputs are pre-zeroed on the device by the caller (atomic
    accumulate).
    """
    var p = _pose_grad_params(bufs, params_obj, 0)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength * p.proj_sidelength_half()
    var stream_addr = Int(py=addrs_obj[8])  # trailing element after 8 buffers

    var ctx = _session_ctx(session_obj)
    var buffers = BackprojectGradBuffers(
        grad_rec=_dptr(addrs_obj[0]),
        rot=_dptr(addrs_obj[1]),
        shifts_2d=_dptr(addrs_obj[2]),
        shifts_3d=_dptr(addrs_obj[3]),
        proj=_dptr(addrs_obj[4]),
        grad_rot=_dptr(addrs_obj[5]),
        grad_shift=_dptr(addrs_obj[6]),
        grad_shift_3d=_dptr(addrs_obj[7]),
    )
    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        _launch_backproject_pose_grad[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_backproject_pose_grad[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()  # own-stream path: writes land before Python reads
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
    indices 2/3 are the unused shift placeholders), then the foreign stream
    address as the trailing element (index 5; 0 for the own-stream path).
    `grad_weight` is pre-zeroed on the device by the caller.
    """
    var p = _pose_grad_params(bufs, params_obj, 1)
    var bv = Int(py=bufs[0].shape[0])
    var total = bv * p.bp * p.proj_sidelength * p.proj_sidelength_half()
    var stream_addr = Int(
        py=addrs_obj[5]
    )  # trailing element after 5 addr slots

    var ctx = _session_ctx(session_obj)
    var buffers = WeightGradBuffers(
        gwvol=_dptr(addrs_obj[0]),
        rot=_dptr(addrs_obj[1]),
        grad_weight=_dptr(addrs_obj[4]),
    )
    if (
        p.interp == CUBIC
    ):  # runtime interp code -> pick the comptime specialisation once
        _launch_weight_grad[CUBIC](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    else:
        _launch_weight_grad[LINEAR](
            ctx, buffers=buffers, total=total, p=p, stream_addr=stream_addr
        )
    if stream_addr == 0:
        ctx.synchronize()  # own-stream path: writes land before Python reads
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
        oversampling=Float32(py=params_obj.oversampling),
        radius_cutoff_sq=Float32(py=params_obj.radius_cutoff_sq),
        has_shifts_2d=Int(py=params_obj.has_shifts_2d),
        interp=Int(py=params_obj.interp),
        has_weights=Int(py=params_obj.has_weights),
        friedel_double=Int(py=params_obj.friedel_double),
        skip_redundant=Int(py=params_obj.skip_redundant),
        ewald_curvature=Float32(py=params_obj.ewald_curvature),
        has_shifts_3d=Int(py=params_obj.has_shifts_3d),
        bv_shift_3d=Int(py=shifts_3d_obj.shape[0]),
    )


@always_inline
def _pose_grad_params(
    bufs: PythonObject, params_obj: PythonObject, friedel_double: Int
) raises -> FourierSliceParams:
    """Params for the pose/weight gradient kernels.

    bufs[0] is the volume (rec / grad_rec / grad_weight_vol), bufs[1] rotations,
    bufs[2] shifts_2d (2D), bufs[3] shifts_3d, bufs[4] the projection-shaped tensor.
    params_obj is a `KernelParams` (read by field name; see `_validation.py`).
    """
    return FourierSliceParams(
        bp=Int(py=bufs[1].shape[1]),
        sidelength=Int(py=bufs[0].shape[2]),
        proj_sidelength=Int(py=bufs[4].shape[2]),
        bv_rot=Int(py=bufs[1].shape[0]),
        bv_shift_2d=Int(py=bufs[2].shape[0]),
        oversampling=Float32(py=params_obj.oversampling),
        radius_cutoff_sq=Float32(py=params_obj.radius_cutoff_sq),
        has_shifts_2d=Int(py=params_obj.has_shifts_2d),
        interp=Int(py=params_obj.interp),
        has_weights=Int(py=params_obj.has_weights),
        friedel_double=friedel_double,  # supplied by the entry point (0 / 1), not the params carrier
        skip_redundant=Int(py=params_obj.skip_redundant),
        ewald_curvature=Float32(py=params_obj.ewald_curvature),
        has_shifts_3d=Int(py=params_obj.has_shifts_3d),
        bv_shift_3d=Int(py=bufs[3].shape[0]),
    )


@always_inline
def _line2d_grad_params(
    bufs: PythonObject, params_obj: PythonObject, weight_grad: Int
) raises -> FourierSliceParams:
    """Params for the 2D line pose/weight gradient kernels.

    Pose grad (weight_grad=0): bufs = (img, direction, shifts_2d, line_pixels,
    grad_dir, grad_shift); friedel_double=0 (Python symmetrises the kx=0 column).
    Weight grad (weight_grad=1): bufs = (gwimg, direction, grad_weight);
    friedel_double=1, no shift.
    """
    if weight_grad != 0:
        var line_half_w = Int(py=bufs[2].shape[2])
        return FourierSliceParams(
            bp=Int(py=bufs[1].shape[1]),
            sidelength=Int(py=bufs[0].shape[1]),
            proj_sidelength=2 * (line_half_w - 1),
            bv_rot=Int(py=bufs[1].shape[0]),
            bv_shift_2d=1,
            oversampling=Float32(py=params_obj.oversampling),
            radius_cutoff_sq=Float32(py=params_obj.radius_cutoff_sq),
            has_shifts_2d=0,
            interp=Int(py=params_obj.interp),
            has_weights=0,
            friedel_double=1,
            skip_redundant=0,
            ewald_curvature=0.0,
            has_shifts_3d=0,
            bv_shift_3d=1,
        )
    var line_half = Int(py=bufs[3].shape[2])
    return FourierSliceParams(
        bp=Int(py=bufs[1].shape[1]),
        sidelength=Int(py=bufs[0].shape[1]),
        proj_sidelength=2 * (line_half - 1),
        bv_rot=Int(py=bufs[1].shape[0]),
        bv_shift_2d=Int(py=bufs[2].shape[0]),
        oversampling=Float32(py=params_obj.oversampling),
        radius_cutoff_sq=Float32(py=params_obj.radius_cutoff_sq),
        has_shifts_2d=Int(py=params_obj.has_shifts_2d),
        interp=Int(py=params_obj.interp),
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        ewald_curvature=0.0,
        has_shifts_3d=0,
        bv_shift_3d=1,
    )


@always_inline
def _forward_line2d_params(
    img_obj: PythonObject,
    direction_obj: PythonObject,
    shifts_2d_obj: PythonObject,
    line_obj: PythonObject,
    params_obj: PythonObject,
) raises -> FourierSliceParams:
    """Params for the forward 2D line kernel; `img_obj` real-view (bv, h, w, 2).
    """
    var line_half = Int(py=line_obj.shape[2])
    return FourierSliceParams(
        bp=Int(py=direction_obj.shape[1]),
        sidelength=Int(py=img_obj.shape[1]),
        proj_sidelength=2 * (line_half - 1),
        bv_rot=Int(py=direction_obj.shape[0]),
        bv_shift_2d=Int(py=shifts_2d_obj.shape[0]),
        oversampling=Float32(py=params_obj.oversampling),
        radius_cutoff_sq=Float32(py=params_obj.radius_cutoff_sq),
        has_shifts_2d=Int(py=params_obj.has_shifts_2d),
        interp=Int(py=params_obj.interp),
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        ewald_curvature=0.0,
        has_shifts_3d=0,
        bv_shift_3d=1,
    )


@always_inline
def _scatter_line2d_params(
    bufs: PythonObject, params_obj: PythonObject
) raises -> FourierSliceParams:
    """Params for the 2D line scatter; bufs = (lines, weights, direction, shifts_2d, vol, wvol).
    """
    var line_half = Int(py=bufs[0].shape[2])
    return FourierSliceParams(
        bp=Int(py=bufs[0].shape[1]),
        sidelength=Int(py=bufs[4].shape[1]),
        proj_sidelength=2 * (line_half - 1),
        bv_rot=Int(py=bufs[2].shape[0]),
        bv_shift_2d=Int(py=bufs[3].shape[0]),
        oversampling=Float32(py=params_obj.oversampling),
        radius_cutoff_sq=Float32(py=params_obj.radius_cutoff_sq),
        has_shifts_2d=Int(py=params_obj.has_shifts_2d),
        interp=Int(py=params_obj.interp),
        has_weights=Int(py=params_obj.has_weights),
        friedel_double=Int(py=params_obj.friedel_double),
        skip_redundant=0,
        ewald_curvature=0.0,
        has_shifts_3d=0,
        bv_shift_3d=1,
    )


@always_inline
def _forward_line_params(
    rec_obj: PythonObject,
    direction_obj: PythonObject,
    shifts_3d_obj: PythonObject,
    line_obj: PythonObject,
    params_obj: PythonObject,
) raises -> FourierSliceParams:
    """Params for the forward line kernel; `line_obj` is real-view (bv, bp, w, 2).

    Directions are `(bv_dir, bp, 3)`; `bv_rot` carries the direction broadcast
    batch `bv_dir`.
    """
    var line_half = Int(py=line_obj.shape[2])
    return FourierSliceParams(
        bp=Int(py=direction_obj.shape[1]),
        sidelength=Int(py=rec_obj.shape[2]),
        proj_sidelength=2
        * (line_half - 1),  # even box whose rfft half-width is line_half
        bv_rot=Int(py=direction_obj.shape[0]),
        bv_shift_2d=1,  # unused (a line has no image plane)
        oversampling=Float32(py=params_obj.oversampling),
        radius_cutoff_sq=Float32(py=params_obj.radius_cutoff_sq),
        has_shifts_2d=0,
        interp=Int(py=params_obj.interp),
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        ewald_curvature=0.0,  # unused for a 1D line
        has_shifts_3d=Int(py=params_obj.has_shifts_3d),
        bv_shift_3d=Int(py=shifts_3d_obj.shape[0]),
    )


@always_inline
def _scatter_line_params(
    bufs: PythonObject, params_obj: PythonObject
) raises -> FourierSliceParams:
    """Params for the line scatter kernel; bufs = (lines, weights, direction, shifts_3d, vol, wvol).
    """
    var line_half = Int(py=bufs[0].shape[2])
    return FourierSliceParams(
        bp=Int(py=bufs[0].shape[1]),
        sidelength=Int(py=bufs[4].shape[2]),
        proj_sidelength=2 * (line_half - 1),
        bv_rot=Int(py=bufs[2].shape[0]),
        bv_shift_2d=1,  # unused
        oversampling=Float32(py=params_obj.oversampling),
        radius_cutoff_sq=Float32(py=params_obj.radius_cutoff_sq),
        has_shifts_2d=0,
        interp=Int(py=params_obj.interp),
        has_weights=Int(py=params_obj.has_weights),
        friedel_double=Int(py=params_obj.friedel_double),
        skip_redundant=0,
        ewald_curvature=0.0,
        has_shifts_3d=Int(py=params_obj.has_shifts_3d),
        bv_shift_3d=Int(py=bufs[3].shape[0]),
    )


@always_inline
def _line_grad_params(
    bufs: PythonObject, params_obj: PythonObject, weight_grad: Int
) raises -> FourierSliceParams:
    """Params for the line pose/weight gradient kernels.

    Pose grad (weight_grad=0): bufs = (vol, direction, shifts_3d, line_pixels,
    grad_dir, grad_shift_3d); friedel_double=0 (the Python layer symmetrises the
    kx=0 plane). Weight grad (weight_grad=1): bufs = (gwvol, direction,
    grad_weight); friedel_double=1 (adjoint of the reconstruction weight splat).
    """
    if weight_grad != 0:
        var line_half_w = Int(py=bufs[2].shape[2])
        return FourierSliceParams(
            bp=Int(py=bufs[1].shape[1]),
            sidelength=Int(py=bufs[0].shape[2]),
            proj_sidelength=2 * (line_half_w - 1),
            bv_rot=Int(py=bufs[1].shape[0]),
            bv_shift_2d=1,
            oversampling=Float32(py=params_obj.oversampling),
            radius_cutoff_sq=Float32(py=params_obj.radius_cutoff_sq),
            has_shifts_2d=0,
            interp=Int(py=params_obj.interp),
            has_weights=0,
            friedel_double=1,
            skip_redundant=0,
            ewald_curvature=0.0,
            has_shifts_3d=0,
            bv_shift_3d=1,
        )
    var line_half = Int(py=bufs[3].shape[2])
    return FourierSliceParams(
        bp=Int(py=bufs[1].shape[1]),
        sidelength=Int(py=bufs[0].shape[2]),
        proj_sidelength=2 * (line_half - 1),
        bv_rot=Int(py=bufs[1].shape[0]),
        bv_shift_2d=1,
        oversampling=Float32(py=params_obj.oversampling),
        radius_cutoff_sq=Float32(py=params_obj.radius_cutoff_sq),
        has_shifts_2d=0,
        interp=Int(py=params_obj.interp),
        has_weights=0,
        friedel_double=0,
        skip_redundant=0,
        ewald_curvature=0.0,
        has_shifts_3d=Int(py=params_obj.has_shifts_3d),
        bv_shift_3d=Int(py=bufs[2].shape[0]),
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
        oversampling=Float32(py=params_obj.oversampling),
        radius_cutoff_sq=Float32(py=params_obj.radius_cutoff_sq),
        has_shifts_2d=Int(py=params_obj.has_shifts_2d),
        interp=Int(py=params_obj.interp),
        has_weights=Int(py=params_obj.has_weights),
        friedel_double=Int(py=params_obj.friedel_double),
        skip_redundant=Int(py=params_obj.skip_redundant),
        ewald_curvature=Float32(py=params_obj.ewald_curvature),
        has_shifts_3d=Int(py=params_obj.has_shifts_3d),
        bv_shift_3d=Int(py=bufs[4].shape[0]),
    )
