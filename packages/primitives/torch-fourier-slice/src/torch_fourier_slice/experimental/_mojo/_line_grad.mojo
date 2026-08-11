"""Per-pixel backward ops for the central-*line* pose / weight gradients.

The line analogues of `_pose_grad.mojo`. A line samples along a direction
`u = (u_z, u_y, u_x)` with `k = s_x * u`, so the pose gradient is simply the
gradient w.r.t. that 3-vector: `d(value)/du_a = g_a * s_x` summed over the line.
There is no rotation matrix (hence no gauge column), no 2D image-plane shift and
no Ewald curvature; only the 3D (volume-frame) shift contributes.

The heavy machinery -- interpolation-with-spatial-gradient, the 3D-shift
coupling into the pose grad, and the weight-splat adjoint gather -- is reused
verbatim from the slice kernels; only the 1D geometry and line I/O differ.
"""

from std.atomic import Atomic, Ordering
from std.math import cos, floor, sin

from layout import TileTensor, row_major

from _common import (
    C2,
    CUBIC,
    Float32Ptr,
    FourierSliceParams,
    _cmul,
    _cubic_kernel,
    _line_k,
    _line_shift_phase,
    _rfft_half,
)
from _gather_grad import _interp3d_with_grad
from _pose_grad import _couple_shift3d, _gather_weight_grad, _redot


@always_inline
def _line_phase_factor(
    shifts_3d: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    kz: Float32,
    ky: Float32,
    kx: Float32,
    p: FourierSliceParams,
) -> C2:
    """3D-shift phase factor exp(i*phase) for this line pixel (identity if none).
    """
    if p.has_shifts_3d == 0:
        return C2(1.0, 0.0)
    var phase = _line_shift_phase(p, shifts_3d, i_bv, i_bp, kz, ky, kx)
    return C2(cos(phase), sin(phase))


@always_inline
def _accumulate_line_pose_grads(
    grad_dir: Float32Ptr,
    grad_shift_3d: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    sx: Float32,
    kz: Float32,
    ky: Float32,
    kx: Float32,
    rot_cotangent: C2,
    gz: C2,
    gy: C2,
    gx: C2,
    shift_cotangent: C2,
    modulated: C2,
    p: FourierSliceParams,
):
    """Atomically add this line pixel's direction + 3D-shift grads.

    `k = s_x * u`, so `d(value)/du_a = g_a * s_x`; the direction grad is
    `(dz, dy, dx) * s_x` with `d_a = Re[cotangent * conj(g_a)]`. The 3D shift
    term ramps with the rotated coordinate `(kz, ky, kx)`, as in the slice kernel.
    """
    var db = 0 if p.bv_rot == 1 else i_bv
    var dbase = (db * p.bp + i_bp) * 3
    var dz = _redot(rot_cotangent, gz)
    var dy = _redot(rot_cotangent, gy)
    var dx = _redot(rot_cotangent, gx)
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](  # d/du_z
        grad_dir + dbase + 0, dz * sx
    )
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](  # d/du_y
        grad_dir + dbase + 1, dy * sx
    )
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](  # d/du_x
        grad_dir + dbase + 2, dx * sx
    )

    if p.has_shifts_3d != 0:
        var sb3 = 0 if p.bv_shift_3d == 1 else i_bv
        var s3 = (sb3 * p.bp + i_bp) * 3
        var scale3 = p.two_pi_over_sidelength()
        var p3z = _cmul(C2(0.0, scale3 * kz), modulated)
        var p3y = _cmul(C2(0.0, scale3 * ky), modulated)
        var p3x = _cmul(C2(0.0, scale3 * kx), modulated)
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](
            grad_shift_3d + s3 + 0, _redot(shift_cotangent, p3z)
        )
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](
            grad_shift_3d + s3 + 1, _redot(shift_cotangent, p3y)
        )
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](
            grad_shift_3d + s3 + 2, _redot(shift_cotangent, p3x)
        )


@always_inline
def _forward_line_pose_grad_pixel[
    interp: Int
](
    rec: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    grad_line: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift_3d: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Direction/3D-shift grads for the forward line projection (volume = rec).
    """
    var coord_x = Float32(x)
    if coord_x * coord_x > p.radius_cutoff_sq:
        return
    var db = 0 if p.bv_rot == 1 else i_bv
    var sx = coord_x * p.oversampling
    var k = _line_k(direction, (db * p.bp + i_bp) * 3, sx)
    var half = _rfft_half(p.sidelength)
    var rec_b = TileTensor(
        rec + i_bv * p.sidelength * p.sidelength * half * 2,
        row_major(p.sidelength, p.sidelength, half, 2),
    )
    var vg = _interp3d_with_grad[interp](rec_b, k[0], k[1], k[2], 0)
    var val = C2(vg[0], vg[1])
    var gz = C2(vg[2], vg[3])
    var gy = C2(vg[4], vg[5])
    var gx = C2(vg[6], vg[7])
    if p.has_shifts_3d != 0:
        _couple_shift3d(shifts_3d, i_bv, i_bp, val, p, gz, gy, gx)
    var line_half = p.proj_sidelength_half()
    var off = ((i_bv * p.bp + i_bp) * line_half + x) * 2
    var gp = C2(grad_line[off], grad_line[off + 1])
    var pf = _line_phase_factor(shifts_3d, i_bv, i_bp, k[0], k[1], k[2], p)
    # direction cotangent: interp grad paired with grad_line * conj(phase); the
    # shift term pairs grad_line against the forward value modulated by the phase.
    var gpc = _cmul(gp, C2(pf[0], -pf[1]))
    var modulated = _cmul(val, pf)
    _accumulate_line_pose_grads(
        grad_dir,
        grad_shift_3d,
        i_bv,
        i_bp,
        sx,
        k[0],
        k[1],
        k[2],
        gpc,
        gz,
        gy,
        gx,
        gp,
        modulated,
        p,
    )


@always_inline
def _backproject_line_pose_grad_pixel[
    interp: Int
](
    grad_rec: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    lines: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift_3d: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Direction/3D-shift grads for the line insertion (volume = grad_data_rec).
    """
    var coord_x = Float32(x)
    if coord_x * coord_x > p.radius_cutoff_sq:
        return
    var db = 0 if p.bv_rot == 1 else i_bv
    var sx = coord_x * p.oversampling
    var k = _line_k(direction, (db * p.bp + i_bp) * 3, sx)
    var half = _rfft_half(p.sidelength)
    var grad_rec_b = TileTensor(
        grad_rec + i_bv * p.sidelength * p.sidelength * half * 2,
        row_major(p.sidelength, p.sidelength, half, 2),
    )
    var vg = _interp3d_with_grad[interp](grad_rec_b, k[0], k[1], k[2], 1)
    var val = C2(vg[0], vg[1])
    var gz = C2(vg[2], vg[3])
    var gy = C2(vg[4], vg[5])
    var gx = C2(vg[6], vg[7])
    if p.has_shifts_3d != 0:
        _couple_shift3d(shifts_3d, i_bv, i_bp, val, p, gz, gy, gx)
    var line_half = p.proj_sidelength_half()
    var off = ((i_bv * p.bp + i_bp) * line_half + x) * 2
    var pv = C2(lines[off], lines[off + 1])
    var pf = _line_phase_factor(shifts_3d, i_bv, i_bp, k[0], k[1], k[2], p)
    # insertion applies the conjugate phase to the line value; both the direction
    # and shift terms pair that against the gathered grad_rec field.
    var pvc = _cmul(pv, C2(pf[0], -pf[1]))
    _accumulate_line_pose_grads(
        grad_dir,
        grad_shift_3d,
        i_bv,
        i_bp,
        sx,
        k[0],
        k[1],
        k[2],
        pvc,
        gz,
        gy,
        gx,
        pvc,
        val,
        p,
    )


@always_inline
def _weight_line_grad_pixel[
    interp: Int
](
    gwvol: Float32Ptr,
    direction: Float32Ptr,
    grad_weight: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    x: Int,
    p: FourierSliceParams,
):
    """grad w.r.t. one input line weight: gather grad_weight_rec with splat weights.
    """
    var coord_x = Float32(x)
    if coord_x * coord_x > p.radius_cutoff_sq:
        return
    var db = 0 if p.bv_rot == 1 else i_bv
    var sx = coord_x * p.oversampling
    var k = _line_k(direction, (db * p.bp + i_bp) * 3, sx)
    var kz = k[0]
    var ky = k[1]
    var kx = k[2]
    var kz_floor = floor(kz)
    var ky_floor = floor(ky)
    var kx_floor = floor(kx)
    var z0 = Int(kz_floor)
    var y0 = Int(ky_floor)
    var x0 = Int(kx_floor)
    var fz = kz - kz_floor
    var fy = ky - ky_floor
    var fx = kx - kx_floor
    var acc: Float32 = 0.0
    comptime if interp == CUBIC:
        for oz in range(-1, 3):
            var wz = _cubic_kernel(fz - Float32(oz))
            for oy in range(-1, 3):
                var wzy = wz * _cubic_kernel(fy - Float32(oy))
                for ox in range(-1, 3):
                    var w = wzy * _cubic_kernel(fx - Float32(ox))
                    acc += w * _gather_weight_grad(
                        gwvol,
                        i_bv,
                        p.sidelength,
                        z0 + oz,
                        y0 + oy,
                        x0 + ox,
                        p.friedel_double,
                    )
    else:
        var ifz = 1.0 - fz
        var ify = 1.0 - fy
        var ifx = 1.0 - fx
        var sl = p.sidelength
        var fd = p.friedel_double
        acc = (
            ifz
            * ify
            * ifx
            * _gather_weight_grad(gwvol, i_bv, sl, z0, y0, x0, fd)
            + ifz
            * ify
            * fx
            * _gather_weight_grad(gwvol, i_bv, sl, z0, y0, x0 + 1, fd)
            + ifz
            * fy
            * ifx
            * _gather_weight_grad(gwvol, i_bv, sl, z0, y0 + 1, x0, fd)
            + ifz
            * fy
            * fx
            * _gather_weight_grad(gwvol, i_bv, sl, z0, y0 + 1, x0 + 1, fd)
            + fz
            * ify
            * ifx
            * _gather_weight_grad(gwvol, i_bv, sl, z0 + 1, y0, x0, fd)
            + fz
            * ify
            * fx
            * _gather_weight_grad(gwvol, i_bv, sl, z0 + 1, y0, x0 + 1, fd)
            + fz
            * fy
            * ifx
            * _gather_weight_grad(gwvol, i_bv, sl, z0 + 1, y0 + 1, x0, fd)
            + fz
            * fy
            * fx
            * _gather_weight_grad(gwvol, i_bv, sl, z0 + 1, y0 + 1, x0 + 1, fd)
        )
    var line_half = p.proj_sidelength_half()
    grad_weight[(i_bv * p.bp + i_bp) * line_half + x] = acc
