"""Per-pixel backward ops for the 2D->1D central-line pose / weight gradients.

The 2D analogue of `_line_grad.mojo` (which is itself the line analogue of
`_pose_grad.mojo`). A 2D line samples along `k = s_x * u` with `u = (u_y, u_x)`,
so the pose gradient is the gradient w.r.t. that 2-vector: `d(value)/du_a =
g_a * s_x` summed over the line. An optional yx image shift adds a phase ramp,
handled exactly as the 3D line's `shifts_3d` (coupling into the direction grad
plus its own gradient).

Reuses the 2D interpolation-with-spatial-gradient (`_interp2d_with_grad`) and the
`_redot` helper; only the 1D line geometry and I/O differ.
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
    _line2d_shift_phase,
    _line_k_2d,
    _rfft_half,
)
from _gather_grad import _interp2d_with_grad
from _pose_grad import _redot


@always_inline
def _line2d_phase_factor(
    shifts_2d: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    ky: Float32,
    kx: Float32,
    p: FourierSliceParams,
) -> C2:
    """2D-shift phase factor exp(i*phase) for this line pixel (identity if none).
    """
    if p.has_shifts_2d == 0:
        return C2(1.0, 0.0)
    var phase = _line2d_shift_phase(p, shifts_2d, i_bv, i_bp, ky, kx)
    return C2(cos(phase), sin(phase))


@always_inline
def _couple_shift2d(
    shifts_2d: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    val: C2,
    p: FourierSliceParams,
    mut gy: C2,
    mut gx: C2,
):
    """Augment the spatial grads with the 2D-shift phase derivative (d/dk of exp).
    """
    var sb = 0 if p.bv_shift_2d == 1 else i_bv
    var s2 = (sb * p.bp + i_bp) * 2
    var c = p.two_pi_over_sidelength()
    var wy = c * shifts_2d[s2]
    var wx = c * shifts_2d[s2 + 1]
    gy = C2(gy[0] - wy * val[1], gy[1] + wy * val[0])
    gx = C2(gx[0] - wx * val[1], gx[1] + wx * val[0])


@always_inline
def _accumulate_line2d_pose_grads(
    grad_dir: Float32Ptr,
    grad_shift: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    sx: Float32,
    ky: Float32,
    kx: Float32,
    rot_cotangent: C2,
    gy: C2,
    gx: C2,
    shift_cotangent: C2,
    modulated: C2,
    p: FourierSliceParams,
):
    """Atomically add this line pixel's direction grad `(dy, dx)*s_x` + 2D-shift grad.

    `k = s_x * u`, so `d(value)/du_a = g_a * s_x`; `d_a = Re[cotangent*conj(g_a)]`.
    The shift term ramps with the sample coordinate `(ky, kx)`.
    """
    var db = 0 if p.bv_rot == 1 else i_bv
    var dbase = (db * p.bp + i_bp) * 2
    var dy = _redot(rot_cotangent, gy)
    var dx = _redot(rot_cotangent, gx)
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](  # d/du_y
        grad_dir + dbase + 0, dy * sx
    )
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](  # d/du_x
        grad_dir + dbase + 1, dx * sx
    )

    if p.has_shifts_2d != 0:
        var sb = 0 if p.bv_shift_2d == 1 else i_bv
        var s2 = (sb * p.bp + i_bp) * 2
        var scale = p.two_pi_over_sidelength()
        var p2y = _cmul(C2(0.0, scale * ky), modulated)
        var p2x = _cmul(C2(0.0, scale * kx), modulated)
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](
            grad_shift + s2 + 0, _redot(shift_cotangent, p2y)
        )
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](
            grad_shift + s2 + 1, _redot(shift_cotangent, p2x)
        )


@always_inline
def _forward_line2d_pose_grad_pixel[
    interp: Int
](
    img: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    grad_line: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Direction/2D-shift grad for the forward 2D line projection (image = img).
    """
    var coord_x = Float32(x)
    if coord_x * coord_x > p.radius_cutoff_sq:
        return
    var db = 0 if p.bv_rot == 1 else i_bv
    var sx = coord_x * p.oversampling
    var k = _line_k_2d(direction, (db * p.bp + i_bp) * 2, sx)
    var half = _rfft_half(p.sidelength)
    var img_b = TileTensor(
        img + i_bv * p.sidelength * half * 2,
        row_major(p.sidelength, half, 2),
    )
    var vg = _interp2d_with_grad[interp](img_b, k[0], k[1], 0)
    var val = C2(vg[0], vg[1])
    var gy = C2(vg[2], vg[3])
    var gx = C2(vg[4], vg[5])
    if p.has_shifts_2d != 0:
        _couple_shift2d(shifts_2d, i_bv, i_bp, val, p, gy, gx)
    var line_half = p.proj_sidelength_half()
    var off = ((i_bv * p.bp + i_bp) * line_half + x) * 2
    var gp = C2(grad_line[off], grad_line[off + 1])
    var pf = _line2d_phase_factor(shifts_2d, i_bv, i_bp, k[0], k[1], p)
    var gpc = _cmul(gp, C2(pf[0], -pf[1]))
    var modulated = _cmul(val, pf)
    _accumulate_line2d_pose_grads(
        grad_dir,
        grad_shift,
        i_bv,
        i_bp,
        sx,
        k[0],
        k[1],
        gpc,
        gy,
        gx,
        gp,
        modulated,
        p,
    )


@always_inline
def _backproject_line2d_pose_grad_pixel[
    interp: Int
](
    grad_img: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    lines: Float32Ptr,
    grad_dir: Float32Ptr,
    grad_shift: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Direction/2D-shift grad for the 2D line insertion (image = grad_data_img).
    """
    var coord_x = Float32(x)
    if coord_x * coord_x > p.radius_cutoff_sq:
        return
    var db = 0 if p.bv_rot == 1 else i_bv
    var sx = coord_x * p.oversampling
    var k = _line_k_2d(direction, (db * p.bp + i_bp) * 2, sx)
    var half = _rfft_half(p.sidelength)
    var grad_img_b = TileTensor(
        grad_img + i_bv * p.sidelength * half * 2,
        row_major(p.sidelength, half, 2),
    )
    var vg = _interp2d_with_grad[interp](grad_img_b, k[0], k[1], 1)
    var val = C2(vg[0], vg[1])
    var gy = C2(vg[2], vg[3])
    var gx = C2(vg[4], vg[5])
    if p.has_shifts_2d != 0:
        _couple_shift2d(shifts_2d, i_bv, i_bp, val, p, gy, gx)
    var line_half = p.proj_sidelength_half()
    var off = ((i_bv * p.bp + i_bp) * line_half + x) * 2
    var pv = C2(lines[off], lines[off + 1])
    var pf = _line2d_phase_factor(shifts_2d, i_bv, i_bp, k[0], k[1], p)
    var pvc = _cmul(pv, C2(pf[0], -pf[1]))
    _accumulate_line2d_pose_grads(
        grad_dir,
        grad_shift,
        i_bv,
        i_bp,
        sx,
        k[0],
        k[1],
        pvc,
        gy,
        gx,
        pvc,
        val,
        p,
    )


@always_inline
def _gather_weight_grad_2d(
    gwimg: Float32Ptr,
    i_bv: Int,
    sidelength: Int,
    y_in: Int,
    x_in: Int,
    friedel_double: Int,
) -> Float32:
    """Read the (real) 2D weight-image grad at the cell(s) the weight splat wrote.

    Transpose of `_accumulate_weight_2d`: same index/Friedel logic, summing reads.
    """
    var sidelength_half = _rfft_half(sidelength)
    var y = y_in
    var x = x_in
    if x < 0:
        x = -x
        y = -y
    if x >= sidelength_half:
        return 0.0
    var hi = sidelength // 2
    var lo = -sidelength // 2 + 1
    if y > hi or y < lo:
        return 0.0
    var y_eff = sidelength + y if y < 0 else y
    if y_eff >= sidelength:
        return 0.0
    var acc = gwimg[(i_bv * sidelength + y_eff) * sidelength_half + x]
    if friedel_double != 0 and x == 0:
        var y_eff2 = sidelength - y_eff if y_eff != 0 else 0
        if y_eff2 >= sidelength:
            return acc
        if y_eff2 == y_eff:
            return acc
        acc = acc + gwimg[(i_bv * sidelength + y_eff2) * sidelength_half + x]
    return acc


@always_inline
def _weight_line2d_grad_pixel[
    interp: Int
](
    gwimg: Float32Ptr,
    direction: Float32Ptr,
    grad_weight: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    x: Int,
    p: FourierSliceParams,
):
    """grad w.r.t. one input line weight: gather grad_weight_img with splat weights.
    """
    var coord_x = Float32(x)
    if coord_x * coord_x > p.radius_cutoff_sq:
        return
    var db = 0 if p.bv_rot == 1 else i_bv
    var sx = coord_x * p.oversampling
    var k = _line_k_2d(direction, (db * p.bp + i_bp) * 2, sx)
    var ky = k[0]
    var kx = k[1]
    var ky_floor = floor(ky)
    var kx_floor = floor(kx)
    var y0 = Int(ky_floor)
    var x0 = Int(kx_floor)
    var fy = ky - ky_floor
    var fx = kx - kx_floor
    var acc: Float32 = 0.0
    var sl = p.sidelength
    var fd = p.friedel_double
    comptime if interp == CUBIC:
        for oy in range(-1, 3):
            var wy = _cubic_kernel(fy - Float32(oy))
            for ox in range(-1, 3):
                var w = wy * _cubic_kernel(fx - Float32(ox))
                acc += w * _gather_weight_grad_2d(
                    gwimg, i_bv, sl, y0 + oy, x0 + ox, fd
                )
    else:
        var ify = 1.0 - fy
        var ifx = 1.0 - fx
        acc = (
            ify * ifx * _gather_weight_grad_2d(gwimg, i_bv, sl, y0, x0, fd)
            + ify * fx * _gather_weight_grad_2d(gwimg, i_bv, sl, y0, x0 + 1, fd)
            + fy * ifx * _gather_weight_grad_2d(gwimg, i_bv, sl, y0 + 1, x0, fd)
            + fy
            * fx
            * _gather_weight_grad_2d(gwimg, i_bv, sl, y0 + 1, x0 + 1, fd)
        )
    var line_half = p.proj_sidelength_half()
    grad_weight[(i_bv * p.bp + i_bp) * line_half + x] = acc
