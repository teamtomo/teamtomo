"""Per-pixel backward ops for gradients w.r.t. rotations, shifts_2d and weights.

These complement the volume/projection gradients (which are the plain
scatter / forward-projection adjoints). For one output pixel they accumulate:

- a 3x3 rotation-matrix gradient, via the chain rule through the rotated sample
  coordinate using the analytical spatial gradient of the interpolated field;
- a 2-vector shift gradient, via the derivative of the phase ramp;
- (backprojection only) a real weight gradient, the adjoint of the weight splat.

Accumulators are per pose, so contributions are added atomically. `Re[a conj(b)]`
for two complex `(re, im)` pairs is the lane dot product `a[0]b[0] + a[1]b[1]`.
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
    _ewald_sz,
    _fourier_coord,
    _rfft_half,
    _rotated_coord,
    _shift_phase,
)
from _gather_grad import _interp3d_with_grad


@always_inline
def _redot(a: C2, b: C2) -> Float32:
    """Re[a * conj(b)] for complex values stored as (re, im)."""
    return a[0] * b[0] + a[1] * b[1]


@always_inline
def _phase_factor(
    shifts_2d: Float32Ptr,
    shifts_3d: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    coord_y: Float32,
    coord_x: Float32,
    kz: Float32,
    ky: Float32,
    kx: Float32,
    p: FourierSliceParams,
) -> C2:
    """Combined shift phase factor exp(i*phase) for this pixel (identity if none)."""
    if p.has_shifts_2d == 0 and p.has_shifts_3d == 0:
        return C2(1.0, 0.0)
    var phase = _shift_phase(
        p, shifts_2d, shifts_3d, i_bv, i_bp, coord_y, coord_x, kz, ky, kx
    )
    return C2(cos(phase), sin(phase))


@always_inline
def _accumulate_pose_grads(
    grad_rot: Float32Ptr,
    grad_shift: Float32Ptr,
    grad_shift_3d: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    coord_y: Float32,
    coord_x: Float32,
    sx: Float32,
    sy: Float32,
    sz: Float32,
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
    """Atomically add this pixel's rotation (3x3) and shift (2D + 3D) contributions.

    `rot_cotangent` is the cotangent paired with the interp spatial gradient
    (already augmented for the 3D-shift coupling by the caller);
    `shift_cotangent` / `modulated` define the phase-derivative shift term
    (shift_grad += Re[shift_cotangent * conj(i*K*coord*modulated)]). The 2D shift
    ramps with the image coords (coord_y, coord_x); the 3D shift ramps with the
    rotated sample coordinate (kz, ky, kx).
    """
    # rows of R map to (kz, ky, kx) outputs; columns to (sz, sy, sx) inputs.
    var rb = 0 if p.bv_rot == 1 else i_bv
    var rbase = (rb * p.bp + i_bp) * 9
    var dx = _redot(rot_cotangent, gx)
    var dy = _redot(rot_cotangent, gy)
    var dz = _redot(rot_cotangent, gz)
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](grad_rot + rbase + 1, dz * sy)
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](grad_rot + rbase + 2, dz * sx)
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](grad_rot + rbase + 4, dy * sy)
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](grad_rot + rbase + 5, dy * sx)
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](grad_rot + rbase + 7, dx * sy)
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](grad_rot + rbase + 8, dx * sx)
    # z-input column: only non-zero when Ewald curvature bends the slice (sz != 0).
    if p.ewald_curvature != 0.0:
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](grad_rot + rbase + 0, dz * sz)
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](grad_rot + rbase + 3, dy * sz)
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](grad_rot + rbase + 6, dx * sz)

    if p.has_shifts_2d != 0:
        var sb = 0 if p.bv_shift_2d == 1 else i_bv
        var sbase = (sb * p.bp + i_bp) * 2
        var scale = p.two_pi_over_proj_sidelength()
        var pgr = _cmul(C2(0.0, scale * coord_y), modulated)
        var pgc = _cmul(C2(0.0, scale * coord_x), modulated)
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](
            grad_shift + sbase + 0, _redot(shift_cotangent, pgr)
        )
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](
            grad_shift + sbase + 1, _redot(shift_cotangent, pgc)
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
def _couple_shift3d(
    shifts_3d: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    val: C2,
    p: FourierSliceParams,
    mut gz: C2,
    mut gy: C2,
    mut gx: C2,
):
    """Augment the spatial grads with the 3D-shift phase derivative.

    The sampled field carries `exp(i*scale3d*(k . t3d))` with `k` the rotated
    coordinate, so `d/dk_a` adds `i*scale3d*t3d_a*val` to the raw spatial grad
    `g_a`. This makes the rotation Jacobian see the 3D shift's `R`-dependence.
    """
    var sb3 = 0 if p.bv_shift_3d == 1 else i_bv
    var s3 = (sb3 * p.bp + i_bp) * 3
    var c = p.two_pi_over_sidelength()
    var wz = c * shifts_3d[s3]
    var wy = c * shifts_3d[s3 + 1]
    var wx = c * shifts_3d[s3 + 2]
    gz = C2(gz[0] - wz * val[1], gz[1] + wz * val[0])
    gy = C2(gy[0] - wy * val[1], gy[1] + wy * val[0])
    gx = C2(gx[0] - wx * val[1], gx[1] + wx * val[0])


@always_inline
def _forward_pose_grad_pixel[
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
    i_bv: Int,
    i_bp: Int,
    y: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Rotation/shift grads for the forward projection (volume = rec)."""
    var coord_y = _fourier_coord(y, p.proj_sidelength)
    var coord_x = Float32(x)
    if coord_y * coord_y + coord_x * coord_x > p.radius_cutoff_sq:
        return
    var sx = coord_x * p.oversampling
    var sy = coord_y * p.oversampling
    var sz = _ewald_sz(p, sx, sy)
    var rb = 0 if p.bv_rot == 1 else i_bv
    var k = _rotated_coord(rot, (rb * p.bp + i_bp) * 9, sx, sy, sz)
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
    # 3D shift couples into the rotation grad: the sampled field carries the
    # phase exp(i*scale3d*k.t3d), so augment each spatial grad by d/dk of it.
    if p.has_shifts_3d != 0:
        _couple_shift3d(shifts_3d, i_bv, i_bp, val, p, gz, gy, gx)
    var off = (
        (
            ((i_bv * p.bp + i_bp) * p.proj_sidelength + y) * p.proj_sidelength_half()
            + x
        )
    ) * 2
    var gp = C2(grad_proj[off], grad_proj[off + 1])
    var pf = _phase_factor(
        shifts_2d, shifts_3d, i_bv, i_bp, coord_y, coord_x, k[0], k[1], k[2], p
    )
    # rotation cotangent: grad of interp = grad_proj * conj(phase); shift uses the
    # raw grad_proj against the forward value modulated by the phase.
    var gpc = _cmul(gp, C2(pf[0], -pf[1]))
    var modulated = _cmul(val, pf)
    _accumulate_pose_grads(
        grad_rot,
        grad_shift,
        grad_shift_3d,
        i_bv,
        i_bp,
        coord_y,
        coord_x,
        sx,
        sy,
        sz,
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
def _backproject_pose_grad_pixel[
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
    i_bv: Int,
    i_bp: Int,
    y: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Rotation/shift grads for the backprojection (volume = grad_data_rec)."""
    var coord_y = _fourier_coord(y, p.proj_sidelength)
    var coord_x = Float32(x)
    if coord_y * coord_y + coord_x * coord_x > p.radius_cutoff_sq:
        return
    # the scatter skipped the redundant x=0 half, so these pixels carry no grad.
    if x == 0 and y >= p.proj_sidelength // 2:
        return
    var sx = coord_x * p.oversampling
    var sy = coord_y * p.oversampling
    var sz = _ewald_sz(p, sx, sy)
    var rb = 0 if p.bv_rot == 1 else i_bv
    var k = _rotated_coord(rot, (rb * p.bp + i_bp) * 9, sx, sy, sz)
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
    # 3D shift couples into the rotation grad (same augmentation as the forward).
    if p.has_shifts_3d != 0:
        _couple_shift3d(shifts_3d, i_bv, i_bp, val, p, gz, gy, gx)
    var off = (
        (
            ((i_bv * p.bp + i_bp) * p.proj_sidelength + y) * p.proj_sidelength_half()
            + x
        )
    ) * 2
    var pv = C2(proj[off], proj[off + 1])
    var pf = _phase_factor(
        shifts_2d, shifts_3d, i_bv, i_bp, coord_y, coord_x, k[0], k[1], k[2], p
    )
    # backprojection applies the conjugate phase to the projection value; both the
    # rotation and shift terms pair that against the gathered grad_rec field.
    var pvc = _cmul(pv, C2(pf[0], -pf[1]))
    _accumulate_pose_grads(
        grad_rot,
        grad_shift,
        grad_shift_3d,
        i_bv,
        i_bp,
        coord_y,
        coord_x,
        sx,
        sy,
        sz,
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


# ---------------------------------------------------------------------------
# Weight gradient (backprojection): adjoint of the real weight splat
# ---------------------------------------------------------------------------


@always_inline
def _gather_weight_grad(
    gwvol: Float32Ptr,
    i_bv: Int,
    sidelength: Int,
    z_in: Int,
    y_in: Int,
    x_in: Int,
    friedel_double: Int,
) -> Float32:
    """Read the (real) weight-volume grad at the cell(s) the weight splat wrote.

    Transpose of `_accumulate_weight`: same index/Friedel logic, summing reads.
    """
    var sidelength_half = _rfft_half(sidelength)
    var z = z_in
    var y = y_in
    var x = x_in
    if x < 0:
        x = -x
        y = -y
        z = -z
    if x >= sidelength_half:
        return 0.0
    var hi = sidelength // 2
    var lo = -sidelength // 2 + 1
    if y > hi or y < lo or z > hi or z < lo:
        return 0.0
    var y_eff = sidelength + y if y < 0 else y
    var z_eff = sidelength + z if z < 0 else z
    if y_eff >= sidelength or z_eff >= sidelength:
        return 0.0
    var off = ((i_bv * sidelength + z_eff) * sidelength + y_eff) * sidelength_half + x
    var acc = gwvol[off]
    if friedel_double != 0 and x == 0:
        var z_eff2 = sidelength - z_eff if z_eff != 0 else 0
        var y_eff2 = sidelength - y_eff if y_eff != 0 else 0
        if z_eff2 >= sidelength or y_eff2 >= sidelength:
            return acc
        if y_eff2 == y_eff and z_eff2 == z_eff:
            return acc
        var off2 = (
            (i_bv * sidelength + z_eff2) * sidelength + y_eff2
        ) * sidelength_half + x
        acc = acc + gwvol[off2]
    return acc


@always_inline
def _g(gwvol: Float32Ptr, i_bv: Int, p: FourierSliceParams, z: Int, y: Int, x: Int) -> Float32:
    return _gather_weight_grad(gwvol, i_bv, p.sidelength, z, y, x, p.friedel_double)


@always_inline
def _weight_grad_pixel[
    interp: Int
](
    gwvol: Float32Ptr,
    rot: Float32Ptr,
    grad_weight: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    y: Int,
    x: Int,
    p: FourierSliceParams,
):
    """grad w.r.t. one input weight: gather grad_weight_rec with the splat weights."""
    var coord_y = _fourier_coord(y, p.proj_sidelength)
    var coord_x = Float32(x)
    if coord_y * coord_y + coord_x * coord_x > p.radius_cutoff_sq:
        return
    if x == 0 and y >= p.proj_sidelength // 2:
        return
    var sx = coord_x * p.oversampling
    var sy = coord_y * p.oversampling
    var rb = 0 if p.bv_rot == 1 else i_bv
    var k = _rotated_coord(rot, (rb * p.bp + i_bp) * 9, sx, sy, _ewald_sz(p, sx, sy))
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
        acc = (
            ifz * ify * ifx * _g(gwvol, i_bv, p, z0, y0, x0)
            + ifz * ify * fx * _g(gwvol, i_bv, p, z0, y0, x0 + 1)
            + ifz * fy * ifx * _g(gwvol, i_bv, p, z0, y0 + 1, x0)
            + ifz * fy * fx * _g(gwvol, i_bv, p, z0, y0 + 1, x0 + 1)
            + fz * ify * ifx * _g(gwvol, i_bv, p, z0 + 1, y0, x0)
            + fz * ify * fx * _g(gwvol, i_bv, p, z0 + 1, y0, x0 + 1)
            + fz * fy * ifx * _g(gwvol, i_bv, p, z0 + 1, y0 + 1, x0)
            + fz * fy * fx * _g(gwvol, i_bv, p, z0 + 1, y0 + 1, x0 + 1)
        )
    var off = (
        (i_bv * p.bp + i_bp) * p.proj_sidelength + y
    ) * p.proj_sidelength_half() + x
    grad_weight[off] = acc
