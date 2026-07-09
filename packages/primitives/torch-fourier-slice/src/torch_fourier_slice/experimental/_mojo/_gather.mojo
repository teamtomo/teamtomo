"""Forward gather: sample one rfft voxel and interpolate (linear / cubic).

The volume is a per-volume 4D `TileTensor` `[d, h, w, 2]` (the trailing axis is
the complex (re, im) pair); the caller slices the batch axis and hands a single
volume's view in. Sizes come from the tensor itself (`rec.dim[...]()`), so the
cube edge / rfft width are no longer threaded as parameters.

Integer voxel indices are `(z, y, x)`; the continuous sample coordinate is
`(kz, ky, kx)`.
"""

from std.math import floor

from layout import TensorLayout, TileTensor

from _common import C2, _cubic_kernel, _load_c2


@always_inline
def _sample_rfft_3d[
    L: TensorLayout
](
    rec: TileTensor[DType.float32, L, MutAnyOrigin],
    z_in: Int,
    y_in: Int,
    x_in: Int,
) -> C2:
    """Sample a complex voxel with Friedel symmetry; clamp out-of-range indices.

    Negative kx (x < 0) is folded onto positive kx with conjugation; remaining
    axes are clamped to their valid range and wrapped into the rfft layout.
    """
    comptime assert rec.flat_rank == 4, "volume view must be 4D [d, h, w, 2]"
    var sidelength = Int(rec.dim[0]())
    var sidelength_half = Int(rec.dim[2]())
    var z = z_in
    var y = y_in
    var x = x_in
    var need_conj = False
    if x < 0:
        x = -x
        y = -y
        z = -z
        need_conj = True
    if x > sidelength_half - 1:
        x = sidelength_half - 1
    var hi = sidelength // 2
    var lo = -sidelength // 2 + 1
    if y > hi:
        y = hi
    elif y < lo:
        y = lo
    if z > hi:
        z = hi
    elif z < lo:
        z = lo
    if y < 0:
        y = sidelength + y
    if z < 0:
        z = sidelength + z
    if y > sidelength - 1:
        y = sidelength - 1
    if z > sidelength - 1:
        z = sidelength - 1
    var v = _load_c2(rec, z, y, x)
    return C2(v[0], -v[1]) if need_conj else v


@always_inline
def _sample_rfft_3d_drop[
    L: TensorLayout
](
    rec: TileTensor[DType.float32, L, MutAnyOrigin],
    z_in: Int,
    y_in: Int,
    x_in: Int,
) -> C2:
    """Sample with Friedel symmetry but *drop* (zero) out-of-range voxels.

    The read-side mirror of `_accumulate_3d`: where the splat discards a corner,
    this returns 0. Used by the backprojection gradient gather so it is the exact
    adjoint of the scatter (which drops, where the forward gather clamps).
    """
    comptime assert rec.flat_rank == 4, "volume view must be 4D [d, h, w, 2]"
    var sidelength = Int(rec.dim[0]())
    var sidelength_half = Int(rec.dim[2]())
    var z = z_in
    var y = y_in
    var x = x_in
    var need_conj = False
    if x < 0:
        x = -x
        y = -y
        z = -z
        need_conj = True
    if x >= sidelength_half:
        return C2(0.0, 0.0)
    var hi = sidelength // 2
    var lo = -sidelength // 2 + 1
    if y > hi or y < lo or z > hi or z < lo:
        return C2(0.0, 0.0)
    var y_eff = sidelength + y if y < 0 else y
    var z_eff = sidelength + z if z < 0 else z
    if y_eff >= sidelength or z_eff >= sidelength:
        return C2(0.0, 0.0)
    var v = _load_c2(rec, z_eff, y_eff, x)
    return C2(v[0], -v[1]) if need_conj else v


@always_inline
def _interp3d_linear[
    L: TensorLayout
](
    rec: TileTensor[DType.float32, L, MutAnyOrigin],
    kz: Float32,
    ky: Float32,
    kx: Float32,
) -> C2:
    """Trilinear interpolation of the rfft volume at the sample coordinate."""
    var kz_floor = floor(kz)
    var ky_floor = floor(ky)
    var kx_floor = floor(kx)
    var z = Int(kz_floor)
    var y = Int(ky_floor)
    var x = Int(kx_floor)
    var fz = kz - kz_floor
    var fy = ky - ky_floor
    var fx = kx - kx_floor
    var p000 = _sample_rfft_3d(rec, z, y, x)
    var p001 = _sample_rfft_3d(rec, z, y, x + 1)
    var p010 = _sample_rfft_3d(rec, z, y + 1, x)
    var p011 = _sample_rfft_3d(rec, z, y + 1, x + 1)
    var p100 = _sample_rfft_3d(rec, z + 1, y, x)
    var p101 = _sample_rfft_3d(rec, z + 1, y, x + 1)
    var p110 = _sample_rfft_3d(rec, z + 1, y + 1, x)
    var p111 = _sample_rfft_3d(rec, z + 1, y + 1, x + 1)
    var p00 = p000 + (p001 - p000) * fx
    var p01 = p010 + (p011 - p010) * fx
    var p10 = p100 + (p101 - p100) * fx
    var p11 = p110 + (p111 - p110) * fx
    var p0 = p00 + (p01 - p00) * fy
    var p1 = p10 + (p11 - p10) * fy
    return p0 + (p1 - p0) * fz


@always_inline
def _interp3d_cubic[
    L: TensorLayout
](
    rec: TileTensor[DType.float32, L, MutAnyOrigin],
    kz: Float32,
    ky: Float32,
    kx: Float32,
) -> C2:
    """Tricubic (Catmull-Rom) interpolation over the surrounding 4x4x4 voxels."""
    var kz_floor = floor(kz)
    var ky_floor = floor(ky)
    var kx_floor = floor(kx)
    var z = Int(kz_floor)
    var y = Int(ky_floor)
    var x = Int(kx_floor)
    var fz = kz - kz_floor
    var fy = ky - ky_floor
    var fx = kx - kx_floor
    var acc = C2(0.0, 0.0)
    for oz in range(-1, 3):
        var wz = _cubic_kernel(fz - Float32(oz))
        for oy in range(-1, 3):
            var wzy = wz * _cubic_kernel(fy - Float32(oy))
            for ox in range(-1, 3):
                var w = wzy * _cubic_kernel(fx - Float32(ox))
                var s = _sample_rfft_3d(rec, z + oz, y + oy, x + ox)
                acc = acc + s * w
    return acc


@always_inline
def _interp3d[
    L: TensorLayout
](
    rec: TileTensor[DType.float32, L, MutAnyOrigin],
    kz: Float32,
    ky: Float32,
    kx: Float32,
    interp: Int,
) -> C2:
    """Interpolate the rfft volume (interp: 0 = trilinear, 1 = tricubic)."""
    if interp == 1:
        return _interp3d_cubic(rec, kz, ky, kx)
    return _interp3d_linear(rec, kz, ky, kx)
