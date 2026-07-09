"""Scatter: atomic accumulation of a value into the rfft volume (linear / cubic).

The volume / weight volume are per-volume `TileTensor` views (`[d, h, w, 2]` and
`[d, h, w]`); accumulation is addressed through the layout via `_atomic_add_at`.
Integer voxel indices are `(z, y, x)`; the continuous sample coordinate is
`(kz, ky, kx)`.
"""

from std.math import floor

from layout import Coord, TensorLayout, TileTensor

from _common import FourierSliceParams, _atomic_add_at, _cubic_kernel


@always_inline
def _accumulate_3d[
    L: TensorLayout
](
    vol: TileTensor[DType.float32, L, MutAnyOrigin],
    z_in: Int,
    y_in: Int,
    x_in: Int,
    vre: Float32,
    vim: Float32,
    friedel_double: Int,
):
    """Atomically add a complex value into the rfft volume with Friedel symmetry."""
    comptime assert vol.flat_rank == 4, "volume view must be 4D [d, h, w, 2]"
    var sidelength = Int(vol.dim[0]())
    var sidelength_half = Int(vol.dim[2]())
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
        return
    var hi = sidelength // 2
    var lo = -sidelength // 2 + 1
    if y > hi or y < lo or z > hi or z < lo:
        return
    var y_eff = sidelength + y if y < 0 else y
    var z_eff = sidelength + z if z < 0 else z
    if y_eff >= sidelength or z_eff >= sidelength:
        return
    var fim = -vim if need_conj else vim
    _atomic_add_at(vol, Coord(z_eff, y_eff, x, 0), vre)
    _atomic_add_at(vol, Coord(z_eff, y_eff, x, 1), fim)
    # On the x=0 plane, also insert the Friedel-symmetric conjugate counterpart.
    if friedel_double != 0 and x == 0:
        var z_eff2 = sidelength - z_eff if z_eff != 0 else 0
        var y_eff2 = sidelength - y_eff if y_eff != 0 else 0
        if z_eff2 >= sidelength or y_eff2 >= sidelength:
            return
        if y_eff2 == y_eff and z_eff2 == z_eff:
            return
        _atomic_add_at(vol, Coord(z_eff2, y_eff2, x, 0), vre)
        _atomic_add_at(vol, Coord(z_eff2, y_eff2, x, 1), -fim)


@always_inline
def _accumulate_weight[
    L: TensorLayout
](
    wvol: TileTensor[DType.float32, L, MutAnyOrigin],
    z_in: Int,
    y_in: Int,
    x_in: Int,
    w: Float32,
    friedel_double: Int,
):
    """Atomically add a real weight into the (real) weight volume with symmetry."""
    comptime assert wvol.flat_rank == 3, "weight view must be 3D [d, h, w]"
    var sidelength = Int(wvol.dim[0]())
    var sidelength_half = Int(wvol.dim[2]())
    var z = z_in
    var y = y_in
    var x = x_in
    if x < 0:
        x = -x
        y = -y
        z = -z
    if x >= sidelength_half:
        return
    var hi = sidelength // 2
    var lo = -sidelength // 2 + 1
    if y > hi or y < lo or z > hi or z < lo:
        return
    var y_eff = sidelength + y if y < 0 else y
    var z_eff = sidelength + z if z < 0 else z
    if y_eff >= sidelength or z_eff >= sidelength:
        return
    _atomic_add_at(wvol, Coord(z_eff, y_eff, x), w)
    if friedel_double != 0 and x == 0:
        var z_eff2 = sidelength - z_eff if z_eff != 0 else 0
        var y_eff2 = sidelength - y_eff if y_eff != 0 else 0
        if z_eff2 >= sidelength or y_eff2 >= sidelength:
            return
        if y_eff2 == y_eff and z_eff2 == z_eff:
            return
        _atomic_add_at(wvol, Coord(z_eff2, y_eff2, x), w)


@always_inline
def _splat_one[
    Lv: TensorLayout, Lw: TensorLayout
](
    vol: TileTensor[DType.float32, Lv, MutAnyOrigin],
    wvol: TileTensor[DType.float32, Lw, MutAnyOrigin],
    p: FourierSliceParams,
    z: Int,
    y: Int,
    x: Int,
    w: Float32,
    vre: Float32,
    vim: Float32,
    wval: Float32,
):
    """Accumulate a single interpolation corner (data + optional weight)."""
    _accumulate_3d(vol, z, y, x, vre * w, vim * w, p.friedel_double)
    if p.has_weights != 0:
        _accumulate_weight(wvol, z, y, x, wval * w, p.friedel_double)


@always_inline
def _splat_linear[
    Lv: TensorLayout, Lw: TensorLayout
](
    vol: TileTensor[DType.float32, Lv, MutAnyOrigin],
    wvol: TileTensor[DType.float32, Lw, MutAnyOrigin],
    p: FourierSliceParams,
    kz: Float32,
    ky: Float32,
    kx: Float32,
    vre: Float32,
    vim: Float32,
    wval: Float32,
):
    """Trilinearly splat a complex value to the 8 surrounding voxels."""
    var kz_floor = floor(kz)
    var ky_floor = floor(ky)
    var kx_floor = floor(kx)
    var z = Int(kz_floor)
    var y = Int(ky_floor)
    var x = Int(kx_floor)
    var fz = kz - kz_floor
    var fy = ky - ky_floor
    var fx = kx - kx_floor
    var ifz = 1.0 - fz
    var ify = 1.0 - fy
    var ifx = 1.0 - fx
    _splat_one(vol, wvol, p, z, y, x, ifz * ify * ifx, vre, vim, wval)
    _splat_one(vol, wvol, p, z, y, x + 1, ifz * ify * fx, vre, vim, wval)
    _splat_one(vol, wvol, p, z, y + 1, x, ifz * fy * ifx, vre, vim, wval)
    _splat_one(vol, wvol, p, z, y + 1, x + 1, ifz * fy * fx, vre, vim, wval)
    _splat_one(vol, wvol, p, z + 1, y, x, fz * ify * ifx, vre, vim, wval)
    _splat_one(vol, wvol, p, z + 1, y, x + 1, fz * ify * fx, vre, vim, wval)
    _splat_one(vol, wvol, p, z + 1, y + 1, x, fz * fy * ifx, vre, vim, wval)
    _splat_one(vol, wvol, p, z + 1, y + 1, x + 1, fz * fy * fx, vre, vim, wval)


@always_inline
def _splat_cubic[
    Lv: TensorLayout, Lw: TensorLayout
](
    vol: TileTensor[DType.float32, Lv, MutAnyOrigin],
    wvol: TileTensor[DType.float32, Lw, MutAnyOrigin],
    p: FourierSliceParams,
    kz: Float32,
    ky: Float32,
    kx: Float32,
    vre: Float32,
    vim: Float32,
    wval: Float32,
):
    """Tricubically splat a complex value to the surrounding 4x4x4 voxels."""
    var kz_floor = floor(kz)
    var ky_floor = floor(ky)
    var kx_floor = floor(kx)
    var z = Int(kz_floor)
    var y = Int(ky_floor)
    var x = Int(kx_floor)
    var fz = kz - kz_floor
    var fy = ky - ky_floor
    var fx = kx - kx_floor
    for oz in range(-1, 3):
        var wz = _cubic_kernel(fz - Float32(oz))
        for oy in range(-1, 3):
            var wzy = wz * _cubic_kernel(fy - Float32(oy))
            for ox in range(-1, 3):
                var w = wzy * _cubic_kernel(fx - Float32(ox))
                _splat_one(vol, wvol, p, z + oz, y + oy, x + ox, w, vre, vim, wval)


@always_inline
def _splat[
    Lv: TensorLayout, Lw: TensorLayout
](
    vol: TileTensor[DType.float32, Lv, MutAnyOrigin],
    wvol: TileTensor[DType.float32, Lw, MutAnyOrigin],
    p: FourierSliceParams,
    kz: Float32,
    ky: Float32,
    kx: Float32,
    vre: Float32,
    vim: Float32,
    wval: Float32,
):
    """Splat a complex value (p.interp: 0 = trilinear, 1 = tricubic)."""
    if p.interp == 1:
        _splat_cubic(vol, wvol, p, kz, ky, kx, vre, vim, wval)
    else:
        _splat_linear(vol, wvol, p, kz, ky, kx, vre, vim, wval)
