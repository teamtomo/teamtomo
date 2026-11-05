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

from _common import (
    C2,
    CUBIC,
    _cubic_kernel,
    _in_nyquist_cube,
    _load_c2,
    _load_c2_line,
    _rfft_voxel,
)


@always_inline
def _sample_rfft_3d[
    L: TensorLayout
](
    rec: TileTensor[DType.float32, L, MutAnyOrigin],
    z_in: Int,
    y_in: Int,
    x_in: Int,
) -> C2:
    """Read the DFT value at an integer frequency from the rfft volume.

    Frequencies past the stored grid are read periodically and negative kx is
    folded onto the stored half with conjugation (`_rfft_voxel`), so a stencil
    reaching past the edge reads the true neighbour values. The read-side mirror
    of `_accumulate_3d`, so every gather is the exact adjoint of the scatter.
    """
    comptime assert rec.flat_rank == 4, "volume view must be 4D [d, h, w, 2]"
    var v = _rfft_voxel(z_in, y_in, x_in, Int(rec.dim[0]()))
    var val = _load_c2(rec, v.z, v.y, v.x)
    return C2(val[0], -val[1]) if v.conj else val


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
    """Tricubic (Catmull-Rom) interpolation over the surrounding 4x4x4 voxels.
    """
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
    L: TensorLayout, //, interp: Int
](
    rec: TileTensor[DType.float32, L, MutAnyOrigin],
    kz: Float32,
    ky: Float32,
    kx: Float32,
) -> C2:
    """Interpolate the rfft volume (comptime interp: LINEAR = trilinear, CUBIC = tricubic).

    Points outside the Nyquist cube read as 0.
    """
    if not _in_nyquist_cube(kz, ky, kx, Int(rec.dim[0]())):
        return C2(0.0, 0.0)
    comptime if interp == CUBIC:
        return _interp3d_cubic(rec, kz, ky, kx)
    return _interp3d_linear(rec, kz, ky, kx)


# ===========================================================================
# 2D image gather (sample a 2D rfft image; used by the 2D->1D line kernels)
#
# The image is a per-image 3D `TileTensor` `[h, w, 2]` (rfft, DC at origin).
# Integer pixel indices are `(y, x)`; the continuous sample coord is `(ky, kx)`.
# ===========================================================================


@always_inline
def _sample_rfft_2d[
    L: TensorLayout
](img: TileTensor[DType.float32, L, MutAnyOrigin], y_in: Int, x_in: Int,) -> C2:
    """Read the DFT value at an integer frequency from the rfft image.

    The 2D analogue of `_sample_rfft_3d`: the exact read-side mirror of the
    2D scatter, so gather and scatter are exact adjoints.
    """
    comptime assert img.flat_rank == 3, "image view must be 3D [h, w, 2]"
    var v = _rfft_voxel(0, y_in, x_in, Int(img.dim[0]()))
    var val = _load_c2_line(img, v.y, v.x)
    return C2(val[0], -val[1]) if v.conj else val


@always_inline
def _interp2d_linear[
    L: TensorLayout
](
    img: TileTensor[DType.float32, L, MutAnyOrigin],
    ky: Float32,
    kx: Float32,
) -> C2:
    """Bilinear interpolation of the rfft image at the sample coordinate."""
    var ky_floor = floor(ky)
    var kx_floor = floor(kx)
    var y = Int(ky_floor)
    var x = Int(kx_floor)
    var fy = ky - ky_floor
    var fx = kx - kx_floor
    var p00 = _sample_rfft_2d(img, y, x)
    var p01 = _sample_rfft_2d(img, y, x + 1)
    var p10 = _sample_rfft_2d(img, y + 1, x)
    var p11 = _sample_rfft_2d(img, y + 1, x + 1)
    var p0 = p00 + (p01 - p00) * fx
    var p1 = p10 + (p11 - p10) * fx
    return p0 + (p1 - p0) * fy


@always_inline
def _interp2d_cubic[
    L: TensorLayout
](
    img: TileTensor[DType.float32, L, MutAnyOrigin],
    ky: Float32,
    kx: Float32,
) -> C2:
    """Bicubic (Catmull-Rom) interpolation over the surrounding 4x4 pixels."""
    var ky_floor = floor(ky)
    var kx_floor = floor(kx)
    var y = Int(ky_floor)
    var x = Int(kx_floor)
    var fy = ky - ky_floor
    var fx = kx - kx_floor
    var acc = C2(0.0, 0.0)
    for oy in range(-1, 3):
        var wy = _cubic_kernel(fy - Float32(oy))
        for ox in range(-1, 3):
            var w = wy * _cubic_kernel(fx - Float32(ox))
            acc = acc + _sample_rfft_2d(img, y + oy, x + ox) * w
    return acc


@always_inline
def _interp2d[
    L: TensorLayout, //, interp: Int
](
    img: TileTensor[DType.float32, L, MutAnyOrigin],
    ky: Float32,
    kx: Float32,
) -> C2:
    """Interpolate the rfft image (comptime interp: LINEAR = bilinear, CUBIC = bicubic).

    Points outside the Nyquist square read as 0.
    """
    if not _in_nyquist_cube(0.0, ky, kx, Int(img.dim[0]())):
        return C2(0.0, 0.0)
    comptime if interp == CUBIC:
        return _interp2d_cubic(img, ky, kx)
    return _interp2d_linear(img, ky, kx)
