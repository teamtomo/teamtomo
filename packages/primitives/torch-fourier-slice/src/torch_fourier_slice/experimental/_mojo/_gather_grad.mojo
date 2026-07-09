"""Interpolation with analytical spatial gradients (linear / cubic).

Used by the backward pose/shift kernels: alongside the interpolated value it
returns the partial derivatives of the interpolated field w.r.t. the continuous
sample coordinate, `(d/dkz, d/dky, d/dkx)`. The result is packed into a `C8`:
lanes 0-1 = value, 2-3 = d/dkz, 4-5 = d/dky, 6-7 = d/dkx (each a complex pair).

The volume is a per-volume 4D `TileTensor` `[d, h, w, 2]` (see `_gather.mojo`).
"""

from std.math import floor

from layout import TensorLayout, TileTensor

from _common import C2, C8, _cubic_kernel, _cubic_kernel_derivative
from _gather import _sample_rfft_3d, _sample_rfft_3d_drop


@always_inline
def _pack(val: C2, gz: C2, gy: C2, gx: C2) -> C8:
    return C8(val[0], val[1], gz[0], gz[1], gy[0], gy[1], gx[0], gx[1])


@always_inline
def _smp[
    L: TensorLayout
](
    rec: TileTensor[DType.float32, L, MutAnyOrigin],
    z: Int,
    y: Int,
    x: Int,
    drop: Int,
) -> C2:
    """Sample, clamping (drop=0, forward gather) or dropping (drop=1, scatter adjoint).
    """
    if drop != 0:
        return _sample_rfft_3d_drop(rec, z, y, x)
    return _sample_rfft_3d(rec, z, y, x)


@always_inline
def _interp3d_linear_with_grad[
    L: TensorLayout
](
    rec: TileTensor[DType.float32, L, MutAnyOrigin],
    kz: Float32,
    ky: Float32,
    kx: Float32,
    drop: Int,
) -> C8:
    """Trilinear value + analytical (d/dkz, d/dky, d/dkx) at the sample point."""
    var kz_floor = floor(kz)
    var ky_floor = floor(ky)
    var kx_floor = floor(kx)
    var z = Int(kz_floor)
    var y = Int(ky_floor)
    var x = Int(kx_floor)
    var fz = kz - kz_floor
    var fy = ky - ky_floor
    var fx = kx - kx_floor
    var p000 = _smp(rec, z, y, x, drop)
    var p001 = _smp(rec, z, y, x + 1, drop)
    var p010 = _smp(rec, z, y + 1, x, drop)
    var p011 = _smp(rec, z, y + 1, x + 1, drop)
    var p100 = _smp(rec, z + 1, y, x, drop)
    var p101 = _smp(rec, z + 1, y, x + 1, drop)
    var p110 = _smp(rec, z + 1, y + 1, x, drop)
    var p111 = _smp(rec, z + 1, y + 1, x + 1, drop)
    var p00 = p000 + (p001 - p000) * fx
    var p01 = p010 + (p011 - p010) * fx
    var p10 = p100 + (p101 - p100) * fx
    var p11 = p110 + (p111 - p110) * fx
    var p0 = p00 + (p01 - p00) * fy
    var p1 = p10 + (p11 - p10) * fy
    var val = p0 + (p1 - p0) * fz
    var gz = p1 - p0
    var gy = (1.0 - fz) * (p01 - p00) + fz * (p11 - p10)
    var gx = (
        (1.0 - fz) * (1.0 - fy) * (p001 - p000)
        + (1.0 - fz) * fy * (p011 - p010)
        + fz * (1.0 - fy) * (p101 - p100)
        + fz * fy * (p111 - p110)
    )
    return _pack(val, gz, gy, gx)


@always_inline
def _interp3d_cubic_with_grad[
    L: TensorLayout
](
    rec: TileTensor[DType.float32, L, MutAnyOrigin],
    kz: Float32,
    ky: Float32,
    kx: Float32,
    drop: Int,
) -> C8:
    """Tricubic value + analytical (d/dkz, d/dky, d/dkx) over the 4x4x4 stencil."""
    var kz_floor = floor(kz)
    var ky_floor = floor(ky)
    var kx_floor = floor(kx)
    var z = Int(kz_floor)
    var y = Int(ky_floor)
    var x = Int(kx_floor)
    var fz = kz - kz_floor
    var fy = ky - ky_floor
    var fx = kx - kx_floor
    var val = C2(0.0, 0.0)
    var gz = C2(0.0, 0.0)
    var gy = C2(0.0, 0.0)
    var gx = C2(0.0, 0.0)
    for oz in range(-1, 3):
        var wz = _cubic_kernel(fz - Float32(oz))
        var dwz = _cubic_kernel_derivative(fz - Float32(oz))
        for oy in range(-1, 3):
            var wy = _cubic_kernel(fy - Float32(oy))
            var dwy = _cubic_kernel_derivative(fy - Float32(oy))
            for ox in range(-1, 3):
                var wx = _cubic_kernel(fx - Float32(ox))
                var dwx = _cubic_kernel_derivative(fx - Float32(ox))
                var s = _smp(rec, z + oz, y + oy, x + ox, drop)
                val = val + s * (wz * wy * wx)
                gz = gz + s * (dwz * wy * wx)
                gy = gy + s * (wz * dwy * wx)
                gx = gx + s * (wz * wy * dwx)
    return _pack(val, gz, gy, gx)


@always_inline
def _interp3d_with_grad[
    L: TensorLayout
](
    rec: TileTensor[DType.float32, L, MutAnyOrigin],
    kz: Float32,
    ky: Float32,
    kx: Float32,
    interp: Int,
    drop: Int,
) -> C8:
    """Interpolate + spatial gradients (interp: 0 = trilinear, 1 = tricubic).

    `drop`: 0 clamps out-of-range voxels (forward gather), 1 drops them (the
    exact adjoint of the scatter, for the backprojection gradient).
    """
    if interp == 1:
        return _interp3d_cubic_with_grad(rec, kz, ky, kx, drop)
    return _interp3d_linear_with_grad(rec, kz, ky, kx, drop)
