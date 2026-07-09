"""Per-pixel ops shared by the CPU loops and the GPU threads.

One forward pixel gathers a sample; one scatter pixel splats a sample. Pixel
indices into the projection are `(y, x)`; the sample coordinate is
`(kz, ky, kx)`.
"""

from std.math import cos, sin

from layout import TileTensor, row_major

from _common import (
    C2,
    FP,
    FourierSliceParams,
    _cmul,
    _ewald_sz,
    _fourier_coord,
    _load_c2,
    _rfft_half,
    _rotated_coord,
    _shift_phase,
    _store_c2,
)
from _gather import _interp3d
from _scatter import _splat


@always_inline
def _project_pixel(
    rec: FP,
    rot: FP,
    shifts_2d: FP,
    shifts_3d: FP,
    proj: FP,
    i_bv: Int,
    i_bp: Int,
    y: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Gather one output rfft pixel (i_bv, i_bp, y, x) of the central slice."""
    var coord_y = _fourier_coord(y, p.proj_sidelength)
    var coord_x = Float32(x)
    if coord_y * coord_y + coord_x * coord_x > p.radius_cutoff_sq:
        return  # outside the cutoff: leave the (pre-zeroed) output at 0

    var rb = 0 if p.bv_rot == 1 else i_bv
    var sx = coord_x * p.oversampling
    var sy = coord_y * p.oversampling
    var k = _rotated_coord(
        rot, (rb * p.bp + i_bp) * 9, sx, sy, _ewald_sz(p, sx, sy)
    )
    # per-volume 4D view of the rfft cube [d, h, w, 2] (zero-copy over `rec`)
    var half = _rfft_half(p.sidelength)
    var rec_b = TileTensor(
        rec + i_bv * p.sidelength * p.sidelength * half * 2,
        row_major(p.sidelength, p.sidelength, half, 2),
    )
    var val = _interp3d(rec_b, k[0], k[1], k[2], p.interp)

    if p.has_shifts_2d != 0 or p.has_shifts_3d != 0:
        var phase = _shift_phase(
            p, shifts_2d, shifts_3d, i_bv, i_bp, coord_y, coord_x, k[0], k[1], k[2]
        )
        val = _cmul(val, C2(cos(phase), sin(phase)))

    # per-volume 4D view of the projection block [bp, h, w, 2]
    var proj_half = p.proj_sidelength_half()
    var proj_b = TileTensor(
        proj + i_bv * p.bp * p.proj_sidelength * proj_half * 2,
        row_major(p.bp, p.proj_sidelength, proj_half, 2),
    )
    _store_c2(proj_b, i_bp, y, x, val)


@always_inline
def _scatter_pixel(
    inp: FP,
    weights: FP,
    rot: FP,
    shifts_2d: FP,
    shifts_3d: FP,
    vol: FP,
    wvol: FP,
    i_bv: Int,
    i_bp: Int,
    y: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Splat one input rfft pixel (i_bv, i_bp, y, x) into the volume (adjoint)."""
    var coord_y = _fourier_coord(y, p.proj_sidelength)
    var coord_x = Float32(x)
    if coord_y * coord_y + coord_x * coord_x > p.radius_cutoff_sq:
        return
    # backprojection skips the redundant half of the x=0 line (mirror handles it)
    if p.skip_redundant != 0 and x == 0 and y >= p.proj_sidelength // 2:
        return

    # per-volume 4D view of the input projection block [bp, h, w, 2]
    var proj_half = p.proj_sidelength_half()
    var inp_b = TileTensor(
        inp + i_bv * p.bp * p.proj_sidelength * proj_half * 2,
        row_major(p.bp, p.proj_sidelength, proj_half, 2),
    )
    var vin = _load_c2(inp_b, i_bp, y, x)
    var vre = vin[0]
    var vim = vin[1]

    # the 3D shift phase needs the rotated coordinate, so compute `k` first
    var rb = 0 if p.bv_rot == 1 else i_bv
    var sx = coord_x * p.oversampling
    var sy = coord_y * p.oversampling
    var k = _rotated_coord(
        rot, (rb * p.bp + i_bp) * 9, sx, sy, _ewald_sz(p, sx, sy)
    )
    if p.has_shifts_2d != 0 or p.has_shifts_3d != 0:
        # conjugate combined phase factor (adjoint of the forward shift)
        var phase = _shift_phase(
            p, shifts_2d, shifts_3d, i_bv, i_bp, coord_y, coord_x, k[0], k[1], k[2]
        )
        var cr = cos(phase)
        var ci = sin(phase)
        var nre = vre * cr + vim * ci
        vim = -vre * ci + vim * cr
        vre = nre

    var wval: Float32 = 0.0
    if p.has_weights != 0:
        wval = weights[
            (
                ((i_bv * p.bp + i_bp) * p.proj_sidelength + y)
                * p.proj_sidelength_half()
                + x
            )
        ]

    # per-volume views: complex data [d, h, w, 2] and real weights [d, h, w]
    var half = _rfft_half(p.sidelength)
    var vol_b = TileTensor(
        vol + i_bv * p.sidelength * p.sidelength * half * 2,
        row_major(p.sidelength, p.sidelength, half, 2),
    )
    var wvol_b = TileTensor(
        wvol + i_bv * p.sidelength * p.sidelength * half,
        row_major(p.sidelength, p.sidelength, half),
    )
    _splat(vol_b, wvol_b, p, k[0], k[1], k[2], vre, vim, wval)
