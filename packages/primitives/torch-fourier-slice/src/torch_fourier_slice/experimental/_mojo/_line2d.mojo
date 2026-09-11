"""Per-pixel ops for 2D->1D central-line extraction / insertion.

The exact dimensional analogue of `_line.mojo` one dimension lower: sample a 2D
rfft image `[h, w, 2]` (DC at origin) along a direction `u = (u_y, u_x)` on the
circle, `k = s_x * u`, producing a 1D rfft half-line. Reuses the 2D image gather
(`_interp2d`) and scatter (`_splat2d`); only the coordinate setup and the 1D I/O
differ. No 2D shift and no in-plane gauge (a bare line has neither).

Line pixel index into the stack is `x`; the sample coordinate is `(ky, kx)`.
"""

from std.math import cos, sin

from layout import TileTensor, row_major

from _common import (
    C2,
    Float32Ptr,
    FourierSliceParams,
    _cmul,
    _line2d_shift_phase,
    _line_k_2d,
    _load_c2_line,
    _rfft_half,
    _store_c2_line,
)
from _gather import _interp2d
from _scatter import _splat2d


@always_inline
def _project_line2d_pixel[
    interp: Int
](
    img: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    line: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Gather one output rfft pixel (i_bv, i_bp, x) of the 2D central line."""
    var coord_x = Float32(x)
    if coord_x * coord_x > p.radius_cutoff_sq:
        return  # outside the cutoff: leave the (pre-zeroed) output at 0

    var db = 0 if p.bv_rot == 1 else i_bv
    var sx = coord_x * p.oversampling
    var k = _line_k_2d(direction, (db * p.bp + i_bp) * 2, sx)

    # per-image 3D view of the rfft image [h, w, 2] (zero-copy over `img`)
    var half = _rfft_half(p.sidelength)
    var img_b = TileTensor(
        img + i_bv * p.sidelength * half * 2,
        row_major(p.sidelength, half, 2),
    )
    var val = _interp2d[interp](img_b, k[0], k[1])

    if p.has_shifts_2d != 0:
        var phase = _line2d_shift_phase(p, shifts_2d, i_bv, i_bp, k[0], k[1])
        val = _cmul(val, C2(cos(phase), sin(phase)))

    # per-image 3D view of the line block [bp, w, 2]
    var line_half = p.proj_sidelength_half()
    var line_b = TileTensor(
        line + i_bv * p.bp * line_half * 2,
        row_major(p.bp, line_half, 2),
    )
    _store_c2_line(line_b, i_bp, x, val)


@always_inline
def _scatter_line2d_pixel[
    interp: Int
](
    inp: Float32Ptr,
    weights: Float32Ptr,
    direction: Float32Ptr,
    shifts_2d: Float32Ptr,
    vol: Float32Ptr,
    wvol: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Splat one input rfft pixel (i_bv, i_bp, x) of the line into the 2D image.
    """
    var coord_x = Float32(x)
    if coord_x * coord_x > p.radius_cutoff_sq:
        return

    # per-image 3D view of the input line block [bp, w, 2]
    var line_half = p.proj_sidelength_half()
    var inp_b = TileTensor(
        inp + i_bv * p.bp * line_half * 2,
        row_major(p.bp, line_half, 2),
    )
    var vin = _load_c2_line(inp_b, i_bp, x)
    var vre = vin[0]
    var vim = vin[1]

    var db = 0 if p.bv_rot == 1 else i_bv
    var sx = coord_x * p.oversampling
    var k = _line_k_2d(direction, (db * p.bp + i_bp) * 2, sx)

    if p.has_shifts_2d != 0:
        # conjugate phase factor (adjoint of the forward line shift)
        var phase = _line2d_shift_phase(p, shifts_2d, i_bv, i_bp, k[0], k[1])
        var cr = cos(phase)
        var ci = sin(phase)
        var nre = vre * cr + vim * ci
        vim = -vre * ci + vim * cr
        vre = nre

    var wval: Float32 = 0.0
    if p.has_weights != 0:
        wval = weights[(i_bv * p.bp + i_bp) * line_half + x]

    # per-image views: complex data [h, w, 2] and real weights [h, w]
    var half = _rfft_half(p.sidelength)
    var vol_b = TileTensor(
        vol + i_bv * p.sidelength * half * 2,
        row_major(p.sidelength, half, 2),
    )
    var wvol_b = TileTensor(
        wvol + i_bv * p.sidelength * half,
        row_major(p.sidelength, half),
    )
    _splat2d[interp](vol_b, wvol_b, p, k[0], k[1], vre, vim, wval)
