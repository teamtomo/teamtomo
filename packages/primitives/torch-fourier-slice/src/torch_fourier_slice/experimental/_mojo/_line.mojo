"""Per-pixel ops for central-*line* extraction / insertion (3D <-> 1D).

A central line is the degenerate central slice whose in-plane (y) axis is
collapsed to the single DC row: the node is a 1D complex rfft array of Fourier
coefficients sampled along a **direction** `u` on the sphere (a zyx unit vector,
the real-space line direction; the same vector in Fourier space). One line pixel
`x` maps to the 3D sample coordinate `k = s * u`; the gather (`_interp3d`) and
scatter (`_splat`) are reused verbatim from the slice kernels -- only the
coordinate setup, the radius test, and the 1D I/O layout differ.

Unlike a 2D slice (which needs a full rotation matrix), a 1D line needs only its
direction: rotating about the line's own axis is a gauge the values are blind to.
Line pixel index into the stack is `x`; the sample coordinate is `(kz, ky, kx)`.
Ewald curvature and the 2D image-plane shift do not apply to a bare line, so
only the 3D (volume-frame) shift is honoured.
"""

from std.math import cos, sin

from layout import TileTensor, row_major

from _common import (
    C2,
    Float32Ptr,
    FourierSliceParams,
    _cmul,
    _line_k,
    _line_shift_phase,
    _load_c2_line,
    _rfft_half,
    _store_c2_line,
)
from _gather import _interp3d
from _scatter import _splat


@always_inline
def _project_line_pixel[
    interp: Int
](
    rec: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    line: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Gather one output rfft pixel (i_bv, i_bp, x) of the central line."""
    var coord_x = Float32(x)
    if coord_x * coord_x > p.radius_cutoff_sq:
        return  # outside the cutoff: leave the (pre-zeroed) output at 0

    # per-node direction broadcasts over the volume batch via bv_rot (= bv_dir).
    var db = 0 if p.bv_rot == 1 else i_bv
    var sx = coord_x * p.oversampling
    var k = _line_k(direction, (db * p.bp + i_bp) * 3, sx)

    # per-volume 4D view of the rfft cube [d, h, w, 2] (zero-copy over `rec`)
    var half = _rfft_half(p.sidelength)
    var rec_b = TileTensor(
        rec + i_bv * p.sidelength * p.sidelength * half * 2,
        row_major(p.sidelength, p.sidelength, half, 2),
    )
    var val = _interp3d[interp](rec_b, k[0], k[1], k[2])

    if p.has_shifts_3d != 0:
        var phase = _line_shift_phase(
            p, shifts_3d, i_bv, i_bp, k[0], k[1], k[2]
        )
        val = _cmul(val, C2(cos(phase), sin(phase)))

    # per-volume 3D view of the line block [bp, w, 2]
    var line_half = p.proj_sidelength_half()
    var line_b = TileTensor(
        line + i_bv * p.bp * line_half * 2,
        row_major(p.bp, line_half, 2),
    )
    _store_c2_line(line_b, i_bp, x, val)


@always_inline
def _scatter_line_pixel[
    interp: Int
](
    inp: Float32Ptr,
    weights: Float32Ptr,
    direction: Float32Ptr,
    shifts_3d: Float32Ptr,
    vol: Float32Ptr,
    wvol: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    x: Int,
    p: FourierSliceParams,
):
    """Splat one input rfft pixel (i_bv, i_bp, x) of the line into the volume.
    """
    var coord_x = Float32(x)
    if coord_x * coord_x > p.radius_cutoff_sq:
        return

    # per-volume 3D view of the input line block [bp, w, 2]
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
    var k = _line_k(direction, (db * p.bp + i_bp) * 3, sx)

    if p.has_shifts_3d != 0:
        # conjugate phase factor (adjoint of the forward line shift)
        var phase = _line_shift_phase(
            p, shifts_3d, i_bv, i_bp, k[0], k[1], k[2]
        )
        var cr = cos(phase)
        var ci = sin(phase)
        var nre = vre * cr + vim * ci
        vim = -vre * ci + vim * cr
        vre = nre

    var wval: Float32 = 0.0
    if p.has_weights != 0:
        wval = weights[(i_bv * p.bp + i_bp) * line_half + x]

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
    _splat[interp](vol_b, wvol_b, p, k[0], k[1], k[2], vre, vim, wval)
