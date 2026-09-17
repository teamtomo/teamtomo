"""Per-sample interpolation cores, generic over rank and interpolation kind.

Each function handles ONE sample `s` and is shared verbatim by the CPU loops
and the GPU threads. `ndim` (1, 2, 3) and `interp` (NEAREST, LINEAR, CUBIC) are
compile-time parameters, so the tap loops below have constant trip counts and
no per-sample branching on either.

The interpolation is separable: per axis a stencil of `T` taps (1, 2 or 4) with
weights `w[a][k]` and derivatives `dw[a][k]`; a sample touches the `T ** ndim`
product taps. Edge handling mirrors the torch path of this package:

- a sample whose coordinate lies outside `[0, D_a - 1]` on any axis reads /
  writes / receives gradient nothing (it is masked to zero);
- LINEAR taps that fall outside the image are dropped (this only happens at the
  upper edge where their weight is exactly 0 -- `torch.ceil` semantics);
- CUBIC taps outside the image are clamped to the edge (`grid_sample`'s
  `padding_mode="border"` behaviour for its 4x4 stencil).
"""

from std.math import floor, round

from _common import (
    CUBIC,
    LINEAR,
    NEAREST,
    F32Ptr,
    InterpParams,
    _add,
    _cubic_kernel,
    _cubic_kernel_derivative,
)


@always_inline
def _taps[interp: Int]() -> Int:
    """Taps per axis for an interpolation kind."""
    comptime if interp == NEAREST:
        return 1
    comptime if interp == LINEAR:
        return 2
    return 4


@always_inline
def _ipow(base: Int, exp: Int) -> Int:
    var r = 1
    for _ in range(exp):
        r *= base
    return r


@always_inline
def _base_index[interp: Int](x: Float32) -> Int:
    """Index of the first tap along an axis for coordinate `x` (may be < 0 for CUBIC)."""
    comptime if interp == NEAREST:
        return Int(round(x))
    elif interp == LINEAR:
        return Int(floor(x))
    else:
        return Int(floor(x)) - 1


@always_inline
def _axis_stencil[
    interp: Int
](
    x: Float32,
    a: Int,
    mut base: InlineArray[Int, 3],
    mut w: InlineArray[Float32, 12],
    mut dw: InlineArray[Float32, 12],
):
    """Fill axis `a`'s base index, tap weights and weight derivatives for coordinate `x`.

    Weights live at `w[a * 4 + k]` (only the first `_taps[interp]()` are used).
    """
    base[a] = _base_index[interp](x)
    comptime if interp == NEAREST:
        w[a * 4] = 1.0
        dw[a * 4] = 0.0
    elif interp == LINEAR:
        var t = x - floor(x)
        w[a * 4] = 1.0 - t
        w[a * 4 + 1] = t
        dw[a * 4] = -1.0
        dw[a * 4 + 1] = 1.0
    else:
        var t = x - floor(x)
        # taps at offsets -1, 0, 1, 2 from floor(x): kernel argument t - (k - 1)
        for k in range(4):
            var s = t - Float32(k - 1)
            w[a * 4 + k] = _cubic_kernel(s)
            dw[a * 4 + k] = _cubic_kernel_derivative(s)


struct Stencil[ndim: Int, interp: Int, grad: Bool = False]:
    """The product stencil of one sample: flat spatial offsets and weights.

    `sp[i]` is the flat spatial index of tap `i` (or -1 if the tap is dropped),
    `wt[i]` its weight, and -- only when the comptime `grad` flag is set --
    `dwt[i * ndim + a]` its weight's derivative w.r.t. coordinate `a`.
    """

    comptime T = _taps[Self.interp]()
    comptime NT = _ipow(Self.T, Self.ndim)
    comptime NG = Self.NT * Self.ndim if Self.grad else 1

    var inside: Bool
    var sp: InlineArray[Int, Self.NT]
    var wt: InlineArray[Float32, Self.NT]
    var dwt: InlineArray[Float32, Self.NG]

    @always_inline
    def __init__(out self, coords: F32Ptr, s: Int, p: InterpParams):
        self.inside = True
        self.sp = InlineArray[Int, Self.NT](uninitialized=True)
        self.wt = InlineArray[Float32, Self.NT](uninitialized=True)
        self.dwt = InlineArray[Float32, Self.NG](uninitialized=True)

        var dims = InlineArray[Int, 3](uninitialized=True)
        var base = InlineArray[Int, 3](uninitialized=True)
        var w = InlineArray[Float32, 12](uninitialized=True)
        var dw = InlineArray[Float32, 12](uninitialized=True)
        for a in range(Self.ndim):
            var d = p.dim(a)
            dims[a] = d
            var x = coords[unsafe_offset=s * Self.ndim + a]
            if x < 0.0 or x > Float32(d - 1):
                self.inside = False
                return
            _axis_stencil[Self.interp](x, a, base, w, dw)

        # row-major strides of the spatial axes (innermost axis has stride 1)
        var stride = InlineArray[Int, 3](uninitialized=True)
        stride[Self.ndim - 1] = 1
        for j in range(1, Self.ndim):
            var a = Self.ndim - 1 - j
            stride[a] = stride[a + 1] * dims[a + 1]

        # per-axis tap offsets (already clamped and scaled by the stride) and
        # validity, so the product loop below is pure adds and multiplies
        var off = InlineArray[Int, 12](uninitialized=True)
        for a in range(Self.ndim):
            var d = dims[a]
            for k in range(Self.T):
                var idx = base[a] + k
                if idx < 0 or idx >= d:
                    comptime if Self.interp == CUBIC:
                        idx = 0 if idx < 0 else d - 1
                    else:
                        off[a * 4 + k] = -1  # dropped tap (weight is exactly 0)
                        continue
                off[a * 4 + k] = idx * stride[a]

        for i in range(Self.NT):
            var rem = i
            var flat = 0
            var wgt: Float32 = 1.0
            var ks = InlineArray[Int, 3](uninitialized=True)
            for j in range(Self.ndim):
                var a = Self.ndim - 1 - j  # innermost axis varies fastest
                var k = rem % Self.T
                rem //= Self.T
                ks[a] = k
                var o = off[a * 4 + k]
                if o < 0:
                    flat = -1
                if flat >= 0:
                    flat += o
                wgt *= w[a * 4 + k]
            self.sp[i] = flat
            self.wt[i] = wgt
            comptime if Self.grad:
                for a in range(Self.ndim):
                    var g: Float32 = 1.0
                    for b in range(Self.ndim):
                        var kb = ks[b]
                        g *= dw[b * 4 + kb] if b == a else w[b * 4 + kb]
                    self.dwt[i * Self.ndim + a] = g


# ---------------------------------------------------------------------------
# Sampling (gather) and its backward
# ---------------------------------------------------------------------------


@always_inline
def _sample_one[
    ndim: Int, interp: Int
](img: F32Ptr, coords: F32Ptr, dst: F32Ptr, s: Int, p: InterpParams):
    """Write samples[s, :, :] = image interpolated at coordinates[s]."""
    var ci = p.c * p.inner
    var o = s * ci
    var st = Stencil[ndim, interp](coords, s, p)
    if not st.inside:
        for j in range(ci):
            dst[unsafe_offset=o + j] = 0.0
        return
    var spatial = p.spatial_size()
    for ch in range(p.c):
        var cb = ch * spatial * p.inner
        for k in range(p.inner):
            var acc: Float32 = 0.0
            for i in range(st.NT):
                var sp = st.sp[i]
                if sp < 0:
                    continue
                acc += img[unsafe_offset=cb + sp * p.inner + k] * st.wt[i]
            dst[unsafe_offset=o + ch * p.inner + k] = acc


@always_inline
def _sample_backward_one[
    ndim: Int, interp: Int, atomic: Bool = True
](
    img: F32Ptr,
    coords: F32Ptr,
    grad_out: F32Ptr,
    grad_img: F32Ptr,
    grad_coords: F32Ptr,
    s: Int,
    p: InterpParams,
):
    """Adjoint of `_sample_one` for sample `s`.

    Splats `grad_out[s]` into `grad_img` (if `need_grad_image`; atomically unless
    the caller guarantees disjoint writes, see `_add`) and writes
    `grad_coords[s]` = d(loss)/d(coordinates[s]) via the analytical spatial
    derivative of the interpolant (if `need_grad_coords`).
    """
    if p.need_grad_coords != 0:
        _sample_backward_impl[ndim, interp, atomic, True](
            img, coords, grad_out, grad_img, grad_coords, s, p
        )
    else:
        _sample_backward_impl[ndim, interp, atomic, False](
            img, coords, grad_out, grad_img, grad_coords, s, p
        )


@always_inline
def _sample_backward_impl[
    ndim: Int, interp: Int, atomic: Bool, grad_coords_needed: Bool
](
    img: F32Ptr,
    coords: F32Ptr,
    grad_out: F32Ptr,
    grad_img: F32Ptr,
    grad_coords: F32Ptr,
    s: Int,
    p: InterpParams,
):
    var need_gi = p.need_grad_image != 0
    var ci = p.c * p.inner
    var o = s * ci
    var st = Stencil[ndim, interp, grad_coords_needed](coords, s, p)
    var gc = InlineArray[Float32, 3](fill=0.0)
    if st.inside:
        var spatial = p.spatial_size()
        for ch in range(p.c):
            var cb = ch * spatial * p.inner
            for k in range(p.inner):
                var g = grad_out[unsafe_offset=o + ch * p.inner + k]
                for i in range(st.NT):
                    var sp = st.sp[i]
                    if sp < 0:
                        continue
                    var idx = cb + sp * p.inner + k
                    if need_gi:
                        _add[atomic](grad_img, idx, g * st.wt[i])
                    comptime if grad_coords_needed:
                        var gv = g * img[unsafe_offset=idx]
                        for a in range(ndim):
                            gc[a] += gv * st.dwt[i * ndim + a]
    comptime if grad_coords_needed:
        for a in range(ndim):
            grad_coords[unsafe_offset=s * ndim + a] = gc[a]


# ---------------------------------------------------------------------------
# Insertion (scatter) and its backward
# ---------------------------------------------------------------------------


@always_inline
def _insert_one[
    ndim: Int, interp: Int, atomic: Bool = True
](
    values: F32Ptr,
    coords: F32Ptr,
    img: F32Ptr,
    wimg: F32Ptr,
    s: Int,
    p: InterpParams,
):
    """Accumulate image += splat(values[s]) at coordinates[s]; weights += splat(1).

    Adds atomically unless the caller guarantees disjoint writes (see `_add`).
    """
    var st = Stencil[ndim, interp](coords, s, p)
    if not st.inside:
        return
    var ci = p.c * p.inner
    var o = s * ci
    var spatial = p.spatial_size()
    var has_w = p.has_weights != 0
    for i in range(st.NT):
        var sp = st.sp[i]
        if sp < 0:
            continue
        var wgt = st.wt[i]
        for ch in range(p.c):
            var cb = ch * spatial * p.inner + sp * p.inner
            for k in range(p.inner):
                _add[atomic](
                    img, cb + k, values[unsafe_offset=o + ch * p.inner + k] * wgt
                )
        if has_w:
            _add[atomic](wimg, sp, wgt)


@always_inline
def _insert_backward_one[
    ndim: Int, interp: Int
](
    values: F32Ptr,
    coords: F32Ptr,
    grad_img: F32Ptr,
    grad_wimg: F32Ptr,
    grad_values: F32Ptr,
    grad_coords: F32Ptr,
    s: Int,
    p: InterpParams,
):
    """Adjoint of `_insert_one` for sample `s`.

    `grad_values[s]` gathers `grad_img` with the forward stencil (if
    `need_grad_values`); `grad_coords[s]` chains `grad_img` (and `grad_wimg`,
    if `has_grad_weights`) through the weights' coordinate derivatives (if
    `need_grad_coords`).
    """
    if p.need_grad_coords != 0:
        _insert_backward_impl[ndim, interp, True](
            values, coords, grad_img, grad_wimg, grad_values, grad_coords, s, p
        )
    else:
        _insert_backward_impl[ndim, interp, False](
            values, coords, grad_img, grad_wimg, grad_values, grad_coords, s, p
        )


@always_inline
def _insert_backward_impl[
    ndim: Int, interp: Int, grad_coords_needed: Bool
](
    values: F32Ptr,
    coords: F32Ptr,
    grad_img: F32Ptr,
    grad_wimg: F32Ptr,
    grad_values: F32Ptr,
    grad_coords: F32Ptr,
    s: Int,
    p: InterpParams,
):
    var need_gv = p.need_grad_values != 0
    var has_gw = p.has_grad_weights != 0
    var ci = p.c * p.inner
    var o = s * ci
    var st = Stencil[ndim, interp, grad_coords_needed](coords, s, p)
    var gc = InlineArray[Float32, 3](fill=0.0)
    if st.inside:
        var spatial = p.spatial_size()
        for ch in range(p.c):
            for k in range(p.inner):
                var vo = o + ch * p.inner + k
                var v = values[unsafe_offset=vo]
                var acc: Float32 = 0.0
                for i in range(st.NT):
                    var sp = st.sp[i]
                    if sp < 0:
                        continue
                    var gi = grad_img[unsafe_offset=ch * spatial * p.inner + sp * p.inner + k]
                    acc += gi * st.wt[i]
                    comptime if grad_coords_needed:
                        for a in range(ndim):
                            gc[a] += gi * v * st.dwt[i * ndim + a]
                if need_gv:
                    grad_values[unsafe_offset=vo] = acc
        comptime if grad_coords_needed:
            if has_gw:
                for i in range(st.NT):
                    var sp = st.sp[i]
                    if sp < 0:
                        continue
                    var gw = grad_wimg[unsafe_offset=sp]
                    for a in range(ndim):
                        gc[a] += gw * st.dwt[i * ndim + a]
    else:
        if need_gv:
            for j in range(ci):
                grad_values[unsafe_offset=o + j] = 0.0
    comptime if grad_coords_needed:
        for a in range(ndim):
            grad_coords[unsafe_offset=s * ndim + a] = gc[a]
