"""Per-sample interpolation cores, generic over rank and interpolation kind.

Each function handles ONE sample `s` and is shared verbatim by the CPU loops
and the GPU threads. `ndim` (1, 2, 3) and `interp` (NEAREST, LINEAR, CUBIC) are
compile-time parameters, and every loop over axes or stencil taps is a
`comptime for`: the tap loops are unrolled and the small per-sample arrays are
only ever indexed by compile-time constants, so they stay in registers. (A
runtime-indexed `InlineArray` is lowered to per-thread *local* memory on the
GPU -- ~1 of every 32 bytes fetched per sector is used -- which ncu showed
doubling the time of the trilinear gather.)

The interpolation is separable: per axis a stencil of `T` taps (1, 2 or 4) with
weights `w[a][k]` and derivatives `dw[a][k]`; a sample touches the `T ** ndim`
product taps. Edge handling mirrors the torch path of this package:

- a sample whose coordinate lies outside `[0, D_a - 1]` on any axis reads /
  writes / receives gradient nothing (it is masked to zero);
- taps that fall outside the image are clamped to the edge voxel. For CUBIC
  this is `grid_sample`'s `padding_mode="border"` behaviour for its 4x4
  stencil. A LINEAR (or NEAREST) tap can only fall outside at the upper edge,
  where the coordinate is exactly `D_a - 1` and the tap's weight is exactly 0:
  clamping it onto its neighbour (which the torch path also does, via
  `padding_mode="border"` / `ceil == floor`) leaves values and gradients
  unchanged, and lets the tap loops run without a validity branch per tap.
"""

from std.math import floor

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
def _tap_index(i: Int, a: Int, T: Int, ndim: Int) -> Int:
    """Axis-`a` tap of product tap `i` (the innermost axis varies fastest)."""
    return (i // _ipow(T, ndim - 1 - a)) % T


@always_inline
def _round_half_even(x: Float32) -> Float32:
    """Round to nearest, ties to even -- `torch.round`'s convention.

    `std.math.round` rounds ties away from zero, which disagrees with the torch
    path of this package on every coordinate with an exact .5 fraction (they
    are common with grid-derived coordinates).
    """
    var r = floor(x + 0.5)
    if r - x == 0.5 and Int(r) % 2 != 0:
        r -= 1.0
    return r


@always_inline
def _base_index[interp: Int](x: Float32) -> Int:
    """Index of the first tap along an axis for coordinate `x` (may be < 0 for CUBIC).
    """
    comptime if interp == NEAREST:
        return Int(_round_half_even(x))
    elif interp == LINEAR:
        return Int(floor(x))
    else:
        return Int(floor(x)) - 1


@always_inline
def _axis_weights[
    interp: Int, a: Int
](
    x: Float32,
    mut w: InlineArray[Float32, 12],
    mut dw: InlineArray[Float32, 12],
):
    """Fill axis `a`'s tap weights and weight derivatives for coordinate `x`.

    Weights live at `w[a * 4 + k]` (only the first `_taps[interp]()` are used).
    """
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
        comptime for k in range(4):
            var s = t - Float32(k - 1)
            w[a * 4 + k] = _cubic_kernel(s)
            dw[a * 4 + k] = _cubic_kernel_derivative(s)


struct Stencil[ndim: Int, interp: Int]:
    """The separable stencil of one sample, held per axis.

    For axis `a` and tap `k < T`: `off[a * 4 + k]` is the tap's stride-scaled
    spatial offset (clamped into the image), `w[a * 4 + k]` its weight and
    `dw[a * 4 + k]` the weight's derivative w.r.t. the coordinate. Callers
    enumerate the `NT = T ** ndim` product taps with `comptime for i in
    range(NT)` and combine the per-axis entries through `offset[i]()`,
    `weight[i]()` and `dweight[i, axis]()`, whose indices are all compile-time
    constants.
    """

    comptime T = _taps[Self.interp]()
    comptime NT = _ipow(Self.T, Self.ndim)

    var inside: Bool
    var off: InlineArray[Int, 12]
    var w: InlineArray[Float32, 12]
    var dw: InlineArray[Float32, 12]

    @always_inline
    def __init__(out self, coords: F32Ptr, s: Int, p: InterpParams):
        self.inside = True
        self.off = InlineArray[Int, 12](uninitialized=True)
        self.w = InlineArray[Float32, 12](uninitialized=True)
        self.dw = InlineArray[Float32, 12](uninitialized=True)

        var dims = InlineArray[Int, 3](uninitialized=True)
        var base = InlineArray[Int, 3](uninitialized=True)
        comptime for a in range(Self.ndim):
            var d = p.dim(a)
            dims[a] = d
            var x = coords[unsafe_offset=s * Self.ndim + a]
            if x < 0.0 or x > Float32(d - 1):
                self.inside = False
                return
            base[a] = _base_index[Self.interp](x)
            _axis_weights[Self.interp, a](x, self.w, self.dw)

        # row-major strides of the spatial axes (innermost axis has stride 1)
        var stride = InlineArray[Int, 3](uninitialized=True)
        stride[Self.ndim - 1] = 1
        comptime for j in range(1, Self.ndim):
            comptime a = Self.ndim - 1 - j
            stride[a] = stride[a + 1] * dims[a + 1]

        # per-axis tap offsets, clamped into the image (see the module
        # docstring) and scaled by the stride, so a product tap's offset is a
        # plain sum of `ndim` of these
        comptime for a in range(Self.ndim):
            var d = dims[a]
            comptime for k in range(Self.T):
                var idx = base[a] + k
                if idx < 0:
                    idx = 0
                elif idx >= d:
                    idx = d - 1
                self.off[a * 4 + k] = idx * stride[a]

    @always_inline
    def offset[i: Int](self) -> Int:
        """Flat spatial offset of product tap `i`."""
        var flat = 0
        comptime for j in range(Self.ndim):
            comptime a = Self.ndim - 1 - j
            comptime k = _tap_index(i, a, Self.T, Self.ndim)
            flat += self.off[a * 4 + k]
        return flat

    @always_inline
    def weight[i: Int](self) -> Float32:
        """Weight of product tap `i` (product of its axis weights, innermost first).
        """
        var g: Float32 = 1.0
        comptime for j in range(Self.ndim):
            comptime a = Self.ndim - 1 - j
            comptime k = _tap_index(i, a, Self.T, Self.ndim)
            g *= self.w[a * 4 + k]
        return g

    @always_inline
    def dweight[i: Int, axis: Int](self) -> Float32:
        """d(weight of product tap `i`) / d(coordinate `axis`)."""
        var g: Float32 = 1.0
        comptime for b in range(Self.ndim):
            comptime k = _tap_index(i, b, Self.T, Self.ndim)
            comptime if b == axis:
                g *= self.dw[b * 4 + k]
            else:
                g *= self.w[b * 4 + k]
        return g


# ---------------------------------------------------------------------------
# Sampling (gather) and its backward
#
# `inner` (1 for a real image, 2 for complex) selects the vector width `W` of
# the per-voxel loads and stores: a complex voxel is one 8-byte access rather
# than two 4-byte ones -- half the memory instructions and L1 wavefronts of the
# gathers, which are latency-bound. The CPU driver leaves `W = 0` and each
# core branches once (uniformly) on the runtime `p.inner`; the GPU kernels are
# specialised on `W`, so a kernel never carries the registers of the other
# width's code path.
# ---------------------------------------------------------------------------


@always_inline
def _gather[
    ndim: Int, interp: Int, W: Int
](img: F32Ptr, base: Int, st: Stencil[ndim, interp]) -> SIMD[DType.float32, W]:
    """Sum over the stencil of `img[base + W * offset : + W] * weight`."""
    var acc = SIMD[DType.float32, W](0)
    comptime for i in range(st.NT):
        acc += img.unsafe_load[width=W](
            offset=base + st.offset[i]() * W
        ) * SIMD[DType.float32, W](st.weight[i]())
    return acc


@always_inline
def _gather_channels[
    ndim: Int, interp: Int, W: Int
](
    img: F32Ptr,
    dst: F32Ptr,
    o: Int,
    spatial: Int,
    c: Int,
    st: Stencil[ndim, interp],
):
    for ch in range(c):
        dst.unsafe_store(
            offset=o + ch * W,
            val=_gather[ndim, interp, W](img, ch * spatial * W, st),
        )


@always_inline
def _sample_one[
    ndim: Int, interp: Int, W: Int = 0
](img: F32Ptr, coords: F32Ptr, dst: F32Ptr, s: Int, p: InterpParams):
    """Write samples[s, :, :] = image interpolated at coordinates[s].

    `W` is the vector width of a voxel (`p.inner`), or 0 to branch on `p.inner`
    at runtime.
    """
    var ci = p.c * p.inner
    var o = s * ci
    var st = Stencil[ndim, interp](coords, s, p)
    if not st.inside:
        for j in range(ci):
            dst[unsafe_offset=o + j] = 0.0
        return
    var spatial = p.spatial_size()
    comptime if W == 0:
        if p.inner == 2:
            _gather_channels[ndim, interp, 2](img, dst, o, spatial, p.c, st)
        else:
            _gather_channels[ndim, interp, 1](img, dst, o, spatial, p.c, st)
    else:
        _gather_channels[ndim, interp, W](img, dst, o, spatial, p.c, st)


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
        if p.need_grad_image != 0:
            _sample_backward_impl[ndim, interp, atomic, True, True](
                img, coords, grad_out, grad_img, grad_coords, s, p
            )
        else:
            _sample_backward_impl[ndim, interp, atomic, False, True](
                img, coords, grad_out, grad_img, grad_coords, s, p
            )
    else:
        _sample_backward_impl[ndim, interp, atomic, True, False](
            img, coords, grad_out, grad_img, grad_coords, s, p
        )


@always_inline
def _sample_backward_taps[
    ndim: Int,
    interp: Int,
    atomic: Bool,
    grad_image_needed: Bool,
    grad_coords_needed: Bool,
    W: Int,
](
    img: F32Ptr,
    grad_out: F32Ptr,
    grad_img: F32Ptr,
    go: Int,
    base: Int,
    st: Stencil[ndim, interp],
    mut gc: InlineArray[Float32, 3],
):
    """One channel of `_sample_backward_impl`: `grad_out[go : go + W]` against
    the stencil at `img[base + W * offset]`."""
    var g = grad_out.unsafe_load[width=W](offset=go)
    comptime for i in range(st.NT):
        var idx = base + st.offset[i]() * W
        comptime if grad_image_needed:
            var contrib = g * SIMD[DType.float32, W](st.weight[i]())
            comptime for k in range(W):
                _add[atomic](grad_img, idx + k, contrib[k])
        comptime if grad_coords_needed:
            var gv = (g * img.unsafe_load[width=W](offset=idx)).reduce_add()
            comptime for a in range(ndim):
                gc[a] += gv * st.dweight[i, a]()


@always_inline
def _sample_backward_channels[
    ndim: Int,
    interp: Int,
    atomic: Bool,
    grad_image_needed: Bool,
    grad_coords_needed: Bool,
    W: Int,
](
    img: F32Ptr,
    grad_out: F32Ptr,
    grad_img: F32Ptr,
    o: Int,
    spatial: Int,
    c: Int,
    st: Stencil[ndim, interp],
    mut gc: InlineArray[Float32, 3],
):
    for ch in range(c):
        _sample_backward_taps[
            ndim, interp, atomic, grad_image_needed, grad_coords_needed, W
        ](img, grad_out, grad_img, o + ch * W, ch * spatial * W, st, gc)


@always_inline
def _sample_backward_impl[
    ndim: Int,
    interp: Int,
    atomic: Bool,
    grad_image_needed: Bool,
    grad_coords_needed: Bool,
    W: Int = 0,
](
    img: F32Ptr,
    coords: F32Ptr,
    grad_out: F32Ptr,
    grad_img: F32Ptr,
    grad_coords: F32Ptr,
    s: Int,
    p: InterpParams,
):
    """`_sample_backward_one` with the `need_grad_*` flags fixed at compile time.

    The two gradients need very different register budgets (the coordinate
    gradient carries `ndim` weight derivatives per tap), so a GPU kernel
    specialised on the flags it actually needs runs at a higher occupancy
    than one holding both paths behind a runtime branch. `W` as in
    `_sample_one`.
    """
    var o = s * p.c * p.inner
    var st = Stencil[ndim, interp](coords, s, p)
    var gc = InlineArray[Float32, 3](fill=0.0)
    if st.inside:
        var spatial = p.spatial_size()
        comptime if W == 0:
            if p.inner == 2:
                _sample_backward_channels[
                    ndim,
                    interp,
                    atomic,
                    grad_image_needed,
                    grad_coords_needed,
                    2,
                ](img, grad_out, grad_img, o, spatial, p.c, st, gc)
            else:
                _sample_backward_channels[
                    ndim,
                    interp,
                    atomic,
                    grad_image_needed,
                    grad_coords_needed,
                    1,
                ](img, grad_out, grad_img, o, spatial, p.c, st, gc)
        else:
            _sample_backward_channels[
                ndim, interp, atomic, grad_image_needed, grad_coords_needed, W
            ](img, grad_out, grad_img, o, spatial, p.c, st, gc)
    comptime if grad_coords_needed:
        comptime for a in range(ndim):
            grad_coords[unsafe_offset=s * ndim + a] = gc[a]


# ---------------------------------------------------------------------------
# Insertion (scatter) and its backward
# ---------------------------------------------------------------------------


@always_inline
def _scatter[
    ndim: Int, interp: Int, atomic: Bool, W: Int
](img: F32Ptr, base: Int, v: SIMD[DType.float32, W], st: Stencil[ndim, interp]):
    """`img[base + W * offset : + W] += v * weight` over the stencil."""
    comptime for i in range(st.NT):
        var idx = base + st.offset[i]() * W
        var contrib = v * SIMD[DType.float32, W](st.weight[i]())
        comptime for k in range(W):
            _add[atomic](img, idx + k, contrib[k])


@always_inline
def _scatter_channels[
    ndim: Int, interp: Int, atomic: Bool, W: Int
](
    values: F32Ptr,
    img: F32Ptr,
    o: Int,
    spatial: Int,
    c: Int,
    st: Stencil[ndim, interp],
):
    for ch in range(c):
        _scatter[ndim, interp, atomic, W](
            img,
            ch * spatial * W,
            values.unsafe_load[width=W](offset=o + ch * W),
            st,
        )


@always_inline
def _insert_one[
    ndim: Int, interp: Int, atomic: Bool = True, W: Int = 0
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
    `W` as in `_sample_one`.
    """
    var st = Stencil[ndim, interp](coords, s, p)
    if not st.inside:
        return
    var o = s * p.c * p.inner
    var spatial = p.spatial_size()
    comptime if W == 0:
        if p.inner == 2:
            _scatter_channels[ndim, interp, atomic, 2](
                values, img, o, spatial, p.c, st
            )
        else:
            _scatter_channels[ndim, interp, atomic, 1](
                values, img, o, spatial, p.c, st
            )
    else:
        _scatter_channels[ndim, interp, atomic, W](
            values, img, o, spatial, p.c, st
        )
    if p.has_weights != 0:
        comptime for i in range(st.NT):
            _add[atomic](wimg, st.offset[i](), st.weight[i]())


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
        if p.need_grad_values != 0:
            _insert_backward_impl[ndim, interp, True, True](
                values,
                coords,
                grad_img,
                grad_wimg,
                grad_values,
                grad_coords,
                s,
                p,
            )
        else:
            _insert_backward_impl[ndim, interp, False, True](
                values,
                coords,
                grad_img,
                grad_wimg,
                grad_values,
                grad_coords,
                s,
                p,
            )
    else:
        _insert_backward_impl[ndim, interp, True, False](
            values, coords, grad_img, grad_wimg, grad_values, grad_coords, s, p
        )


@always_inline
def _insert_backward_taps[
    ndim: Int,
    interp: Int,
    grad_values_needed: Bool,
    grad_coords_needed: Bool,
    W: Int,
](
    values: F32Ptr,
    grad_img: F32Ptr,
    grad_values: F32Ptr,
    vo: Int,
    base: Int,
    st: Stencil[ndim, interp],
    mut gc: InlineArray[Float32, 3],
):
    """One channel of `_insert_backward_impl`: `values[vo : vo + W]` against the
    stencil at `grad_img[base + W * offset]`."""
    var v = values.unsafe_load[width=W](offset=vo)
    var acc = SIMD[DType.float32, W](0)
    comptime for i in range(st.NT):
        var gi = grad_img.unsafe_load[width=W](offset=base + st.offset[i]() * W)
        acc += gi * SIMD[DType.float32, W](st.weight[i]())
        comptime if grad_coords_needed:
            var giv = (gi * v).reduce_add()
            comptime for a in range(ndim):
                gc[a] += giv * st.dweight[i, a]()
    comptime if grad_values_needed:
        grad_values.unsafe_store(offset=vo, val=acc)


@always_inline
def _insert_backward_channels[
    ndim: Int,
    interp: Int,
    grad_values_needed: Bool,
    grad_coords_needed: Bool,
    W: Int,
](
    values: F32Ptr,
    grad_img: F32Ptr,
    grad_values: F32Ptr,
    o: Int,
    spatial: Int,
    c: Int,
    st: Stencil[ndim, interp],
    mut gc: InlineArray[Float32, 3],
):
    for ch in range(c):
        _insert_backward_taps[
            ndim, interp, grad_values_needed, grad_coords_needed, W
        ](values, grad_img, grad_values, o + ch * W, ch * spatial * W, st, gc)


@always_inline
def _insert_backward_impl[
    ndim: Int,
    interp: Int,
    grad_values_needed: Bool,
    grad_coords_needed: Bool,
    W: Int = 0,
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
    """`_insert_backward_one` with the `need_grad_*` flags fixed at compile time
    (see `_sample_backward_impl`). `W` as in `_sample_one`."""
    var ci = p.c * p.inner
    var o = s * ci
    var st = Stencil[ndim, interp](coords, s, p)
    var gc = InlineArray[Float32, 3](fill=0.0)
    if st.inside:
        var spatial = p.spatial_size()
        comptime if W == 0:
            if p.inner == 2:
                _insert_backward_channels[
                    ndim, interp, grad_values_needed, grad_coords_needed, 2
                ](values, grad_img, grad_values, o, spatial, p.c, st, gc)
            else:
                _insert_backward_channels[
                    ndim, interp, grad_values_needed, grad_coords_needed, 1
                ](values, grad_img, grad_values, o, spatial, p.c, st, gc)
        else:
            _insert_backward_channels[
                ndim, interp, grad_values_needed, grad_coords_needed, W
            ](values, grad_img, grad_values, o, spatial, p.c, st, gc)
        comptime if grad_coords_needed:
            if p.has_grad_weights != 0:
                comptime for i in range(st.NT):
                    var gw = grad_wimg[unsafe_offset=st.offset[i]()]
                    comptime for a in range(ndim):
                        gc[a] += gw * st.dweight[i, a]()
    else:
        comptime if grad_values_needed:
            for j in range(ci):
                grad_values[unsafe_offset=o + j] = 0.0
    comptime if grad_coords_needed:
        comptime for a in range(ndim):
            grad_coords[unsafe_offset=s * ndim + a] = gc[a]
