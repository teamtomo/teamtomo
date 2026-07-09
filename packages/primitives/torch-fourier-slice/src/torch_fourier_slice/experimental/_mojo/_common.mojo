"""Shared types, constants, and small helpers for the projector kernels.

Naming: volume spatial axes are (d, h, w) with sample coordinates (z, y, x);
the cube edge is `sidelength` (= h, = d) and `sidelength_half = w` is the rfft
width. `bv` is the batch of volumes, `bp` the batch of projections.
"""

from std.atomic import Atomic
from std.python import PythonObject

from layout import Coord, TensorLayout, TileTensor

comptime C2 = SIMD[DType.float32, 2]  # a complex value as (re, im)
comptime C8 = SIMD[DType.float32, 8]  # value + 3 spatial gradients, each (re, im)
comptime FP = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime BLOCK = 256
comptime PI: Float32 = 3.14159265358979323846


@always_inline
def _rfft_half(sidelength: Int) -> Int:
    """Non-redundant rfft width of a cube edge: `sidelength // 2 + 1`."""
    return sidelength // 2 + 1


@fieldwise_init
struct FourierSliceParams(Copyable, Movable):
    """Shapes and scalar parameters for one projection / scatter call.

    Volumes are cubic, so the real-space depth/height/width all equal
    `sidelength`; the rfft widths and the shift phase scale are *derived* from
    `sidelength` / `proj_sidelength` by the inlined accessors below rather than
    stored as redundant fields.
    """

    var bp: Int  # projections per volume
    var sidelength: Int  # cubic volume edge (real-space d == h == w)
    var proj_sidelength: Int  # projection edge
    var bv_rot: Int
    var bv_shift_2d: Int
    var oversampling: Float32
    var radius_cutoff_sq: Float32
    var has_shifts_2d: Int
    var interp: Int  # 0 = trilinear, 1 = tricubic
    var has_weights: Int  # scatter only
    var friedel_double: Int  # scatter only
    var skip_redundant: Int  # scatter only
    var ewald_curvature: Float32  # signed Ewald z-offset coeff; 0 = flat slice
    var has_shifts_3d: Int  # 3D (zyx, pre-rotation, volume-frame) shift present
    var bv_shift_3d: Int  # broadcast batch of the 3D shift

    @always_inline
    def sidelength_half(self) -> Int:
        """rfft width of the volume (= sidelength // 2 + 1)."""
        return _rfft_half(self.sidelength)

    @always_inline
    def proj_sidelength_half(self) -> Int:
        """rfft width of the projection (= proj_sidelength // 2 + 1)."""
        return _rfft_half(self.proj_sidelength)

    @always_inline
    def two_pi_over_proj_sidelength(self) -> Float32:
        """Phase-ramp scale `-2*pi / proj_sidelength` (2D image-plane shift)."""
        return -2.0 * PI / Float32(self.proj_sidelength)

    @always_inline
    def two_pi_over_sidelength(self) -> Float32:
        """Phase-ramp scale `-2*pi / sidelength` (3D volume-frame shift)."""
        return -2.0 * PI / Float32(self.sidelength)


@always_inline
def _ptr(t: PythonObject) raises -> FP:
    """Typed float32 pointer to a contiguous CPU tensor's buffer."""
    return FP(unsafe_from_address=Int(py=t.data_ptr()))


@always_inline
def _dptr(addr: PythonObject) raises -> FP:
    """Typed float32 pointer into GPU memory, from a raw device virtual address.

    The GPU entry points read/write the memory backing torch device tensors
    directly (no host round-trip). Python passes each buffer's device VA -- the
    CUDA ``data_ptr()`` or the Metal ``gpuAddress`` (see ``experimental/_gpu.py``,
    since torch's MPS ``data_ptr()`` is an ``MTLBuffer`` object pointer, not a
    VA) -- and this rebuilds a device pointer the kernels can dereference.
    """
    return FP(unsafe_from_address=Int(py=addr))


@always_inline
def _cmul(a: C2, b: C2) -> C2:
    return C2(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@always_inline
def _load_c2[
    L: TensorLayout
](t: TileTensor[DType.float32, L, MutAnyOrigin], i: Int, j: Int, k: Int) -> C2:
    """Load a complex value (re, im) from a 4D `[.., .., .., 2]` tile at (i, j, k)."""
    comptime assert t.flat_rank == 4, "complex tile must be 4D [.., .., .., 2]"
    return C2(
        rebind[Scalar[DType.float32]](t[i, j, k, 0]),
        rebind[Scalar[DType.float32]](t[i, j, k, 1]),
    )


@always_inline
def _store_c2[
    L: TensorLayout
](t: TileTensor[DType.float32, L, MutAnyOrigin], i: Int, j: Int, k: Int, v: C2):
    """Store a complex value (re, im) into a 4D `[.., .., .., 2]` tile at (i, j, k)."""
    comptime assert t.flat_rank == 4, "complex tile must be 4D [.., .., .., 2]"
    t[i, j, k, 0] = rebind[t.ElementType](v[0])
    t[i, j, k, 1] = rebind[t.ElementType](v[1])


@always_inline
def _atomic_add_at[
    L: TensorLayout
](t: TileTensor[DType.float32, L, MutAnyOrigin], coord: Coord[...], v: Float32) where (
    coord.flat_rank == t.flat_rank
):
    """Atomically add `v` into `t` at `coord` (addressed through the layout).

    Works for any rank/layout: `ptr_at_offset` resolves the element address via
    the tensor's strides, and `Atomic.fetch_add` lowers to the right CPU/GPU
    atomic. Used by the scatter accumulation (volume + weight volume).
    """
    _ = Atomic.fetch_add(t.ptr_at_offset(coord), v)


@always_inline
def _fourier_coord(i: Int, n: Int) -> Float32:
    """rfft frequency for array index `i` (DC at 0): positive then negative."""
    return Float32(i) if i <= n // 2 else Float32(i - n)


@always_inline
def _ewald_sz(p: FourierSliceParams, sx: Float32, sy: Float32) -> Float32:
    """Signed Ewald-sphere z-offset for a slice frequency (0 if curvature off).

    The slice bends onto the sphere by `dz = curvature * |k_xy|^2`; the signed
    `ewald_curvature` coeff folds the wavelength / pixel-size constants and the
    sign (none/positive/negative -> 0/+/-), so the kernel just scales the radius.
    """
    return p.ewald_curvature * (sx * sx + sy * sy)


@always_inline
def _shift_phase(
    p: FourierSliceParams,
    shifts_2d: FP,
    shifts_3d: FP,
    i_bv: Int,
    i_bp: Int,
    coord_y: Float32,
    coord_x: Float32,
    kz: Float32,
    ky: Float32,
    kx: Float32,
) -> Float32:
    """Combined shift phase: 2D (yx, image plane) + 3D (zyx, volume frame).

    The 2D shift is a phase ramp in the projection's own frequencies (post
    rotation); the 3D shift is applied in the volume frame before rotation, so
    its ramp uses the rotated sample coordinate `k = (kz, ky, kx)`.
    """
    var phase: Float32 = 0.0
    if p.has_shifts_2d != 0:
        var sb = 0 if p.bv_shift_2d == 1 else i_bv
        var sbase = (sb * p.bp + i_bp) * 2
        phase += (
            coord_y * shifts_2d[sbase] + coord_x * shifts_2d[sbase + 1]
        ) * p.two_pi_over_proj_sidelength()
    if p.has_shifts_3d != 0:
        var sb3 = 0 if p.bv_shift_3d == 1 else i_bv
        var s3 = (sb3 * p.bp + i_bp) * 3
        phase += (
            kz * shifts_3d[s3] + ky * shifts_3d[s3 + 1] + kx * shifts_3d[s3 + 2]
        ) * p.two_pi_over_sidelength()
    return phase


@always_inline
def _rotated_coord(
    rot: FP, rot_base: Int, sx: Float32, sy: Float32, sz: Float32
) -> SIMD[DType.float32, 4]:
    """Rotate a central-slice frequency (z=sz, y=sy, x=sx) into a volume coordinate.

    Rotation matrices are **zyx** convention: rows and columns are both ordered
    (z, y, x), stored row-major as `rot[row*3 + col]`. `sz` is the Ewald z-offset
    of the slice (0 for a flat slice), so the z-input column (cols 0/3/6) only
    contributes when curvature is enabled.

    Returns the sample coordinate (kz, ky, kx) in lanes 0..2 (lane 3 unused).
    """
    var kz = rot[rot_base + 0] * sz + rot[rot_base + 1] * sy + rot[rot_base + 2] * sx
    var ky = rot[rot_base + 3] * sz + rot[rot_base + 4] * sy + rot[rot_base + 5] * sx
    var kx = rot[rot_base + 6] * sz + rot[rot_base + 7] * sy + rot[rot_base + 8] * sx
    return SIMD[DType.float32, 4](kz, ky, kx, 0.0)


@always_inline
def _cubic_kernel(s_in: Float32) -> Float32:
    """Catmull-Rom cubic convolution weight (a = -0.5), support |s| <= 2."""
    var s = -s_in if s_in < 0 else s_in
    var a: Float32 = -0.5
    if s <= 1.0:
        return (a + 2.0) * s * s * s - (a + 3.0) * s * s + 1.0
    if s <= 2.0:
        return a * s * s * s - 5.0 * a * s * s + 8.0 * a * s - 4.0 * a
    return 0.0


@always_inline
def _cubic_kernel_derivative(s_in: Float32) -> Float32:
    """d/ds of `_cubic_kernel` (needed for spatial gradients of tricubic interp)."""
    var sign: Float32 = -1.0 if s_in < 0 else 1.0
    var s = -s_in if s_in < 0 else s_in
    var a: Float32 = -0.5
    if s <= 1.0:
        return sign * (3.0 * (a + 2.0) * s * s - 2.0 * (a + 3.0) * s)
    if s <= 2.0:
        return sign * (3.0 * a * s * s - 10.0 * a * s + 8.0 * a)
    return 0.0
