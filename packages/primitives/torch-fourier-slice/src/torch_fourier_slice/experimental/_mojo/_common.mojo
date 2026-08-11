"""Shared types, constants, and small helpers for the projector kernels.

Naming: volume spatial axes are (d, h, w) with sample coordinates (z, y, x);
the cube edge is `sidelength` (= h, = d) and `sidelength_half = w` is the rfft
width. `bv` is the batch of volumes, `bp` the batch of projections.
"""

from std.atomic import Atomic, Ordering
from std.python import PythonObject

from layout import Coord, TensorLayout, TileTensor

comptime C2 = SIMD[DType.float32, 2]  # a complex value as (re, im)
comptime C6 = SIMD[
    DType.float32, 6
]  # value + 2 spatial gradients, each (re, im) -- 2D interp-with-grad
comptime C8 = SIMD[
    DType.float32, 8
]  # value + 3 spatial gradients, each (re, im)
# Raw pointer into a contiguous float32 buffer (a torch tensor viewed as real float32,
# on CPU or GPU). Grouped into `Buffers` where several travel together; used bare only
# where a single buffer is passed. Origin erased (MutAnyOrigin) as it aliases foreign
# torch memory the kernels read/write in place.
comptime Float32Ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin]
comptime BLOCK = 256
comptime PI: Float32 = 3.14159265358979323846

# Interpolation kind — a COMPILE-TIME parameter of the gather/pixel/kernel chain, so each
# variant is specialised (no per-voxel runtime branch). The runtime `KernelParams.interp`
# code is read once at the entry-point boundary to pick the specialisation.
comptime LINEAR = 0
comptime CUBIC = 1


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
    var interp: Int  # LINEAR / CUBIC code; read once at the boundary to pick the comptime kernel
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


# --------------------------------------------------------------------------
# Per-kernel buffer bundles
#
# One struct per kernel naming exactly the buffers it uses, so the launchers and
# entry points pass a single named bundle instead of a positional pointer list.
# Built from torch tensors (`_ptr`, CPU) or raw device addresses (`_dptr`, GPU).
# --------------------------------------------------------------------------


@fieldwise_init
struct ProjectBuffers(Copyable, Movable):
    var rec: Float32Ptr
    var rot: Float32Ptr
    var shifts_2d: Float32Ptr
    var shifts_3d: Float32Ptr
    var proj: Float32Ptr


@fieldwise_init
struct ScatterBuffers(Copyable, Movable):
    var inp: Float32Ptr
    var weights: Float32Ptr
    var rot: Float32Ptr
    var shifts_2d: Float32Ptr
    var shifts_3d: Float32Ptr
    var vol: Float32Ptr
    var wvol: Float32Ptr


@fieldwise_init
struct ProjectLineBuffers(Copyable, Movable):
    """Forward central-*line* extraction (3D volume -> 1D lines).

    Poses are per-node *directions* `(bv_dir, bp, 3)` (zyx unit vectors), not
    rotation matrices -- a bare line carries no gauge (in-plane) freedom and no
    2D image-plane shift.
    """

    var rec: Float32Ptr
    var direction: Float32Ptr
    var shifts_3d: Float32Ptr
    var line: Float32Ptr


@fieldwise_init
struct ScatterLineBuffers(Copyable, Movable):
    """Central-*line* insertion (1D lines -> 3D volume + weights). Directions in.
    """

    var inp: Float32Ptr
    var weights: Float32Ptr
    var direction: Float32Ptr
    var shifts_3d: Float32Ptr
    var vol: Float32Ptr
    var wvol: Float32Ptr


@fieldwise_init
struct ForwardGradBuffers(Copyable, Movable):
    var rec: Float32Ptr
    var rot: Float32Ptr
    var shifts_2d: Float32Ptr
    var shifts_3d: Float32Ptr
    var grad_proj: Float32Ptr
    var grad_rot: Float32Ptr
    var grad_shift: Float32Ptr
    var grad_shift_3d: Float32Ptr


@fieldwise_init
struct ProjectLine2DBuffers(Copyable, Movable):
    """Forward 2D->1D central-line extraction (2D image -> 1D lines).

    Poses are per-node directions `(bv_dir, bp, 2)`; `shifts_2d` is an optional yx
    image translation `(bv_shift, bp, 2)` (phase ramp).
    """

    var img: Float32Ptr
    var direction: Float32Ptr
    var shifts_2d: Float32Ptr
    var line: Float32Ptr


@fieldwise_init
struct ScatterLine2DBuffers(Copyable, Movable):
    """2D->1D central-line insertion (1D lines -> 2D image + weights)."""

    var inp: Float32Ptr
    var weights: Float32Ptr
    var direction: Float32Ptr
    var shifts_2d: Float32Ptr
    var vol: Float32Ptr
    var wvol: Float32Ptr


@fieldwise_init
struct ForwardLine2DGradBuffers(Copyable, Movable):
    """Forward 2D->1D line pose grad: direction + 2D-shift grads."""

    var img: Float32Ptr
    var direction: Float32Ptr
    var shifts_2d: Float32Ptr
    var grad_line: Float32Ptr
    var grad_dir: Float32Ptr
    var grad_shift: Float32Ptr


@fieldwise_init
struct BackprojectLine2DGradBuffers(Copyable, Movable):
    """2D->1D line insertion pose grad: direction + 2D-shift grads."""

    var grad_img: Float32Ptr
    var direction: Float32Ptr
    var shifts_2d: Float32Ptr
    var lines: Float32Ptr
    var grad_dir: Float32Ptr
    var grad_shift: Float32Ptr


@fieldwise_init
struct WeightLine2DGradBuffers(Copyable, Movable):
    """2D->1D line insertion weight grad: adjoint of the real weight splat."""

    var gwimg: Float32Ptr
    var direction: Float32Ptr
    var grad_weight: Float32Ptr


@fieldwise_init
struct ForwardLineGradBuffers(Copyable, Movable):
    """Forward line pose grad: direction + 3D-shift grads (no 2D shift)."""

    var rec: Float32Ptr
    var direction: Float32Ptr
    var shifts_3d: Float32Ptr
    var grad_line: Float32Ptr
    var grad_dir: Float32Ptr
    var grad_shift_3d: Float32Ptr


@fieldwise_init
struct BackprojectLineGradBuffers(Copyable, Movable):
    """Line insertion pose grad: direction + 3D-shift grads (no 2D shift)."""

    var grad_rec: Float32Ptr
    var direction: Float32Ptr
    var shifts_3d: Float32Ptr
    var lines: Float32Ptr
    var grad_dir: Float32Ptr
    var grad_shift_3d: Float32Ptr


@fieldwise_init
struct WeightLineGradBuffers(Copyable, Movable):
    """Line insertion weight grad: adjoint of the real weight splat."""

    var gwvol: Float32Ptr
    var direction: Float32Ptr
    var grad_weight: Float32Ptr


@fieldwise_init
struct BackprojectGradBuffers(Copyable, Movable):
    var grad_rec: Float32Ptr
    var rot: Float32Ptr
    var shifts_2d: Float32Ptr
    var shifts_3d: Float32Ptr
    var proj: Float32Ptr
    var grad_rot: Float32Ptr
    var grad_shift: Float32Ptr
    var grad_shift_3d: Float32Ptr


@fieldwise_init
struct WeightGradBuffers(Copyable, Movable):
    var gwvol: Float32Ptr
    var rot: Float32Ptr
    var grad_weight: Float32Ptr


@always_inline
def _ptr(t: PythonObject) raises -> Float32Ptr:
    """Typed float32 pointer to a contiguous CPU tensor's buffer."""
    return Float32Ptr(unsafe_from_address=Int(py=t.data_ptr()))


@always_inline
def _dptr(addr: PythonObject) raises -> Float32Ptr:
    """Typed float32 pointer into GPU memory, from a raw device virtual address.

    The GPU entry points read/write the memory backing torch device tensors
    directly (no host round-trip). Python passes each buffer's device VA -- the
    CUDA ``data_ptr()`` or the Metal ``gpuAddress`` (see ``experimental/_gpu.py``,
    since torch's MPS ``data_ptr()`` is an ``MTLBuffer`` object pointer, not a
    VA) -- and this rebuilds a device pointer the kernels can dereference.
    """
    return Float32Ptr(unsafe_from_address=Int(py=addr))


@always_inline
def _cmul(a: C2, b: C2) -> C2:
    return C2(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@always_inline
def _load_c2[
    L: TensorLayout
](t: TileTensor[DType.float32, L, MutAnyOrigin], i: Int, j: Int, k: Int) -> C2:
    """Load a complex value (re, im) from a 4D `[.., .., .., 2]` tile at (i, j, k).
    """
    comptime assert t.flat_rank == 4, "complex tile must be 4D [.., .., .., 2]"
    return C2(
        rebind[Scalar[DType.float32]](t[i, j, k, 0]),
        rebind[Scalar[DType.float32]](t[i, j, k, 1]),
    )


@always_inline
def _store_c2[
    L: TensorLayout
](t: TileTensor[DType.float32, L, MutAnyOrigin], i: Int, j: Int, k: Int, v: C2):
    """Store a complex value (re, im) into a 4D `[.., .., .., 2]` tile at (i, j, k).
    """
    comptime assert t.flat_rank == 4, "complex tile must be 4D [.., .., .., 2]"
    t[i, j, k, 0] = rebind[t.ElementType](v[0])
    t[i, j, k, 1] = rebind[t.ElementType](v[1])


@always_inline
def _load_c2_line[
    L: TensorLayout
](t: TileTensor[DType.float32, L, MutAnyOrigin], i: Int, j: Int) -> C2:
    """Load a complex value (re, im) from a 3D `[.., .., 2]` line tile at (i, j).

    A central *line* node has no y axis, so its stack view is one rank lower than
    a central-slice stack (`[bp, w, 2]` rather than `[bp, h, w, 2]`).
    """
    comptime assert t.flat_rank == 3, "complex line tile must be 3D [.., .., 2]"
    return C2(
        rebind[Scalar[DType.float32]](t[i, j, 0]),
        rebind[Scalar[DType.float32]](t[i, j, 1]),
    )


@always_inline
def _store_c2_line[
    L: TensorLayout
](t: TileTensor[DType.float32, L, MutAnyOrigin], i: Int, j: Int, v: C2):
    """Store a complex value (re, im) into a 3D `[.., .., 2]` line tile at (i, j).
    """
    comptime assert t.flat_rank == 3, "complex line tile must be 3D [.., .., 2]"
    t[i, j, 0] = rebind[t.ElementType](v[0])
    t[i, j, 1] = rebind[t.ElementType](v[1])


@always_inline
def _atomic_add_at[
    L: TensorLayout
](
    t: TileTensor[DType.float32, L, MutAnyOrigin], coord: Coord[...], v: Float32
) where (coord.flat_rank == t.flat_rank):
    """Atomically add `v` into `t` at `coord` (addressed through the layout).

    Works for any rank/layout: `ptr_at_offset` resolves the element address via
    the tensor's strides, and `Atomic.fetch_add` lowers to the right CPU/GPU
    atomic. Used by the scatter accumulation (volume + weight volume).

    Explicit `ordering=RELAXED`: nothing else in the scatter synchronizes on
    the order these adds land in (only the final sum matters), and Mojo's
    default ordering for `fetch_add` on non-Apple GPU is SEQUENTIAL --
    substantially more expensive than a relaxed RMW under the heavy
    contention many overlapping central slices produce near the volume's DC
    voxels. CUDA's own `atomicAdd` (e.g. torch-projectors' backprojection
    kernel) has no such sequential-consistency guarantee, so this matches
    that behaviour rather than requiring it.
    """
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](t.ptr_at_offset(coord), v)


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
    shifts_2d: Float32Ptr,
    shifts_3d: Float32Ptr,
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
def _line_k(
    direction: Float32Ptr, base: Int, sx: Float32
) -> SIMD[DType.float32, 4]:
    """3D sample coordinate `k = s_x * u` for a line pixel.

    A central line is sampled along a direction `u = (u_z, u_y, u_x)` on the
    sphere (a zyx unit vector, the real-space line direction; the same vector in
    Fourier space since rotation commutes with the transform). Only the scalar
    frequency `s_x` varies along the node, so `k = s_x * u`; there is no gauge
    (in-plane) degree of freedom to carry, unlike a 2D slice's rotation matrix.

    Returns the sample coordinate (kz, ky, kx) in lanes 0..2 (lane 3 unused).
    """
    return SIMD[DType.float32, 4](
        sx * direction[base + 0],
        sx * direction[base + 1],
        sx * direction[base + 2],
        0.0,
    )


@always_inline
def _line_k_2d(
    direction: Float32Ptr, base: Int, sx: Float32
) -> SIMD[DType.float32, 2]:
    """2D sample coordinate `k = s_x * u` for a 2D->1D line pixel.

    A central line of a 2D image is sampled along a direction `u = (u_y, u_x)` on
    the circle (a yx unit vector, the real-space line direction; same in Fourier
    space). Only the scalar frequency `s_x` varies along the node, so `k = s_x*u`.

    Returns the sample coordinate (ky, kx).
    """
    return SIMD[DType.float32, 2](
        sx * direction[base + 0], sx * direction[base + 1]
    )


@always_inline
def _line2d_shift_phase(
    p: FourierSliceParams,
    shifts_2d: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    ky: Float32,
    kx: Float32,
) -> Float32:
    """Shift phase for a 2D->1D line: the yx image-translation ramp.

    A 2D shift `t = (t_y, t_x)` translates the image, putting `exp(-2*pi*i/N *
    (k . t))` on its FT. Sampled along the line `k = s*u`, this is the per-node
    `s * (u . t)` ramp (the 2D analogue of the 3D line's volume-frame shift).
    """
    if p.has_shifts_2d == 0:
        return 0.0
    var sb = 0 if p.bv_shift_2d == 1 else i_bv
    var s2 = (sb * p.bp + i_bp) * 2
    return (
        ky * shifts_2d[s2] + kx * shifts_2d[s2 + 1]
    ) * p.two_pi_over_sidelength()


@always_inline
def _line_shift_phase(
    p: FourierSliceParams,
    shifts_3d: Float32Ptr,
    i_bv: Int,
    i_bp: Int,
    kz: Float32,
    ky: Float32,
    kx: Float32,
) -> Float32:
    """Shift phase for a central *line*: the 3D (zyx, volume-frame) ramp only.

    A bare line has no image plane, so the 2D projection-plane shift is dropped
    (see the slice kernel's `_shift_phase`). The 3D shift `t` is applied in the
    volume frame before rotation, so its ramp uses the rotated sample coordinate
    `k = s*u`; for a line this collapses to the design's per-node scalar slope
    `s*(u . t)`.
    """
    if p.has_shifts_3d == 0:
        return 0.0
    var sb3 = 0 if p.bv_shift_3d == 1 else i_bv
    var s3 = (sb3 * p.bp + i_bp) * 3
    return (
        kz * shifts_3d[s3] + ky * shifts_3d[s3 + 1] + kx * shifts_3d[s3 + 2]
    ) * p.two_pi_over_sidelength()


@always_inline
def _rotated_coord(
    rot: Float32Ptr, rot_base: Int, sx: Float32, sy: Float32, sz: Float32
) -> SIMD[DType.float32, 4]:
    """Rotate a central-slice frequency (z=sz, y=sy, x=sx) into a volume coordinate.

    Rotation matrices are **zyx** convention: rows and columns are both ordered
    (z, y, x), stored row-major as `rot[row*3 + col]`. `sz` is the Ewald z-offset
    of the slice (0 for a flat slice), so the z-input column (cols 0/3/6) only
    contributes when curvature is enabled.

    Returns the sample coordinate (kz, ky, kx) in lanes 0..2 (lane 3 unused).
    """
    var kz = (
        rot[rot_base + 0] * sz + rot[rot_base + 1] * sy + rot[rot_base + 2] * sx
    )
    var ky = (
        rot[rot_base + 3] * sz + rot[rot_base + 4] * sy + rot[rot_base + 5] * sx
    )
    var kx = (
        rot[rot_base + 6] * sz + rot[rot_base + 7] * sy + rot[rot_base + 8] * sx
    )
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
    """d/ds of `_cubic_kernel` (needed for spatial gradients of tricubic interp).
    """
    var sign: Float32 = -1.0 if s_in < 0 else 1.0
    var s = -s_in if s_in < 0 else s_in
    var a: Float32 = -0.5
    if s <= 1.0:
        return sign * (3.0 * (a + 2.0) * s * s - 2.0 * (a + 3.0) * s)
    if s <= 2.0:
        return sign * (3.0 * a * s * s - 10.0 * a * s + 8.0 * a)
    return 0.0
