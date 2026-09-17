"""Shared types, constants and helpers for the image-interpolation kernels.

Layouts (all C-contiguous float32; complex data is `torch.view_as_real`, so a
complex image carries a trailing axis of length 2 and a real image of length 1):

    image        [c, D_0, ..., D_{ndim-1}, inner]
    coordinates  [n, ndim]          (array coordinates, ordered like the axes)
    samples      [n, c, inner]      (values read from / written to the image)
    weights      [D_0, ..., D_{ndim-1}]   (real, insertion only)

The spatial axes are ordered like the tensor: `(z, y, x)` in 3D, `(y, x)` in 2D
and `(x,)` in 1D. `ndim` and the interpolation kind are compile-time parameters
of every kernel (see `_interp.mojo`), so each of the 3 x 3 variants is
specialised with no per-sample branching on either.
"""

from std.atomic import Atomic, Ordering
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.python import PythonObject

# Raw pointer into a contiguous float32 buffer (a torch tensor, on CPU or GPU).
# Origin erased: it aliases foreign torch memory whose lifetime the Python caller
# manages.
comptime F32Ptr = Pointer[Scalar[DType.float32], UntrackedOrigin[mut=True]]
comptime BLOCK = 256

# Interpolation kinds: compile-time parameters of the kernel chain. The runtime
# `InterpParams.interp` code is read once at the entry-point boundary to select
# the specialisation.
comptime NEAREST = 0
comptime LINEAR = 1
comptime CUBIC = 2

# Cubic convolution coefficient. torch's `grid_sample(mode="bicubic")` and
# `interpolate(mode="bicubic")` both use A = -0.75 (not Catmull-Rom's -0.5), and
# the torch path of this package is the reference these kernels must match.
comptime CUBIC_A: Float32 = -0.75


@fieldwise_init
struct InterpParams(Copyable, Movable):
    """Scalar kernel parameters, read once at the Python boundary.

    `d0, d1, d2` are the spatial extents in axis order; axes beyond `ndim` are
    1 so `spatial_size()` is valid for any rank. The `need_*` / `has_*` flags
    select which optional outputs a backward kernel produces.
    """

    var ndim: Int
    var interp: Int
    var n: Int
    var c: Int
    var inner: Int
    var d0: Int
    var d1: Int
    var d2: Int
    var has_weights: Int
    var need_grad_image: Int
    var need_grad_coords: Int
    var need_grad_values: Int
    var has_grad_weights: Int
    # GPU only: zero `grad_image` in-kernel before accumulating, so the caller
    # can pass an uninitialised buffer and skip one queue round trip
    var zero_grad_image: Int

    @always_inline
    def dim(self, a: Int) -> Int:
        """Extent of spatial axis `a` (0-based, in tensor order)."""
        if a == 0:
            return self.d0
        if a == 1:
            return self.d1
        return self.d2

    @always_inline
    def spatial_size(self) -> Int:
        return self.d0 * self.d1 * self.d2

    @always_inline
    def to_device(self, total: Int) -> DeviceParams:
        """Fixed-width form for a single GPU kernel-launch argument."""
        return DeviceParams(
            Int64(total),
            Int64(self.n),
            Int64(self.c),
            Int64(self.inner),
            Int64(self.d0),
            Int64(self.d1),
            Int64(self.d2),
            Int32(self.has_weights),
            Int32(self.need_grad_image),
            Int32(self.need_grad_coords),
            Int32(self.need_grad_values),
            Int32(self.has_grad_weights),
            Int32(self.zero_grad_image),
        )


@fieldwise_init
struct DeviceParams(Copyable, DevicePassable, Movable):
    """`InterpParams` with fixed-width fields for the host->device launch ABI.

    GPU kernel arguments must be `DevicePassable`, which `Int` is not. `ndim`
    and `interp` are omitted: they are the kernel's compile-time parameters.
    """

    comptime device_type = Self

    var total: Int64
    var n: Int64
    var c: Int64
    var inner: Int64
    var d0: Int64
    var d1: Int64
    var d2: Int64
    var has_weights: Int32
    var need_grad_image: Int32
    var need_grad_coords: Int32
    var need_grad_values: Int32
    var has_grad_weights: Int32
    var zero_grad_image: Int32

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "DeviceParams"

    @always_inline
    def to_params(self, ndim: Int, interp: Int) -> InterpParams:
        return InterpParams(
            ndim,
            interp,
            Int(self.n),
            Int(self.c),
            Int(self.inner),
            Int(self.d0),
            Int(self.d1),
            Int(self.d2),
            Int(self.has_weights),
            Int(self.need_grad_image),
            Int(self.need_grad_coords),
            Int(self.need_grad_values),
            Int(self.has_grad_weights),
            Int(self.zero_grad_image),
        )


@always_inline
def _read_params(params_obj: PythonObject) raises -> InterpParams:
    """Build `InterpParams` from the Python `KernelParams` NamedTuple (by name)."""
    return InterpParams(
        ndim=Int(py=params_obj.ndim),
        interp=Int(py=params_obj.interp),
        n=Int(py=params_obj.n),
        c=Int(py=params_obj.c),
        inner=Int(py=params_obj.inner),
        d0=Int(py=params_obj.d0),
        d1=Int(py=params_obj.d1),
        d2=Int(py=params_obj.d2),
        has_weights=Int(py=params_obj.has_weights),
        need_grad_image=Int(py=params_obj.need_grad_image),
        need_grad_coords=Int(py=params_obj.need_grad_coords),
        need_grad_values=Int(py=params_obj.need_grad_values),
        has_grad_weights=Int(py=params_obj.has_grad_weights),
        zero_grad_image=Int(py=params_obj.zero_grad_image),
    )


@always_inline
def _ptr(t: PythonObject) raises -> F32Ptr:
    """Typed float32 pointer to a contiguous CPU tensor's buffer."""
    return F32Ptr(unsafe_from_address=Int(py=t.data_ptr()))


@always_inline
def _dptr(addr: PythonObject) raises -> F32Ptr:
    """Typed float32 pointer into GPU memory from a raw device virtual address."""
    return F32Ptr(unsafe_from_address=Int(py=addr))


@always_inline
def _atomic_add(p: F32Ptr, offset: Int, v: Float32):
    """Relaxed atomic `p[offset] += v` (lowers to the right CPU / GPU atomic)."""
    _ = Atomic.fetch_add[ordering=Ordering.RELAXED](p.unsafe_offset(offset), v)


@always_inline
def _add[atomic: Bool](p: F32Ptr, offset: Int, v: Float32):
    """`p[offset] += v`, atomically or not (comptime).

    Scatter kernels accumulate atomically when samples run concurrently on the
    same memory (GPU threads, unpartitioned CPU chunks). The CPU driver instead
    partitions samples so that concurrently running work items write disjoint
    regions, and then uses the plain add: a float `fetch_add` is a CAS loop on
    CPUs and roughly doubles the cost of a scatter under contention.
    """
    comptime if atomic:
        _atomic_add(p, offset, v)
    else:
        p[unsafe_offset=offset] += v


@always_inline
def _cubic_kernel(s_in: Float32) -> Float32:
    """Cubic convolution weight with coefficient `CUBIC_A`, support |s| < 2."""
    var s = -s_in if s_in < 0 else s_in
    var a = CUBIC_A
    if s < 1.0:
        return ((a + 2.0) * s - (a + 3.0)) * s * s + 1.0
    if s < 2.0:
        return ((a * s - 5.0 * a) * s + 8.0 * a) * s - 4.0 * a
    return 0.0


@always_inline
def _cubic_kernel_derivative(s_in: Float32) -> Float32:
    """d/ds of `_cubic_kernel`."""
    var sign: Float32 = -1.0 if s_in < 0 else 1.0
    var s = -s_in if s_in < 0 else s_in
    var a = CUBIC_A
    if s < 1.0:
        return sign * (3.0 * (a + 2.0) * s * s - 2.0 * (a + 3.0) * s)
    if s < 2.0:
        return sign * (3.0 * a * s * s - 10.0 * a * s + 8.0 * a)
    return 0.0
