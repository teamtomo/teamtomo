"""CPU entry points of the Mojo image-interpolation kernels (Python module).

Compiled on import via `mojo.importer` (see `_mojo_backend/_kernels.py`). The
per-sample math lives in `_interp.mojo` and is shared with the GPU module; here
each entry point reads its scalar parameters once (`_read_params`), picks the
`(ndim, interp)` specialisation, and runs the samples in parallel chunks with
`parallelize`. Gather kernels run in independent chunks of samples; scatter
kernels partition the samples by slab first so that no two concurrent work items
write the same memory (see "Atomic-free parallel scatter" below).

Every entry point takes `(bufs, params)`: `bufs` is a tuple of contiguous
float32 CPU tensors (see each docstring for the order) and `params` a
`KernelParams` NamedTuple read by field name. Outputs that are accumulated into
(gradient images, inserted images / weights) are prepared by the caller.
"""

from std.math import ceildiv, floor, round
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import num_physical_cores

from max.algorithm import parallelize

from _common import (
    CUBIC,
    LINEAR,
    NEAREST,
    F32Ptr,
    InterpParams,
    _ptr,
    _read_params,
)
from _interp import (
    _base_index,
    _insert_backward_one,
    _insert_one,
    _sample_backward_one,
    _sample_one,
    _taps,
)

# Samples per parallel work item. Small enough to balance 18+ cores on
# moderate `n`, large enough that the per-item dispatch cost is negligible.
comptime CHUNK = 1024
# Below this many samples the thread-pool hand-off costs more than it saves.
comptime SERIAL_MAX = 2048


@always_inline
def _run_chunked[
    body: def (Int) capturing
](n: Int):
    """Run `body(s)` for every sample `s` in `[0, n)`, in parallel chunks."""
    if n <= SERIAL_MAX:
        for s in range(n):
            body(s)
        return
    var n_chunks = ceildiv(n, CHUNK)

    @parameter
    def worker(ci: Int):
        var s0 = ci * CHUNK
        var s1 = min(n, s0 + CHUNK)
        for s in range(s0, s1):
            body(s)

    parallelize[worker](n_chunks, num_physical_cores())


# ---------------------------------------------------------------------------
# Atomic-free parallel scatter
#
# A scatter kernel adds each sample into up to T**ndim voxels. Running samples
# concurrently therefore needs atomics -- which on CPUs are CAS loops on float
# and, worse, make cores fight over the same cache lines. Instead the samples
# are bucketed by the SLAB of axis 0 their stencil starts in (a parallel counting
# sort), and slabs are then processed in two rounds -- even slabs, then odd
# slabs -- one work item per slab. A slab at least T voxels wide only ever
# receives writes from its own samples and from the slab below it, so within a
# round no two work items touch the same memory and the plain add is exact.
# ---------------------------------------------------------------------------


@always_inline
def _slab_of[
    ndim: Int, interp: Int
](coords: F32Ptr, s: Int, p: InterpParams, slab_w: Int) -> Int:
    """Slab index of sample `s` along axis 0, or -1 if it lies outside the image."""
    var x = coords[unsafe_offset=s * ndim]
    var d0 = p.dim(0)
    if x < 0.0 or x > Float32(d0 - 1):
        return -1
    var b = _base_index[interp](x)
    if b < 0:
        b = 0
    return b // slab_w


struct Partition:
    """Samples grouped by slab: `order[start[k] : start[k + 1]]` lists slab k's samples."""

    var n_slabs: Int
    var slab_w: Int
    var start: List[Int]
    var order: List[Int]

    def __init__(out self, n_slabs: Int, slab_w: Int, n: Int):
        self.n_slabs = n_slabs
        self.slab_w = slab_w
        self.start = List[Int](length=n_slabs + 1, fill=0)
        self.order = List[Int](length=n, fill=0)


def _partition[
    ndim: Int, interp: Int
](coords: F32Ptr, p: InterpParams, n_workers: Int) -> Partition:
    """Counting-sort the samples by slab (parallel over chunks of samples)."""
    comptime T = _taps[interp]()
    var d0 = p.dim(0)
    # ~4 slabs per worker and round so the rounds stay balanced; never narrower
    # than the stencil so a sample only reaches into the next slab up
    var slab_w = max(T, ceildiv(d0, 8 * n_workers))
    var n_slabs = ceildiv(d0, slab_w)
    var part = Partition(n_slabs, slab_w, p.n)
    var n_chunks = ceildiv(p.n, CHUNK)
    var hist = List[Int](length=n_chunks * n_slabs, fill=0)

    # per-chunk histograms
    @parameter
    def count(ci: Int):
        var s1 = min(p.n, (ci + 1) * CHUNK)
        for s in range(ci * CHUNK, s1):
            var k = _slab_of[ndim, interp](coords, s, p, slab_w)
            if k >= 0:
                hist[ci * n_slabs + k] += 1

    parallelize[count](n_chunks, n_workers)

    # exclusive prefix over (slab, chunk); hist becomes each chunk's write cursor
    var total = 0
    for k in range(n_slabs):
        part.start[k] = total
        for ci in range(n_chunks):
            var c = hist[ci * n_slabs + k]
            hist[ci * n_slabs + k] = total
            total += c
    part.start[n_slabs] = total

    @parameter
    def place(ci: Int):
        var s1 = min(p.n, (ci + 1) * CHUNK)
        for s in range(ci * CHUNK, s1):
            var k = _slab_of[ndim, interp](coords, s, p, slab_w)
            if k >= 0:
                var at = hist[ci * n_slabs + k]
                part.order[at] = s
                hist[ci * n_slabs + k] = at + 1

    parallelize[place](n_chunks, n_workers)
    return part^


@always_inline
def _run_slabs[
    body: def (Int) capturing
](part: Partition, parity: Int, n_workers: Int):
    """Run `body(s)` for the samples of every slab with the given parity, in parallel.

    (`parity` is an argument rather than a captured loop variable: a
    `@parameter` closure does not capture a `for` induction variable reliably.)
    """
    var n_items = (part.n_slabs - parity + 1) // 2

    @parameter
    def worker(i: Int):
        var k = 2 * i + parity
        for j in range(part.start[k], part.start[k + 1]):
            body(part.order[j])

    parallelize[worker](n_items, n_workers)


@always_inline
def _run_partitioned[
    body: def (Int) capturing
](part: Partition, n_workers: Int):
    """Run `body(s)` for every bucketed sample, even slabs first, then odd slabs."""
    _run_slabs[body](part, 0, n_workers)
    _run_slabs[body](part, 1, n_workers)


# ---------------------------------------------------------------------------
# sample_forward: bufs = (image, coords, out)
# ---------------------------------------------------------------------------


def _sample_forward_cpu[
    ndim: Int, interp: Int
](img: F32Ptr, coords: F32Ptr, dst: F32Ptr, p: InterpParams):
    @parameter
    def body(s: Int):
        _sample_one[ndim, interp](img, coords, dst, s, p)

    _run_chunked[body](p.n)


def sample_forward(bufs: PythonObject, params_obj: PythonObject) raises -> PythonObject:
    """Interpolate: samples = image[coords]; bufs = (image, coords, samples)."""
    var p = _read_params(params_obj)
    var img = _ptr(bufs[0])
    var coords = _ptr(bufs[1])
    var samples = _ptr(bufs[2])
    if p.ndim == 3:
        if p.interp == NEAREST:
            _sample_forward_cpu[3, NEAREST](img, coords, samples, p)
        elif p.interp == LINEAR:
            _sample_forward_cpu[3, LINEAR](img, coords, samples, p)
        else:
            _sample_forward_cpu[3, CUBIC](img, coords, samples, p)
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _sample_forward_cpu[2, NEAREST](img, coords, samples, p)
        elif p.interp == LINEAR:
            _sample_forward_cpu[2, LINEAR](img, coords, samples, p)
        else:
            _sample_forward_cpu[2, CUBIC](img, coords, samples, p)
    else:
        if p.interp == NEAREST:
            _sample_forward_cpu[1, NEAREST](img, coords, samples, p)
        elif p.interp == LINEAR:
            _sample_forward_cpu[1, LINEAR](img, coords, samples, p)
        else:
            _sample_forward_cpu[1, CUBIC](img, coords, samples, p)
    return PythonObject(0)


# ---------------------------------------------------------------------------
# sample_backward: bufs = (image, coords, grad_samples, grad_image, grad_coords)
# ---------------------------------------------------------------------------


def _sample_backward_cpu[
    ndim: Int, interp: Int
](
    img: F32Ptr,
    coords: F32Ptr,
    gout: F32Ptr,
    gimg: F32Ptr,
    gcoords: F32Ptr,
    p: InterpParams,
):
    @parameter
    def body(s: Int):
        _sample_backward_one[ndim, interp, atomic=False](
            img, coords, gout, gimg, gcoords, s, p
        )

    if p.n <= SERIAL_MAX:
        for s in range(p.n):
            body(s)
        return
    if p.need_grad_image == 0:
        _run_chunked[body](p.n)  # grad_coords only: every sample owns its slot
        return
    # samples outside the image are not visited: their grad_coords stay at the
    # caller's zero initialisation
    var workers = num_physical_cores()
    var part = _partition[ndim, interp](coords, p, workers)
    _run_partitioned[body](part, workers)


def sample_backward(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Adjoint of `sample_forward`.

    bufs = (image, coords, grad_samples, grad_image, grad_coords). `grad_image`
    is accumulated into (pre-zeroed by the caller) when `need_grad_image`;
    `grad_coords` is written when `need_grad_coords`. Unused outputs may be any
    valid buffer.
    """
    var p = _read_params(params_obj)
    var img = _ptr(bufs[0])
    var coords = _ptr(bufs[1])
    var gout = _ptr(bufs[2])
    var gimg = _ptr(bufs[3])
    var gcoords = _ptr(bufs[4])
    if p.ndim == 3:
        if p.interp == NEAREST:
            _sample_backward_cpu[3, NEAREST](img, coords, gout, gimg, gcoords, p)
        elif p.interp == LINEAR:
            _sample_backward_cpu[3, LINEAR](img, coords, gout, gimg, gcoords, p)
        else:
            _sample_backward_cpu[3, CUBIC](img, coords, gout, gimg, gcoords, p)
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _sample_backward_cpu[2, NEAREST](img, coords, gout, gimg, gcoords, p)
        elif p.interp == LINEAR:
            _sample_backward_cpu[2, LINEAR](img, coords, gout, gimg, gcoords, p)
        else:
            _sample_backward_cpu[2, CUBIC](img, coords, gout, gimg, gcoords, p)
    else:
        if p.interp == NEAREST:
            _sample_backward_cpu[1, NEAREST](img, coords, gout, gimg, gcoords, p)
        elif p.interp == LINEAR:
            _sample_backward_cpu[1, LINEAR](img, coords, gout, gimg, gcoords, p)
        else:
            _sample_backward_cpu[1, CUBIC](img, coords, gout, gimg, gcoords, p)
    return PythonObject(0)


# ---------------------------------------------------------------------------
# insert_forward: bufs = (values, coords, image, weights)
# ---------------------------------------------------------------------------


def _insert_forward_cpu[
    ndim: Int, interp: Int
](values: F32Ptr, coords: F32Ptr, img: F32Ptr, wimg: F32Ptr, p: InterpParams):
    @parameter
    def body(s: Int):
        _insert_one[ndim, interp, atomic=False](values, coords, img, wimg, s, p)

    if p.n <= SERIAL_MAX:
        for s in range(p.n):
            body(s)
        return
    var workers = num_physical_cores()
    var part = _partition[ndim, interp](coords, p, workers)
    _run_partitioned[body](part, workers)


def insert_forward(bufs: PythonObject, params_obj: PythonObject) raises -> PythonObject:
    """Splat: image += splat(values); weights += splat(1). bufs = (values, coords, image, weights).

    `image` and `weights` are accumulated into IN PLACE (they may already hold
    data). `weights` is only touched when `has_weights`.
    """
    var p = _read_params(params_obj)
    var values = _ptr(bufs[0])
    var coords = _ptr(bufs[1])
    var img = _ptr(bufs[2])
    var wimg = _ptr(bufs[3])
    if p.ndim == 3:
        if p.interp == NEAREST:
            _insert_forward_cpu[3, NEAREST](values, coords, img, wimg, p)
        elif p.interp == LINEAR:
            _insert_forward_cpu[3, LINEAR](values, coords, img, wimg, p)
        else:
            _insert_forward_cpu[3, CUBIC](values, coords, img, wimg, p)
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _insert_forward_cpu[2, NEAREST](values, coords, img, wimg, p)
        elif p.interp == LINEAR:
            _insert_forward_cpu[2, LINEAR](values, coords, img, wimg, p)
        else:
            _insert_forward_cpu[2, CUBIC](values, coords, img, wimg, p)
    else:
        if p.interp == NEAREST:
            _insert_forward_cpu[1, NEAREST](values, coords, img, wimg, p)
        elif p.interp == LINEAR:
            _insert_forward_cpu[1, LINEAR](values, coords, img, wimg, p)
        else:
            _insert_forward_cpu[1, CUBIC](values, coords, img, wimg, p)
    return PythonObject(0)


# ---------------------------------------------------------------------------
# insert_backward:
#   bufs = (values, coords, grad_image, grad_weights, grad_values, grad_coords)
# ---------------------------------------------------------------------------


def _insert_backward_cpu[
    ndim: Int, interp: Int
](
    values: F32Ptr,
    coords: F32Ptr,
    gimg: F32Ptr,
    gwimg: F32Ptr,
    gvalues: F32Ptr,
    gcoords: F32Ptr,
    p: InterpParams,
):
    @parameter
    def body(s: Int):
        _insert_backward_one[ndim, interp](
            values, coords, gimg, gwimg, gvalues, gcoords, s, p
        )

    _run_chunked[body](p.n)


def insert_backward(
    bufs: PythonObject, params_obj: PythonObject
) raises -> PythonObject:
    """Adjoint of `insert_forward`.

    bufs = (values, coords, grad_image, grad_weights, grad_values, grad_coords).
    `grad_values` is written when `need_grad_values`, `grad_coords` when
    `need_grad_coords` (using `grad_weights` too when `has_grad_weights`).
    """
    var p = _read_params(params_obj)
    var values = _ptr(bufs[0])
    var coords = _ptr(bufs[1])
    var gimg = _ptr(bufs[2])
    var gwimg = _ptr(bufs[3])
    var gvalues = _ptr(bufs[4])
    var gcoords = _ptr(bufs[5])
    if p.ndim == 3:
        if p.interp == NEAREST:
            _insert_backward_cpu[3, NEAREST](values, coords, gimg, gwimg, gvalues, gcoords, p)
        elif p.interp == LINEAR:
            _insert_backward_cpu[3, LINEAR](values, coords, gimg, gwimg, gvalues, gcoords, p)
        else:
            _insert_backward_cpu[3, CUBIC](values, coords, gimg, gwimg, gvalues, gcoords, p)
    elif p.ndim == 2:
        if p.interp == NEAREST:
            _insert_backward_cpu[2, NEAREST](values, coords, gimg, gwimg, gvalues, gcoords, p)
        elif p.interp == LINEAR:
            _insert_backward_cpu[2, LINEAR](values, coords, gimg, gwimg, gvalues, gcoords, p)
        else:
            _insert_backward_cpu[2, CUBIC](values, coords, gimg, gwimg, gvalues, gcoords, p)
    else:
        if p.interp == NEAREST:
            _insert_backward_cpu[1, NEAREST](values, coords, gimg, gwimg, gvalues, gcoords, p)
        elif p.interp == LINEAR:
            _insert_backward_cpu[1, LINEAR](values, coords, gimg, gwimg, gvalues, gcoords, p)
        else:
            _insert_backward_cpu[1, CUBIC](values, coords, gimg, gwimg, gvalues, gcoords, p)
    return PythonObject(0)


@export
def PyInit_image_interpolation_cpu() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("image_interpolation_cpu")
        m.def_function[sample_forward](
            "sample_forward", docstring="Interpolate an image at coordinates (CPU)."
        )
        m.def_function[sample_backward](
            "sample_backward", docstring="Adjoint of sample_forward (CPU)."
        )
        m.def_function[insert_forward](
            "insert_forward", docstring="Splat values into an image (CPU)."
        )
        m.def_function[insert_backward](
            "insert_backward", docstring="Adjoint of insert_forward (CPU)."
        )
        return m.finalize()
    except e:
        abort(String("failed to create image_interpolation_cpu module: ", e))
