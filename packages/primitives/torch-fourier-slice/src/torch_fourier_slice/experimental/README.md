# torch_fourier_slice.experimental

Experimental Mojo-backed kernels: the same Fourier-slice operators as the
rest of `torch_fourier_slice`, with their compute kernels written in
[Mojo](https://www.modular.com/mojo) (1.0.0b2) and exposed to Python via
Mojo's Python interop.

**APIs here are experimental and may change without notice.**

## Status

| Op | Geometry | Device | Interp | Grad |
|----|----------|--------|--------|------|
| `project_3d_to_2d_forw` | 3D→2D forward (central slice) | CPU + GPU | linear, cubic | ✓ volume, rotations, shifts |
| `backproject_2d_to_3d_forw` | 2D→3D adjoint (reconstruction) | CPU + GPU | linear, cubic | ✓ projections, weights, rotations, shifts |

Interpolation is selected with ``interpolation="linear"`` (trilinear, default) or
``"cubic"`` (tricubic Catmull-Rom) on both functions.

The 2D→3D scatter is parallelized with atomic adds (Mojo `parallelize` on CPU,
one thread per input rfft pixel on GPU); the device follows the input tensor.

Importing `torch_fourier_slice.experimental` eagerly compiles and loads all Mojo
kernel modules (via `mojo.importer`); `mojo_kernels_available()` reports success.

### Differentiability

Both ops are **fully differentiable**: the forward
projection w.r.t. `reconstruction`, `rotations`, `shifts`; the backprojection
w.r.t. `projections`, `weights`, `rotations`, `shifts`.

The *data* gradients use the adjoint relationship — the two ops are adjoints, so
each one's data backward is the other's kernel: `d/d(volume)` of the projection
is the scatter (pure adjoint), and `d/d(projections)` of the backprojection is
the forward projection (with an exact correction for the Hermitian double-insert
and the skipped x=0 line). These predict a linear loss change exactly (in-band;
near the Nyquist boundary the forward *clamps* and the scatter *drops*
out-of-range samples).

The `rotations`/`shifts`/`weights` gradients are dedicated backward kernels
(`_pose_grad.mojo`): the rotation grad chains the **analytical spatial gradient**
of the interpolated field (`_gather_grad.mojo`, trilinear differences / tricubic
`cubic_kernel_derivative`) through the rotated sample coordinate; the shift grad
differentiates the phase ramp; the weight grad is the exact adjoint of the weight
splat. The backprojection variants gather with *drop* boundary handling (the
scatter's adjoint) rather than the forward gather's *clamp*. All are validated by
finite differences (CPU and GPU, both interpolations).

```python
import torch
from torch_fourier_slice.experimental import (
    project_3d_to_2d, backproject_2d_to_3d_forw,
)

volume_rfft.requires_grad_(True)
projections = project_3d_to_2d(volume_rfft, rotation_matrices)
loss = (projections - target).abs().pow(2).sum()
loss.backward()  # volume_rfft.grad is populated

data_vol, weight_vol = backproject_2d_to_3d_forw(projections, rotation_matrices, weights=ctf2)
```

**The backend follows the input tensor's device** — there is no `backend`
argument. A CPU `reconstruction` runs the CPU kernel; an `mps`/`cuda`
`reconstruction` runs the GPU kernel. Output is returned on the input's device.

```python
from torch_fourier_slice.experimental import project_3d_to_2d

imgs = project_3d_to_2d(volume_rfft, rotations)  # CPU -> CPU kernel
imgs = project_3d_to_2d(volume_rfft.to("mps"), rotations)  # MPS -> GPU kernel
```

The kernels form a single extension module (`_mojo/projectors.mojo`, compiled on
first import) whose per-pixel math is written once and shared by the CPU and GPU
paths. It is split into grouped files imported by the entry module:

| file | contents |
|------|----------|
| `_common.mojo` | types/constants, `FourierSliceParams`, small geometry/complex helpers |
| `_gather.mojo` | sample + interpolate (forward) — linear & cubic |
| `_gather_grad.mojo` | interpolate + analytical spatial gradient — linear & cubic |
| `_scatter.mojo` | atomic accumulate + splat (backward) — linear & cubic |
| `_pixel.mojo` | the per-pixel forward/scatter ops shared by CPU loops and GPU threads |
| `_pose_grad.mojo` | per-pixel rotation/shift/weight gradient ops (shared CPU/GPU) |
| `_device.mojo` | GPU host↔device transfer, kernels, launchers |
| `projectors.mojo` | the Python-facing entry points + `FourierSliceParams` construction |

Naming: integer **volume voxel** indices are `z, y, x`; projection **pixel**
indices are `i_h, i_w`; batch axes are `i_bv, i_bp`; the continuous rotated
sample coordinate is `kz, ky, kx`. The complex Fourier volume/projection data is
read and written through a per-volume 4D `TileTensor` view `[d, h, w, 2]` over the
zero-copy pointer (`_load_c2` / `_store_c2` in `_common.mojo`); atomic scatter
accumulation and the real weight bookkeeping stay on raw pointers.

> Editing a non-entry `.mojo` file may not invalidate the `__mojocache__` (the
> import hook tracks `projectors.mojo`), so clear `_mojo/__mojocache__/` after
> changing a helper file. It is always safe to delete.

**CPU** runs one thread per projection via Mojo's `parallelize`
(`num_physical_cores()` workers); poses are independent and write disjoint
output blocks (scatter uses atomic adds).

**GPU** runs one thread per rfft pixel via Mojo's `DeviceContext`; works on any
GPU Mojo supports (NVIDIA / AMD / Apple),
developed and validated on Apple Silicon. (The kernel currently manages its own
device memory and bridges through the host, so a GPU input is materialised on
the CPU for the upload and the result is moved back to the input device; output
matches the CPU path bit-exactly without shifts, ~1e-5 with shifts due to GPU
transcendental precision.)

For projecting one volume at many orientations, the upload is **cached
transparently** — keyed by the volume tensor's identity + version — and reused
when you pass the same tensor again, removing the otherwise per-call transfer:

```python
imgs_a = project_3d_to_2d_forw(volume_rfft.to("mps"), rotations_a)  # uploads
imgs_b = project_3d_to_2d_forw(volume_rfft.to("mps"), rotations_b)  # reused
```

Note `.to("mps")` returns a *new* tensor each call (different identity), so cache
it once: `vol = volume_rfft.to("mps")` then reuse `vol`. The cache invalidates on
in-place edits (the tensor's `_version` bumps) and frees the device buffer when
the source tensor is garbage-collected. `clear_resident_volume_cache()` frees
eagerly; `reuse_volume=False` disables caching (re-upload every call).

### Performance note (Apple M4 Pro, box 256, validated bit-exact)

| N projections | CPU (14 threads) | GPU stateless | GPU resident |
|---------------|------------------|---------------|--------------|
| 10            | 0.5 ms           | 15.3 ms       | 3.2 ms       |
| 100           | 8.2 ms           | 36.2 ms       | 23.1 ms      |
| 1000          | 77.7 ms          | 258.9 ms      | 247.5 ms     |

The volume-resident path removes a constant ~13 ms upload vs stateless. The GPU
is output-transfer bound here (the per-pixel compute is light and results are
copied back to host every call), so the multithreaded CPU is fastest on this
hardware; the GPU is expected to win when results stay on-device for downstream
GPU work or with heavier per-pixel compute (e.g. cubic interpolation).

### Future optimizations (scatter write contention)

The 2D→3D scatter accumulates with atomics (`_atomic_add_at`). Contention is
*structured*, which makes targeted optimizations possible — none implemented yet,
profile first (the GPU path is currently host-transfer bound, not atomic bound):

- **Low-k is hot, ~1/k.** Every central slice passes through the Fourier origin,
  so a voxel at radius `k` receives ~`P·0.5/k` writes (`P` projections) — singular
  at DC, negligible at high `k`. Contention lives in a small, known core, so a
  cheap fix can special-case it and plain-atomic the bulk.
- **Adjacent lanes collide deterministically.** A warp of consecutive threads is
  consecutive `i_w` in one slice row (same rotation); their interpolation stencils
  overlap (≈half at oversampling 1, less as oversampling grows). So the colliding
  lanes are the *adjacent* ones — warp/stencil pre-aggregation (`warp.sum` → one
  atomic) can cut atomics without runtime `match`/ballot.
- **Replicated reconstructions (memory-budgeted).** Partition projections into
  `G` groups, scatter each into its own volume, sum at the end. Memory is
  `G × bv × volume`, so `G` is a budget knob, not a constant. Key point: a batch
  of `bv` volumes multiplies *memory* but not *contention* — different volumes are
  different buffers and never contend. So the regimes align favorably: with many
  volumes (`bv ≥ workers`) skip replication and partition *volumes* across workers
  for a contention-free scatter at no extra memory; with few volumes (esp.
  `bv = 1`) replication is both most needed and cheapest (`G × volume`). For the
  tight middle and GPU-with-large-`bv`, replicate only the hot low-`k` core (a
  small cube around DC): `bv × volume` shared + `G × bv × tiny`, getting the relief
  exactly where ~`1/k` says it lives for almost no memory. Pick
  `G = clamp(free_mem / (bv · replica_bytes), 1, workers)`, degrading to plain
  atomics when even `G = 2` won't fit. Composes with the two above.

The three above reorder the float32 summation, trading the current exact CPU/GPU
bit-match for ~1e-6 drift.

A fourth, more invasive option is **loop inversion (gather backprojection)** —
contention-free *and* deterministic, but a different operator:

- Make the backprojection **voxel-parallel** instead of pixel-parallel: each
  thread owns one output voxel, finds the projection pixels whose central slice
  passes within interpolation support of it (`|v·n_p| ≤ support`), and sums them.
  One write per voxel → **no atomics, no replication memory, fixed reduction order
  → bit-reproducible**. (It mirrors how the forward projector already gathers.)
- Cost 1 — *finding contributors*: the set of projections hitting voxel `v` is the
  same ~`P/k` (1/k geometry), but a naive per-voxel scan over all `P` is
  `O(P·K³)`, a factor ~`K` worse than the scatter's `O(P·K²)`. It needs an
  orientation index / binning (test only the band of normals ⊥ `v`) to match the
  scatter's work.
- Cost 2 — *kernel shape*: the volume-grid interp kernel is axis-aligned, not
  plane-separable, so the gather must sum the actual pixels whose 3D sample lands
  in `v`'s stencil (careful bookkeeping rather than a clean 2D interp).
- Caveat — *it moves the scatter, doesn't remove it*: there's a "conservation of
  scatter" — exactly one of {forward, adjoint} is a clean gather and the other a
  scatter. A gather backprojection's adjoint scatters into the **projections**
  (2D, cooler) instead of the **volume** (3D, low-`k`-hot), which is a net win, but
  it means re-establishing the adjoint pair (new matching forward + tests) rather
  than dropping into the existing scatter. Most appealing on GPU for occupancy +
  determinism, once the indexing is solved.

## Requirements

These kernels need the optional `mojo` package:

```bash
pip install "mojo==1.0.0b2" --prerelease allow
# or: uv add "mojo==1.0.0b2" --prerelease allow
```

Probe support at runtime with `mojo_kernels_available()`. The Mojo source in
`_mojo/projectors.mojo` is JIT-compiled on first use (cached under
`_mojo/__mojocache__/`, gitignored) via `mojo.importer`.

## How it works

The Python wrapper (`project.py`) prepares contiguous CPU tensors, viewing
complex tensors as interleaved real (`torch.view_as_real`), and passes their raw
`data_ptr()` addresses to the Mojo kernel. The kernel reconstructs typed
`UnsafePointer`s and runs the projection loop in native Mojo — **zero-copy**, no
data marshalling. All shapes/scalars are extracted from Python once up front so
the hot loops never call back into Python.

## Layout note

`project_3d_to_2d_forw` operates on volumes in **rfft layout with DC at the
origin** (`[..., 0, 0, 0]`, unshifted). This is
the *same operation* as `torch_fourier_slice.extract_central_slices_rfft_3d`,
which uses an `fftshift`ed rfft layout (DC centered on the z/y axes). Bridge
between them with a single `fftshift`/`ifftshift` over the non-redundant dims:

```python
import torch
from torch_fourier_slice.experimental import project_3d_to_2d

# volume_rfft is in teamtomo (fftshifted) layout -> convert to DC-at-origin rfft
rfft = torch.fft.ifftshift(volume_rfft, dim=(-3, -2))
proj_rfft = project_3d_to_2d(rfft, rotation_matrices)
# back to teamtomo layout if desired
proj = torch.fft.fftshift(proj_rfft, dim=-2)
```

Within the Nyquist band the two produce identical results (bit-exact for
non-interpolating rotations); they differ only in how out-of-Nyquist corner
samples are extrapolated (the canonical kernel zero-pads, these kernels
clamp to the edge voxel).
