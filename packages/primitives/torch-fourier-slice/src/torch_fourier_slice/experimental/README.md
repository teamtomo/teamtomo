# `torch_fourier_slice.experimental` — Mojo kernels

Fourier-space **extraction** (3D→2D central-slice) and **insertion** (2D→3D
adjoint) for cryo-EM/ET, with the compute kernels written in
[Mojo](https://www.modular.com/mojo) and called from PyTorch. These are the same
operators as the rest of `torch_fourier_slice`, moved into Mojo.

**Experimental — APIs may change without notice.**

This README is written for a **PyTorch/Python programmer with some interest in
HPC** who has not written GPU kernels before. It explains what the backend does,
how the CPU and GPU paths relate, how data crosses the Python↔Mojo boundary, and
why shipping *Mojo source that compiles on import* beats shipping per-platform
CUDA/Metal binaries.

---

## 1. The one idea to take away

There is **one numeric core** — the Fourier-slice math — written once, in Mojo.
It runs in three places from that single source:

```
                         ┌───────────────────────────┐
                         │  per-pixel numeric core    │   _pixel / _gather /
                         │  (Mojo, written ONCE)      │   _scatter / _pose_grad
                         └────────────┬──────────────┘
             ┌────────────────────────┼─
             ▼                        ▼            
   CPU: one thread/pose      GPU (CUDA/AMD/Apple Silicon): one thread
   `parallelize` over poses  per rfft pixel        
             │                        │            
   reads host memory         reads torch DEVICE memory in place (zero-copy)
```

You do **not** maintain a `.cpp`, `.cu` file *and* a `.metal` file *and* a CPU fallback.
You maintain the math once; Mojo lowers it to CPU SIMD, NVIDIA PTX, and Apple
Metal from the same code. That is the whole pitch, and §8 explains why it matters
for distribution.

---

## 2. Quick start

The backend **follows the input tensor's device**. 
A CPU tensor runs the CPU kernel; an `mps`/`cuda` tensor runs the GPU
kernel; the output comes back on the input's device.

There are two layers. Reach for the **real-space** one unless you have a reason
not to — it handles padding, the FFTs and the gridding correction (§7) for you:

```python
import torch
from torch_fourier_slice.experimental import (
    project_3d_to_2d, backproject_2d_to_3d, mojo_kernels_available,
)

assert mojo_kernels_available()          # False if the `mojo` package is missing

volume = ...                             # real (d, d, d), even side
rotations = ...                          # (bp, 3, 3) zyx rotation matrices

images = project_3d_to_2d(volume, rotations)               # CPU  -> CPU kernel
images = project_3d_to_2d(volume.to("cuda"), rotations)    # CUDA -> GPU kernel
recon  = backproject_2d_to_3d(images, rotations)           # the adjoint, real out

# everything is differentiable (see §11):
volume.requires_grad_(True)
loss = (project_3d_to_2d(volume, rotations) - target).pow(2).sum()
loss.backward()                          # volume.grad populated
```

The **Fourier** layer is the same operators with the transforms left to you —
rfft in, rfft out, DC at the origin. Use it when you are already working in
Fourier space and don't want a round trip per call:

```python
from torch_fourier_slice.experimental import extract_central_slices_rfft_3d

volume_rfft = torch.fft.rfftn(torch.fft.fftshift(volume, dim=(-3, -2, -1)),
                              dim=(-3, -2, -1))
slices = extract_central_slices_rfft_3d(volume_rfft.contiguous(), rotations)
```

Interpolation: `interpolation="linear"` (trilinear, default) or `"cubic"`
(tricubic Catmull-Rom) — selected at **compile time** per call (§6). The
real-space layer's gridding correction follows this choice (§7).

---

## 3. What "projection" means here (30 seconds of theory)

By the **Fourier-slice theorem**, a 2D projection image of a 3D volume equals a
central planar slice through the volume's 3D Fourier transform, oriented by the
projection direction. So:

- **Forward (project):** for each 2D output pixel, rotate its frequency
  coordinate into the volume, **sample+interpolate** the 3D rfft volume there,
  apply any shift phase. This is a **gather** (each output reads a few inputs).
- **Backward (backproject):** for each 2D input pixel, rotate into the volume and
  **splat+accumulate** its value into the nearby voxels. This is a **scatter**
  (each input writes a few outputs, hence atomics).

Everything is done in **rfft layout with DC at the origin** (real volume →
`torch.fft.rfftn`, non-redundant half, complex stored as a trailing `2`
dimension via `torch.view_as_real`). See §10 and the layout note at the end.

The **central-line** ops apply the same theorem once more: a line through the
origin of a central slice is a line through the origin of the 3D transform, so a
1D central line is the degenerate central slice whose in-plane axis has collapsed
to its DC row. A line is therefore posed by a bare **direction** rather than a
rotation matrix — rotating about the line's own axis is a gauge its values are
blind to. Same gather/scatter kernels, one dimension lower.

---

## 4. Repository map

Python side (`experimental/`):

| file | role |
|------|------|
| `project.py`, `backproject.py` | **real-space API**: `project_3d_to_2d` / `backproject_2d_to_3d` (+ `_multivolume`) — padding, FFTs, gridding correction |
| `slice_extraction.py`, `slice_insertion.py` | Fourier API: 3D volume ↔ 2D central slices, posed by a rotation matrix |
| `line_extraction.py`, `line_insertion.py` | Fourier API: 3D volume ↔ 1D central lines, posed by a direction |
| `line_extraction_2d.py`, `line_insertion_2d.py` | Fourier API: 2D image ↔ 1D central lines |
| `_gridding.py` | the de-apodization correction for each interpolation kernel (§7) |
| `_autograd.py` | torch `autograd.Function`s wiring forward/backward |
| `_ops.py` | orchestration: validate → build buffers + `KernelParams` → call a kernel |
| `_validation.py` | shape checks, `prep_*`, `interp_code`, the **`KernelParams`** carrier |
| `_gpu.py` | device addresses (CUDA VA / Metal `gpuAddress`), Metal heap residency, launch prep |
| `_kernels.py` | compiles + loads the Mojo module on import; `mojo_kernels_available()` |

Mojo kernels (`experimental/_mojo/`) — one extension module split into grouped
files, all compiled together on first import:

| file | contents |
|------|----------|
| `_common.mojo` | shared types & constants: `Float32Ptr`, `FourierSliceParams`, the per-kernel **buffer structs**, `C2`/`C8` complex SIMD, `LINEAR`/`CUBIC`, geometry/complex helpers, `_ptr`/`_dptr` |
| `_gather.mojo` | **sample + interpolate** the rfft volume (extraction) — linear & cubic |
| `_gather_grad.mojo` | interpolate **+ analytical spatial gradient** (for pose grads) |
| `_scatter.mojo` | **atomic accumulate + splat** into the rfft volume (insertion) |
| `_pixel.mojo` | `_project_pixel` / `_scatter_pixel` — the **per-output-element op**, shared by CPU loops and GPU threads |
| `_pose_grad.mojo` | per-pixel rotation/shift/weight gradient ops (shared CPU/GPU) |
| `_line.mojo`, `_line_grad.mojo` | the 3D↔1D central-line per-pixel ops and their direction/shift/weight gradients |
| `_line2d.mojo`, `_line2d_grad.mojo` | the same, one dimension lower (2D image ↔ 1D line) |
| `_device.mojo` | GPU kernels (one thread per pixel) + their launchers |
| `fourier_slice_kernels.mojo` | Python-facing entry points (CPU + GPU) + `PyInit_fourier_slice_kernels` + `DeviceSession` |

**The numeric core is `_gather` / `_scatter` / `_pixel` / `_pose_grad` (plus the
`_line*` analogues).** Those files never mention CPU or GPU — they are just math
over pointers and indices. `fourier_slice_kernels.mojo` (CPU) and `_device.mojo`
(GPU) are the two *drivers* that call into that core.

---

## 5. CPU vs GPU: same math, different execution strategy

**What they share:** every per-pixel computation. Both paths ultimately call the
*same* `_project_pixel[interp](...)` / `_scatter_pixel[interp](...)`. If you fix a
bug in the interpolation, both devices get the fix — there is no second copy.

**What differs is the parallelism model and where memory lives:**

| | CPU (`fourier_slice_kernels.mojo`) | GPU (`_device.mojo`) |
|---|---|---|
| parallel unit | **one thread per projection (pose)** via Mojo `parallelize`, `num_physical_cores()` workers; each worker loops over that pose's pixels | **one thread per rfft output pixel** via a `DeviceContext` kernel launch (`grid_dim × block_dim`) |
| memory | host tensors, pointer from `data_ptr()` (`_ptr`) | torch **device** tensors, read/written **in place** via raw device addresses (`_dptr`) — no host round-trip |
| scatter safety | atomic adds (poses write overlapping voxels) | atomic adds (adjacent pixels write overlapping voxels) |
| params on device | `FourierSliceParams` used directly | `FourierSliceParams` isn't `DevicePassable`, so the kernel takes its primitive fields as scalars and **rebuilds** it on-device |

Why thread-per-pose on CPU but thread-per-pixel on GPU? A CPU has ~10s of fat
cores that like coarse, cache-friendly chunks of work (a whole pose); a GPU has
~1000s of thin lanes that want the finest independent unit (one pixel) to stay
occupied. Same math, mapped to the hardware's grain.

---

## 6. Compile-time interpolation (`comptime`)

`interpolation="linear"|"cubic"` is resolved **at compile time**, not per voxel.
The interpolation kind is a Mojo `comptime` parameter threaded through the core
(`_interp3d[interp]`, `_project_pixel[interp]`, the kernels). At the Python→Mojo
boundary the runtime code (`KernelParams.interp`, `0`/`1`) is read **once** and
dispatched to the specialized build:

```mojo
if p.interp == CUBIC: _launch_project[CUBIC](...)   # a kernel with NO interp branch,
else:                 _launch_project[LINEAR](...)  # cubic/linear baked in
```

So the hot loop over millions of voxels never asks "linear or cubic?" — that
decision was compiled away. This is a small taste of the bigger Mojo idea:
parameters you know at build time become *specializations*, not runtime branches.

---

## 7. Gridding correction (why the real-space layer divides by `K`)

Sampling the rfft volume at a non-integer coordinate is a **convolution with the
interpolation kernel** `k`. A convolution in Fourier space is a multiplication in
real space by `K`, the continuous Fourier transform of `k`. So an extracted
projection is really the projection of `volume × K`, and an inserted
reconstruction comes out as `reconstruction × K` — the volume is *apodized*,
progressively damped away from the origin.

The fix is to **divide by `K`**: the volume on the way in to an extraction, the
reconstruction on the way out of an insertion. Both directions divide; that is
what `_gridding.py` supplies and what `project.py` / `backproject.py` apply.

`K` depends on which kernel was used, so it has to follow `interpolation`:

| interpolation | kernel | `K(ν)` | `K(0.25)` |
|---|---|---|---|
| `"linear"` | tent | `sinc²(ν)` | 0.81 |
| `"cubic"` | Catmull-Rom (Keys, `a = −1/2`) | `sinc³(ν)·(3·sinc(ν) − 2cos(πν))` | 0.94 |

The cubic form is obtained by integrating the kernel directly; the test suite
re-derives it against numerical quadrature. It apodizes less than the tent, as
you'd expect of the more accurate interpolant — so applying `sinc²` to a cubic
projection *over*-corrects and throws away cubic's advantage entirely.

Both kernels are separable, so the 3D correction is the outer product of the
per-axis 1D transforms, not a function of the frequency magnitude `|ν|`.

Measured on an isotropic blob, whose analytic projection is the same at every
orientation and so is an exact ground truth (relative RMS error, 64³ box):

| | no correction | `sinc²` | matched `K` |
|---|---|---|---|
| linear | 7.8e-4 | — | **1.6e-4** |
| cubic  | 1.6e-5 | 7.9e-4 | **1.4e-5** |

Getting the direction wrong (multiplying rather than dividing) roughly doubles
the error instead of removing it.

---

## 8. Why Mojo-compiled-on-import beats shipping CUDA/Metal binaries

This is the part most relevant to a Python/torch author who has fought
`pip install`.

**The traditional way** to ship a GPU-accelerated Python package is to write the
kernels in CUDA C++ (`.cu`) — and, for Apple, again in Metal — plus C++/pybind
glue, then **precompile binaries** and publish wheels. The problem is the binary
matrix:

```
{CUDA 11.8, 12.1, 12.4, 12.6, 12.8, 13.0, …}   (must match the user's DRIVER)
        ×  {linux x86_64, aarch64, macOS arm64, win}   (manylinux, etc.)
        ×  {py3.9 … 3.13}   ×   ABI
```

That is why torch has a **CUDA-specific index URL** (`pip install torch
--index-url https://download.pytorch.org/whl/cu128`), and why picking the wrong
one silently fails: a wheel built for CUDA 13.0 won't initialize on a machine
whose driver only supports 12.8. (We hit exactly this during development — the
default wheel pulled `cu130`, the box driver capped at 12.8, and it reported
`cuda: False` until we forced the `cu128` build.) You, the package author, own
that whole build+test+publish matrix forever, and your users own the
index-URL-matching ritual.

**The Mojo way:** ship **source** (the `.mojo` text). On first `import`, Mojo's
importer runs `mojo build --emit shared-lib` and compiles the kernels **for the
exact machine they're on** — targeting whatever GPU is present (NVIDIA PTX / AMD
/ Apple Metal) and whatever CUDA toolkit is installed — then caches the `.so`
under `_mojo/__mojocache__/` (keyed by a source hash; gitignored). Concretely:

- **No binary matrix.** One text source, compiled locally to the actual target.
  No `cuXXX` wheels, no manylinux, no per-arch/per-Python builds to publish.
- **No driver/toolkit version roulette.** It compiles against the local toolchain,
  so it targets the CUDA that's actually there — no index-URL to match.
- **Portable from one codebase.** The *same* `.mojo` produces the CPU path, the
  Metal path, and the CUDA path. This package was developed on Apple Silicon and
  runs unchanged on an A100 — no `#ifdef __CUDACC__` fork.
- **Readable & hackable.** The "kernel" is Python-like Mojo you can open, read,
  and edit — not opaque `.cu`/PTX. A newcomer can change the interpolation or add
  an op and just re-import. (Contrast: patch a CUDA kernel → rebuild the wheel
  matrix → republish.)
- **One language across the host/device seam.** No Python↔C++↔CUDA FFI layers to
  keep in sync; host orchestration and device kernels are both Mojo.

**Costs, honestly:** the first import pays a one-time compile (seconds; cached
after); the `mojo` toolchain (`pip install "mojo==1.0.0b2" --prerelease allow`)
must be present; Mojo is early/beta; and on some setups the NVIDIA PTX assembler
path must be pointed at `ptxas`
(`export MODULAR_NVPTX_COMPILER_PATH=$(command -v ptxas)`) before the CUDA build
succeeds. For a research backend that wants to run on a laptop *and* a cloud
A100 without a release-engineering department, the trade is very favorable.

```bash
pip install "mojo==1.0.0b2" --prerelease allow    # or: uv add ...
python -c "from torch_fourier_slice.experimental import mojo_kernels_available as a; print(a())"
```

> Editing a **non-entry** `.mojo` helper may not invalidate `__mojocache__` (the
> import hook tracks `fourier_slice_kernels.mojo`), so delete `_mojo/__mojocache__/` after
> changing a helper. It's always safe to delete.

---

## 9. How data crosses the boundary (zero-copy)

All complex tensors are viewed as interleaved real (`torch.view_as_real`:
`complex64` → trailing dim `2`) and made contiguous, so the kernels see plain
`float32` buffers. **Shapes and scalars are read into Python once** and passed as
a named `KernelParams`, so the hot loops never call back into Python.

**CPU path** (`extract_central_slices_rfft_3d` / `insert_...`):

```
torch tensors ──(data_ptr, _ptr)──▶ Float32Ptr ──▶ parallelize workers ──▶ _project_pixel[interp]
KernelParams  ──(read by name)────▶ FourierSliceParams
```

**GPU path** — **no host round-trip**; the kernel reads/writes the memory backing
the torch device tensors directly:

```
place inputs + PRE-ZEROED outputs on device
        │
        ├─ device addresses:  CUDA → data_ptr() is already a device VA
        │                     Metal → data_ptr() is an MTLBuffer object ptr, so
        │                             recover the real VA = [MTLBuffer gpuAddress]
        │                             + storage_offset   (_gpu.py)
        ├─ stream address:    CUDA → torch's current stream (kernel enqueues on it,
        │                            so it's ordered with surrounding torch ops)
        │                     Metal → 0 (own DeviceContext stream; entry point syncs)
        ▼
Mojo entry point:  _dptr(addr) → build a per-kernel buffer struct (ProjectBuffers …)
                   dispatch _launch_project[LINEAR|CUBIC](ctx, buffers=…, params=…, stream=…)
                   → one GPU thread per rfft pixel → _project_pixel[interp]
```

Two device-specific wrinkles handled in `_gpu.py`:

- **Metal heap residency.** macOS evicts idle GPU heaps after ~1–1.5 s, and Mojo
  doesn't declare foreign (torch-owned) buffers to its command encoder — a kernel
  pointing at an evicted heap silently reads zeros. `revive_heaps()` touches each
  tensor with a tiny torch op right before dispatch to keep it resident.
- **Shared `DeviceContext`.** One process-wide `DeviceSession` holds a single
  `DeviceContext`; constructing one per call leaks the underlying Metal command
  queue and crashes long loops after ~1000 steps.

---

## 10. Naming conventions

Consistent across Python and Mojo:

| name | meaning |
|------|---------|
| `bv` | batch of **v**olumes |
| `bp` | batch of **p**rojections (poses) per volume |
| `bv_rot`, `bv_shift_2d`, `bv_shift_3d` | broadcast batch of rotations / 2D shift / 3D shift |
| `d, h, w` | volume axes (depth, height, width); cubic, so `d == h` |
| `sidelength` | cube edge (`= h = d`); `sidelength_half = w = h//2 + 1` (rfft width) |
| `z, y, x` | integer **voxel** indices into the volume |
| `kz, ky, kx` | the continuous **rotated sample coordinate** (a frequency) |
| `i_bv, i_bp` | batch indices; `i_h, i_w` projection pixel indices |
| rotations | `(bv_rot, bp, 3, 3)`, **zyx** convention (poses a central *slice*) |
| directions | `(bv_rot, bp, 3)` zyx (or `(…, 2)` yx in 2D) unit vectors — poses a central *line*, which has no in-plane gauge to fix |
| `Float32Ptr` | raw pointer into a contiguous float32 buffer (CPU or GPU); grouped into buffer structs where several travel together |
| `C2` / `C6` / `C8` | complex `(re, im)` SIMD / value + 2 spatial gradients (2D) / value + 3 spatial gradients (3D) |
| `KernelParams` (Python) ↔ `FourierSliceParams` (Mojo) | the scalar parameters, read by **name** across the boundary (no positional index conventions) |
| `ProjectBuffers`, `ScatterBuffers`, `…GradBuffers` | per-kernel bundles naming exactly the buffers that kernel uses |
| `LINEAR` / `CUBIC` | the comptime interpolation kinds |

Complex volume/projection data is read/written through a per-volume 4D
`TileTensor` view `[d, h, w, 2]` over the zero-copy pointer (`_load_c2` /
`_store_c2`); atomic scatter accumulation stays on raw pointers.

---

## 11. Differentiability

Every op is **fully differentiable**: the extraction w.r.t. `reconstruction`,
`rotations` (or `directions`), `shifts`; the insertion w.r.t. `projections`,
`weights`, `rotations` (or `directions`), `shifts`.

The **data** gradients use the adjoint relationship — each extraction/insertion
pair are adjoints, so each one's data-backward is the *other* op's kernel
(`d/d(volume)` of an extraction is the scatter; `d/d(projections)` of an
insertion is the gather, with an exact correction for the Hermitian
double-insert and the skipped `x=0` line). The pose / `shifts` / `weights`
gradients are dedicated backward kernels (`_pose_grad.mojo`, `_line_grad.mojo`,
`_line2d_grad.mojo`): the pose grad chains the **analytical spatial gradient** of
the interpolated field (`_gather_grad.mojo`) through the rotated sample
coordinate; the shift grad differentiates the phase ramp; the weight grad is the
exact adjoint of the weight splat. All are validated by finite differences (CPU
and GPU, both interpolations).

---

## Layout note

This applies to the **Fourier layer** only — the real-space layer takes and
returns real tensors, so there is no layout to match.

The Fourier ops use **rfft layout with DC at the origin** (`[..., 0, 0, 0]`,
unshifted). This is the same operation as
`torch_fourier_slice.extract_central_slices_rfft_3d`, which uses an `fftshift`ed
layout (DC centered). Bridge with a single `ifftshift`/`fftshift` over the
non-redundant dims:

```python
rfft = torch.fft.ifftshift(volume_rfft, dim=(-3, -2))   # teamtomo -> DC-at-origin
slices = extract_central_slices_rfft_3d(rfft.contiguous(), rotations)
slices = torch.fft.fftshift(slices, dim=-2)             # back to teamtomo layout
```

Within the Nyquist band the two are identical (bit-exact for non-interpolating
rotations); they differ only in how out-of-Nyquist corner samples are handled
(the canonical kernel zero-pads, these kernels clamp to the edge voxel).
