"""GPU interop helpers: raw device addresses + Metal heap residency.

The Mojo GPU kernels run directly on the memory backing a torch GPU tensor --
no host round-trip. To make that work the Python side has to hand Mojo a raw
device virtual address for each tensor and (on Apple/Metal) keep the tensor's
heap resident and the two command queues synchronised.

Two backends:

* **CUDA** -- ``tensor.data_ptr()`` is already a CUDA device VA, so it is
  passed straight through. Ordering between torch and the Mojo kernel is
  enforced by a full device sync around the launch (correct, if not maximally
  concurrent -- a torch-stream handoff is a later optimisation).
* **Metal (MPS)** -- ``tensor.data_ptr()`` is the ``id<MTLBuffer>`` Obj-C
  object pointer, *not* a GPU VA (verified: ``object_getClassName`` reports
  ``AGXG...Buffer``). We recover the real VA the same way Mojo does
  internally: ``[MTLBuffer gpuAddress] + storage_offset_bytes`` (see
  :func:`gpu_address`). Metal also evicts idle GPU heaps after ~1-1.5s, and
  Mojo does not declare foreign (torch-owned) buffers to its compute encoder,
  so a kernel pointing at an evicted heap silently reads zeros and drops
  writes. :func:`revive_heaps` touches each tensor with a tiny torch op right
  before dispatch to force residency.

Patterned on gabrieldemarmiesse/causal-conv1d-mojo's ``_mps.py``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import time
from functools import lru_cache

import torch

# ---------------------------------------------------------------------------
# Metal gpuAddress extraction (Obj-C)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _objc() -> ctypes.CDLL:
    """Load libobjc and pin the C ABI for the selectors we call.

    ``objc_msgSend.argtypes`` MUST be set on Apple Silicon -- without it the
    default (variadic) ABI is wrong and the call segfaults on entry.
    """
    libobjc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    libobjc.sel_registerName.restype = ctypes.c_void_p
    libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
    libobjc.objc_msgSend.restype = ctypes.c_uint64
    libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return libobjc


@lru_cache(maxsize=1)
def _sel_gpu_address() -> int:
    return _objc().sel_registerName(b"gpuAddress")


def gpu_address(t: torch.Tensor) -> int:
    """Metal GPU virtual address of ``t``'s first element.

    Non-zero storage offsets (sliced views) are handled by adding the byte
    delta between ``tensor.data_ptr()`` and the storage's ``data_ptr()`` to the
    buffer's base ``gpuAddress``. The kernels index ``base + i*stride_i`` with
    element strides, so base + offset is all they need.
    """
    storage = t.untyped_storage()
    buf_obj = storage.data_ptr()
    if buf_obj == 0:
        return 0
    base_gpu = _objc().objc_msgSend(buf_obj, _sel_gpu_address())
    offset_bytes = t.data_ptr() - buf_obj
    return base_gpu + offset_bytes


def device_address(t: torch.Tensor) -> int:
    """Raw device VA of ``t`` for the current GPU backend (0 for empty)."""
    if t.numel() == 0:
        return 0
    if t.device.type == "mps":
        return gpu_address(t)
    return t.data_ptr()


# ---------------------------------------------------------------------------
# Metal heap residency + queue sync
# ---------------------------------------------------------------------------

# macOS evicts idle GPU memory after ~1-1.5s (measured on M-series; matches
# ggml-org/llama.cpp#10119). Touching each heap at least this often keeps it
# comfortably resident while a steady loop pays almost nothing.
_REVIVE_WINDOW_S = 0.35
# Per-storage last-revival stamps, keyed on t.data_ptr() (the fast C accessor;
# untyped_storage() costs ~15us/call, data_ptr ~1us). Keyed per pointer, not a
# single global stamp: a call inside the window may still introduce a tensor
# whose heap has been idle for minutes, which must force a revival.
_revive_stamp: dict[int, float] = {}


def revive_heaps(*tensors: torch.Tensor | None) -> None:
    """Touch each MPS tensor with a tiny GPU op so its ``MTLHeap`` is resident
    when the Mojo kernel dispatches. No-op for empty/None tensors.

    torch MPS tensors are sub-allocations of hazard-tracked ``MTLHeap``s, and
    Mojo's Metal backend only declares resources it allocated itself to its
    encoder (``useResource:`` is skipped for foreign addresses), so a Mojo
    kernel referencing an evicted heap silently reads zeros and drops writes.
    Any submitted torch op referencing the buffer re-maps its heap. Heaps are
    revived individually (torch pools small/large allocations on different
    heaps), and the whole pass is skipped when every argument was revived
    < ``_REVIVE_WINDOW_S`` ago -- far inside the ~1-1.5s eviction horizon.
    """
    live = [t for t in tensors if t is not None and t.numel() > 0]
    if not live:
        return
    now = time.monotonic()
    if all(now - _revive_stamp.get(t.data_ptr(), 0.0) < _REVIVE_WINDOW_S for t in live):
        return
    # Batch one touch per dtype group: a single kernel reading one element of
    # every tensor in the group (layout-agnostic 0-d views, no version bumps).
    groups: dict[torch.dtype, list[torch.Tensor]] = {}
    for t in live:
        groups.setdefault(t.dtype, []).append(t[(0,) * t.dim()])
    for group in groups.values():
        torch.stack(group)
    if len(_revive_stamp) > 65536:
        _revive_stamp.clear()
    for t in live:
        _revive_stamp[t.data_ptr()] = now


def pre_launch_sync(device: torch.device, *tensors: torch.Tensor | None) -> None:
    """Make ``tensors`` safe for a Mojo kernel to read/write, then flush torch.

    On Metal: revive each tensor's heap and flush torch's command queue so any
    pending torch writes land before the Mojo kernel reads them. On CUDA: a
    full device sync serialises torch's stream with the Mojo launch. The Mojo
    entry point syncs its own queue before returning, closing the other side.
    """
    if device.type == "mps":
        revive_heaps(*tensors)
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
