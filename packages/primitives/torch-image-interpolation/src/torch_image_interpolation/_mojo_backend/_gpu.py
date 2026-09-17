"""GPU interop helpers: raw device addresses + Metal heap residency.

Shared verbatim with ``torch_fourier_slice.experimental._gpu`` (that package
depends on this one, so the helpers live here rather than being imported).

The Mojo GPU kernels run directly on the memory backing a torch GPU tensor --
no host round-trip. To make that work the Python side has to hand Mojo a raw
device virtual address for each tensor and (on Apple/Metal) keep the tensor's
heap resident and the two command queues synchronised.

Two backends:

* **CUDA** -- ``tensor.data_ptr()`` is already a CUDA device VA, so it is
  passed straight through. Ordering between torch and the Mojo kernel is the
  stream's job: the launch is enqueued on torch's own current stream (see
  :func:`stream_address`), so no device-wide synchronisation is needed.
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
    lib_path = ctypes.util.find_library("objc")
    if lib_path is None:
        raise OSError("libobjc not found -- Metal GPU interop requires macOS")
    libobjc = ctypes.cdll.LoadLibrary(lib_path)
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


# The CUDA driver's explicit handle for the legacy default stream
# (``CU_STREAM_LEGACY``). torch reports the default stream's ``CUstream`` as 0
# (the NULL stream), which the Mojo entry points reserve for "no external
# stream" (Metal), so the default stream is passed under this alias instead.
_CU_STREAM_LEGACY = 0x1


def stream_address(device: torch.device) -> int:
    """Address of torch's active GPU stream for the launch to enqueue on.

    CUDA: the current stream's ``CUstream`` -- the Mojo kernel wraps it and
    enqueues on it, so ordering with the surrounding torch ops needs no device
    sync. torch's default stream is the NULL stream and is passed as the
    driver's ``CU_STREAM_LEGACY`` alias (otherwise the entry point would take
    the Metal path below and synchronise the whole context on every launch --
    that cost a ``cuStreamSynchronize`` per call, serialising host and GPU).
    Metal/MPS: 0 (no external-stream handoff; the kernel runs on the
    DeviceContext's own stream and the entry point syncs it).
    """
    if device.type == "cuda":
        return _cuda_raw_stream(device) or _CU_STREAM_LEGACY
    return 0


if hasattr(torch._C, "_cuda_getCurrentRawStream"):

    def _cuda_raw_stream(device: torch.device) -> int:
        """Current ``CUstream`` of ``device`` via torch's C accessor (~0.1us).

        ``torch.cuda.current_stream(device).cuda_stream`` builds a Python
        ``Stream`` object on every call (~4us -- a tenth of a small launch).
        """
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return torch._C._cuda_getCurrentRawStream(index)

else:  # pragma: no cover  (older torch)

    def _cuda_raw_stream(device: torch.device) -> int:
        return torch.cuda.current_stream(device).cuda_stream


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
    """Keep each MPS tensor's ``MTLHeap`` resident for the coming dispatch.

    Touches each tensor with a tiny GPU op; a no-op for empty/None tensors.

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
    """Make ``tensors`` safe for a Mojo kernel to read/write before the launch.

    Metal: revive each tensor's heap (else Mojo may read an evicted heap as
    zeros) and flush torch's command queue so pending torch writes land first.
    CUDA: nothing to do -- the kernel is enqueued on torch's own stream (see
    ``stream_address``), so it is already ordered after prior torch ops and
    before later ones; no full device sync is needed.
    """
    if device.type == "mps":
        revive_heaps(*tensors)
        torch.mps.synchronize()


def prepare_launch(device: torch.device, bufs: tuple[torch.Tensor, ...]) -> tuple:
    """Make ``bufs`` launch-safe and return the address tuple Mojo expects.

    Every GPU entry point takes one ``addrs`` tuple: the raw device address of
    each buffer in order, then torch's stream address as the trailing element.
    (The stream is folded in here rather than passed as its own argument to stay
    under Mojo's ``def_function`` arity cap.)
    """
    if device.type == "cuda":
        # data_ptr() is the device VA; ordering is the stream's job (no sync)
        return (
            *[t.data_ptr() if t.numel() else 0 for t in bufs],
            stream_address(device),
        )
    addrs = (*(device_address(t) for t in bufs), stream_address(device))
    pre_launch_sync(device, *bufs)
    return addrs
