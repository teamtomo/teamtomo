"""Non-differentiable kernel calls: prepare buffers, pick CPU/GPU, launch.

Every function here takes contiguous tensors that the kernels read as flat
float32 memory, in the layouts documented in ``_mojo/_common.mojo``::

    image   (c, *spatial) or (*spatial)   float32 or complex64
    coords  (n, ndim)                     float32
    samples n * c elements                image dtype (any shape)
    weights (*spatial,)                   float32

A complex64 tensor is handed over as is: its memory is interleaved (re, im)
float32 pairs, which is exactly the trailing ``inner = 2`` axis the kernels
index -- no ``view_as_real`` views are materialised (each such view is a torch
dispatcher call, and the host path of a small launch is dispatcher-bound).
Scalars cross the boundary once, as a :class:`KernelParams` read by field name
on the Mojo side.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch

from ._gpu import prepare_launch
from ._kernels import cpu_kernels, device_session, gpu_kernels

INTERP_CODES = {
    "nearest": 0,
    "linear": 1,
    "bilinear": 1,
    "trilinear": 1,
    "cubic": 2,
    "bicubic": 2,
}


class KernelParams(NamedTuple):
    """Scalar parameters of one kernel launch (read by NAME on the Mojo side)."""

    ndim: int
    interp: int
    n: int
    c: int
    inner: int
    d0: int
    d1: int
    d2: int
    has_weights: int
    need_grad_image: int
    need_grad_coords: int
    need_grad_values: int
    has_grad_weights: int
    zero_grad_image: int


def _params(
    image: torch.Tensor,
    coords: torch.Tensor,
    ndim: int,
    interp: int,
    *,
    has_weights: bool = False,
    need_grad_image: bool = False,
    need_grad_coords: bool = False,
    need_grad_values: bool = False,
    has_grad_weights: bool = False,
    zero_grad_image: bool = False,
) -> KernelParams:
    """Launch parameters for an ``image`` shaped ``(c, *spatial)`` or ``(*spatial)``."""
    shape = image.shape
    c = shape[0] if len(shape) == ndim + 1 else 1
    d0, d1, d2 = (*shape[len(shape) - ndim :], 1, 1)[:3]
    return KernelParams(
        ndim,
        interp,
        coords.shape[0],
        c,
        2 if image.is_complex() else 1,
        d0,
        d1,
        d2,
        int(has_weights),
        int(need_grad_image),
        int(need_grad_coords),
        int(need_grad_values),
        int(has_grad_weights),
        int(zero_grad_image),
    )


def _launch(name: str, bufs: tuple[torch.Tensor, ...], params: KernelParams) -> None:
    """Run kernel ``name`` on ``bufs``.

    The callers guarantee that every buffer is contiguous, float32 or
    complex64, and on ``bufs[0]``'s device (the kernels only ever see raw
    addresses), so nothing is re-checked here.
    """
    device = bufs[0].device
    if device.type == "cpu":
        getattr(cpu_kernels(), name)(bufs, params)
    else:
        addrs = prepare_launch(device, bufs)
        getattr(gpu_kernels(), name + "_gpu")(device_session(), bufs, params, addrs)


@functools.lru_cache(maxsize=None)
def _dummy(device: torch.device) -> torch.Tensor:
    """Placeholder buffer for outputs a launch does not produce.

    Cached per device: the kernels never touch it (its flag is off), and
    allocating a fresh one would cost a zero-fill launch per backward call.
    """
    return torch.zeros(1, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def sample_forward(
    image: torch.Tensor,
    coords: torch.Tensor,
    ndim: int,
    interp: int,
    out_shape: tuple[int, ...],
) -> torch.Tensor:
    """Image interpolated at coords, as an ``out_shape`` tensor of the image dtype.

    ``out_shape`` must hold ``n * c`` elements; the kernel fills it as a
    contiguous ``(n, c)`` block, so pass the caller's final ``(..., c)`` /
    ``(...)`` shape and no reshape is needed afterwards.
    """
    params = _params(image, coords, ndim, interp)
    out = torch.empty(out_shape, dtype=image.dtype, device=image.device)
    _launch("sample_forward", (image, coords, out), params)
    return out


def sample_backward(
    image: torch.Tensor,
    coords: torch.Tensor,
    grad_samples: torch.Tensor,
    ndim: int,
    interp: int,
    *,
    need_grad_image: bool,
    need_grad_coords: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Adjoint of :func:`sample_forward`; returns ``(grad_image, grad_coords)``."""
    dev = image.device
    # GPU: the kernel zeroes grad_image itself and every thread writes its own
    # grad_coords slot, so both buffers can start uninitialised -- a torch fill
    # would cost a queue round trip before the launch. CPU: the partitioned
    # kernel never visits samples outside the image, so grad_coords must start
    # at zero; a memset is cheap there.
    on_gpu = dev.type != "cpu"
    params = _params(
        image,
        coords,
        ndim,
        interp,
        need_grad_image=need_grad_image,
        need_grad_coords=need_grad_coords,
        zero_grad_image=on_gpu,
    )
    alloc = torch.empty_like if on_gpu else torch.zeros_like
    grad_image = alloc(image) if need_grad_image else _dummy(dev)
    grad_coords = alloc(coords) if need_grad_coords else _dummy(dev)
    _launch(
        "sample_backward",
        (image, coords, grad_samples.contiguous(), grad_image, grad_coords),
        params,
    )
    return (
        grad_image if need_grad_image else None,
        grad_coords if need_grad_coords else None,
    )


# ---------------------------------------------------------------------------
# insertion
# ---------------------------------------------------------------------------


def insert_forward(
    values: torch.Tensor,
    coords: torch.Tensor,
    image: torch.Tensor,
    weights: torch.Tensor | None,
    ndim: int,
    interp: int,
) -> None:
    """Image += splat(values) and, if given, weights += splat(1) -- IN PLACE.

    ``values`` is ``(n, c)`` in the image dtype.
    """
    params = _params(image, coords, ndim, interp, has_weights=weights is not None)
    w = weights if weights is not None else _dummy(image.device)
    _launch("insert_forward", (values, coords, image, w), params)


def insert_backward(
    values: torch.Tensor,
    coords: torch.Tensor,
    grad_image: torch.Tensor,
    grad_weights: torch.Tensor | None,
    ndim: int,
    interp: int,
    *,
    need_grad_values: bool,
    need_grad_coords: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Adjoint of :func:`insert_forward`; returns ``(grad_values, grad_coords)``."""
    params = _params(
        grad_image,
        coords,
        ndim,
        interp,
        need_grad_values=need_grad_values,
        need_grad_coords=need_grad_coords,
        has_grad_weights=grad_weights is not None,
    )
    dev = grad_image.device
    grad_values = torch.empty_like(values) if need_grad_values else _dummy(dev)
    grad_coords = torch.empty_like(coords) if need_grad_coords else _dummy(dev)
    gw = grad_weights.contiguous() if grad_weights is not None else _dummy(dev)
    _launch(
        "insert_backward",
        (values, coords, grad_image.contiguous(), gw, grad_values, grad_coords),
        params,
    )
    return (
        grad_values if need_grad_values else None,
        grad_coords if need_grad_coords else None,
    )


def _no_channels(coords: torch.Tensor, spatial: tuple[int, ...]) -> tuple:
    """Zero-channel image / values placeholders: the kernels then only touch weights."""
    dev = coords.device
    image = torch.empty((0, *spatial), dtype=torch.float32, device=dev)
    values = torch.empty((coords.shape[0], 0), dtype=torch.float32, device=dev)
    return image, values


def insert_weights_forward(
    coords: torch.Tensor, weights: torch.Tensor, ndim: int, interp: int
) -> None:
    """Weights += splat(1) at coords -- IN PLACE (no image update)."""
    image, values = _no_channels(coords, tuple(weights.shape))
    insert_forward(values, coords, image, weights, ndim, interp)


def insert_weights_backward(
    coords: torch.Tensor, grad_weights: torch.Tensor, ndim: int, interp: int
) -> torch.Tensor:
    """d(loss)/d(coords) through the weights splat only."""
    image, values = _no_channels(coords, tuple(grad_weights.shape))
    _, grad_coords = insert_backward(
        values,
        coords,
        image,
        grad_weights,
        ndim,
        interp,
        need_grad_values=False,
        need_grad_coords=True,
    )
    assert grad_coords is not None
    return grad_coords
