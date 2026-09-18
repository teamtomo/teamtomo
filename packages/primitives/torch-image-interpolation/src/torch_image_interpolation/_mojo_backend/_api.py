"""Rank-generic sampling / insertion on top of the Mojo autograd Functions.

The public ``sample_image_{1,2,3}d`` / ``insert_into_image_{1,2,3}d`` validate
their inputs and, when :func:`should_use_mojo` agrees, hand off here. These
wrappers do the shape plumbing shared by all ranks -- flattening the ``(...)``
batch of coordinates, sizing the output -- and call one kernel. Complex tensors
go to the kernels as they are (interleaved float pairs), and the autograd
``Function`` layer is skipped when no input requires a gradient: the host path
of a launch is a string of ~1us dispatcher calls and every one of them shows.
"""

from __future__ import annotations

import torch

from torch_image_interpolation.backend import get_backend

from . import _ops
from ._autograd import (
    InsertFunction,
    InsertImageFunction,
    InsertWeightsFunction,
    SampleFunction,
)
from ._kernels import kernels_available, load_error
from ._ops import INTERP_CODES

_SUPPORTED_DTYPES = (torch.float32, torch.complex64)


def _unsupported_reason(
    op: str,
    image: torch.Tensor,
    coordinates: torch.Tensor,
    interpolation: str,
    values: torch.Tensor | None,
    weights: torch.Tensor | None,
) -> str | None:
    """Why the Mojo kernels cannot serve this call, or None if they can."""
    if interpolation not in INTERP_CODES:
        return f"interpolation {interpolation!r} is not supported"
    if image.dtype not in _SUPPORTED_DTYPES:
        return f"image dtype {image.dtype} is not supported (float32 / complex64 only)"
    if coordinates.device != image.device:
        return "coordinates and image must be on the same device"
    if image.device.type not in ("cpu", "cuda", "mps"):
        return f"device {image.device.type!r} is not supported"
    if op == "insert":
        if not image.is_contiguous():
            return "insertion requires a contiguous image (values are added in place)"
        if values is not None and values.dtype != image.dtype:
            return "values dtype must match the image dtype"
        if weights is not None and (
            weights.dtype != torch.float32 or not weights.is_contiguous()
        ):
            return "weights must be a contiguous float32 tensor"
    if image.device.type == "cpu":
        if not kernels_available("cpu"):
            return f"the Mojo cpu kernels are not available ({load_error('cpu')!r})"
    elif not kernels_available("gpu"):
        return f"the Mojo gpu kernels are not available ({load_error('gpu')!r})"
    return None


def should_use_mojo(
    op: str,
    image: torch.Tensor,
    coordinates: torch.Tensor,
    interpolation: str,
    values: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> bool:
    """Decide whether this call runs on the Mojo kernels.

    Honours :func:`~torch_image_interpolation.backend.get_backend`: ``"torch"``
    never, ``"auto"`` when supported and available, ``"mojo"`` always -- raising
    :class:`RuntimeError` with the reason if that is impossible.
    """
    backend = get_backend()
    if backend == "torch":
        return False
    reason = _unsupported_reason(op, image, coordinates, interpolation, values, weights)
    if reason is None:
        return True
    if backend == "mojo":
        raise RuntimeError(f"backend 'mojo' requested but {reason}")
    return False


def _flatten_coordinates(
    coordinates: torch.Tensor, ndim: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """(..., ndim) [or (...) for 1D] -> contiguous float32 (n, ndim), batch shape.

    Plain ``reshape`` rather than ``einops.pack``: pack concatenates and so
    always copies, which on MPS costs a full command-buffer round trip per call.
    No-op conversions are skipped outright: the host path of a launch is a
    string of ~1us dispatcher calls, and every one of them shows.
    """
    if ndim == 1:
        batch_shape = tuple(coordinates.shape)
        coords = coordinates.reshape(-1, 1)
    else:
        batch_shape = tuple(coordinates.shape[:-1])
        coords = coordinates if coordinates.ndim == 2 else coordinates.reshape(-1, ndim)
    if coords.dtype is not torch.float32:
        coords = coords.to(torch.float32)
    return coords.contiguous(), batch_shape


def sample_image(
    image: torch.Tensor,
    coordinates: torch.Tensor,
    ndim: int,
    interpolation: str,
) -> torch.Tensor:
    """Sample ``image`` (``(*spatial)`` or ``(c, *spatial)``) at ``coordinates``.

    Returns ``(...)`` for a single-channel image, ``(..., c)`` otherwise, in the
    image's dtype -- the same contract as the public ``sample_image_*d``.

    The kernel writes its ``(n, c)`` block straight into a tensor of that final
    shape, and complex images are passed as they are (the kernels read them as
    interleaved float pairs), so no views are created on the way in or out.
    """
    coords, batch_shape = _flatten_coordinates(coordinates, ndim)
    image = image.contiguous()
    out_shape = (
        (*batch_shape, image.shape[0]) if image.ndim == ndim + 1 else batch_shape
    )
    code = INTERP_CODES[interpolation]
    if torch.is_grad_enabled() and (image.requires_grad or coords.requires_grad):
        return SampleFunction.apply(image, coords, ndim, code, out_shape)
    return _ops.sample_forward(image, coords, ndim, code, out_shape)


def insert_into_image(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    image: torch.Tensor,
    weights: torch.Tensor | None,
    ndim: int,
    interpolation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add ``values`` into ``image`` at ``coordinates`` (in place); track ``weights``.

    Same contract as the public ``insert_into_image_*d``: returns the updated
    image (single- or multi-channel, as given) and the float32 weights image.
    """
    is_multichannel = image.ndim == ndim + 1
    if weights is None:
        weights = torch.zeros(
            image.shape[-ndim:], dtype=torch.float32, device=image.device
        )

    c = image.shape[0] if is_multichannel else 1
    values_r = values.reshape(-1, c).contiguous()  # (n, c), image dtype
    coords, _ = _flatten_coordinates(coordinates, ndim)
    code = INTERP_CODES[interpolation]

    tracked = torch.is_grad_enabled() and any(
        t.requires_grad for t in (image, values, coordinates, weights)
    )
    if not tracked:
        _ops.insert_forward(values_r, coords, image, weights, ndim, code)
        return image, weights
    if not (image._is_view() or weights._is_view()):
        image, weights = InsertFunction.apply(
            image, weights, values_r, coords, ndim, code
        )
        return image, weights
    # an in-place Function on a view may return that one tensor only: update the
    # image and the weights with separate Functions (two kernel launches)
    image = InsertImageFunction.apply(image, values_r, coords, ndim, code)
    weights = InsertWeightsFunction.apply(weights, coords, ndim, code)
    return image, weights
