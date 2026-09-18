"""``torch.autograd.Function`` wrappers around the Mojo kernels.

Sampling and insertion are adjoints of one another, so each one's data
gradient is the other's kernel: d/d(image) of a sample is a scatter of the
output gradient, d/d(values) of an insertion is a gather of the image gradient.
Both also differentiate w.r.t. the coordinates via the analytical spatial
derivative of the interpolant (zero for ``nearest``).

Only the coordinates (and, where the coordinate gradient needs it, the image /
values) are saved for backward -- no per-tap intermediates -- so peak memory is
independent of the interpolation stencil size.

Insertion accumulates into its image and weights IN PLACE, like the pure-torch
``index_put_`` implementation. :class:`InsertFunction` updates both in one fused
launch and returns both; autograd only permits that when neither input is a
view of another tensor, so for views :mod:`._api` falls back to
:class:`InsertImageFunction` + :class:`InsertWeightsFunction`, each returning the
single (view) tensor it modified.
"""

from __future__ import annotations

from typing import Any

import torch

from . import _ops


class SampleFunction(torch.autograd.Function):
    """image (``(c, *spatial)`` / ``(*spatial)``) interpolated at coords (n, ndim).

    Returns an ``out_shape`` tensor in the image dtype (``n * c`` elements).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        image: torch.Tensor,
        coords: torch.Tensor,
        ndim: int,
        interp: int,
        out_shape: tuple[int, ...],
    ) -> torch.Tensor:
        ctx.save_for_backward(image, coords)
        ctx.ndim = ndim
        ctx.interp = interp
        return _ops.sample_forward(image, coords, ndim, interp, out_shape)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_samples: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        image, coords = ctx.saved_tensors
        need_image, need_coords = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (need_image or need_coords):
            return None, None, None, None, None
        grad_image, grad_coords = _ops.sample_backward(
            image,
            coords,
            grad_samples,
            ctx.ndim,
            ctx.interp,
            need_grad_image=need_image,
            need_grad_coords=need_coords,
        )
        return grad_image, grad_coords, None, None, None


class InsertFunction(torch.autograd.Function):
    """image += splat(values), weights += splat(1) at coords -- IN PLACE, one launch.

    ``image`` is the caller's ``(*spatial)`` / ``(c, *spatial)`` tensor (real or
    complex) and must not be a view; ``values`` is ``(n, c)`` in the image dtype.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        image: torch.Tensor,
        weights: torch.Tensor,
        values: torch.Tensor,
        coords: torch.Tensor,
        ndim: int,
        interp: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ops.insert_forward(values, coords, image, weights, ndim, interp)
        ctx.mark_dirty(image, weights)
        ctx.save_for_backward(values, coords)
        ctx.ndim = ndim
        ctx.interp = interp
        return image, weights

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_image: torch.Tensor, grad_weights: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
    ]:
        values, coords = ctx.saved_tensors
        need_values, need_coords = ctx.needs_input_grad[2], ctx.needs_input_grad[3]
        grad_values = grad_coords = None
        if need_values or need_coords:
            grad_values, grad_coords = _ops.insert_backward(
                values,
                coords,
                grad_image,
                grad_weights if need_coords else None,
                ctx.ndim,
                ctx.interp,
                need_grad_values=need_values,
                need_grad_coords=need_coords,
            )
        # outputs = inputs + splat(...): identity gradient to image and weights
        return grad_image, grad_weights, grad_values, grad_coords, None, None


class InsertImageFunction(torch.autograd.Function):
    """image (``(c, *spatial)`` / ``(*spatial)``) += splat(values) -- IN PLACE.

    View-safe: the single modified tensor is the only output.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        image: torch.Tensor,
        values: torch.Tensor,
        coords: torch.Tensor,
        ndim: int,
        interp: int,
    ) -> torch.Tensor:
        _ops.insert_forward(values, coords, image, None, ndim, interp)
        ctx.mark_dirty(image)
        ctx.save_for_backward(values, coords)
        ctx.ndim = ndim
        ctx.interp = interp
        return image

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, None, None]:
        values, coords = ctx.saved_tensors
        need_values, need_coords = ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        grad_values = grad_coords = None
        if need_values or need_coords:
            grad_values, grad_coords = _ops.insert_backward(
                values,
                coords,
                grad_image,
                None,
                ctx.ndim,
                ctx.interp,
                need_grad_values=need_values,
                need_grad_coords=need_coords,
            )
        # image_out = image_in + splat(...): identity gradient to the input image
        return grad_image, grad_values, grad_coords, None, None


class InsertWeightsFunction(torch.autograd.Function):
    """weights (*spatial,) += splat(1) at coords -- IN PLACE (view-safe)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, weights: torch.Tensor, coords: torch.Tensor, ndim: int, interp: int
    ) -> torch.Tensor:
        _ops.insert_weights_forward(coords, weights, ndim, interp)
        ctx.mark_dirty(weights)
        ctx.save_for_backward(coords)
        ctx.ndim = ndim
        ctx.interp = interp
        return weights

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, None, None]:
        (coords,) = ctx.saved_tensors
        grad_coords = None
        if ctx.needs_input_grad[1]:
            grad_coords = _ops.insert_weights_backward(
                coords, grad_weights, ctx.ndim, ctx.interp
            )
        return grad_weights, grad_coords, None, None
