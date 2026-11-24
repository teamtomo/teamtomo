"""Helpers shared by the per-family backend modules."""

from __future__ import annotations

import math

import torch


def reconstruction_volume_shape(
    proj_sidelength: int, oversampling: float
) -> tuple[int, int, int]:
    """Cubic rfft volume (d, h, w) for inserting ``proj_sidelength`` slices."""
    raw = math.ceil(proj_sidelength * oversampling)
    side = raw + (raw % 2)  # ensure even
    return (side, side, side // 2 + 1)


def reduce_to(grad: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum a per-(bv_rot/bv_shift_2d, bp, ...) grad down to ``target``'s shape.

    The pose tensors broadcast over the volume batch (leading size-1 dims and/or
    ``bv_rot/bv_shift_2d == 1``); the corresponding gradient sums those axes.
    """
    while grad.dim() > target.dim():
        grad = grad.sum(0)
    for dim in range(grad.dim()):
        if target.shape[dim] == 1 and grad.shape[dim] != 1:
            grad = grad.sum(dim, keepdim=True)
    return grad.reshape(target.shape).to(device=target.device, dtype=target.dtype)


def hermitian_planes(n: int) -> tuple[int, ...]:
    """Rfft x indices whose plane stores its own Hermitian mirror, for edge ``n``.

    See "Hermitian bookkeeping" in ``experimental/README.md``.
    """
    return (0, n // 2) if n % 2 == 0 else (0,)


def symmetrise_hermitian_planes(grad_volume: torch.Tensor) -> torch.Tensor:
    """Adjoint of the Hermitian double-insert on a volume's kx = 0 and n/2 planes."""
    out = grad_volume.contiguous().clone()
    for x in hermitian_planes(grad_volume.shape[-3]):
        plane = grad_volume[..., x]
        mirror = plane.flip(dims=(-2, -1)).roll(shifts=(1, 1), dims=(-2, -1))
        out[..., x] = plane + torch.conj(mirror)
    return out


def symmetrise_hermitian_columns(grad_image: torch.Tensor) -> torch.Tensor:
    """Adjoint of the Hermitian double-insert on an image's kx = 0 and n/2 columns."""
    out = grad_image.contiguous().clone()
    for x in hermitian_planes(grad_image.shape[-2]):
        column = grad_image[..., x]
        out[..., x] = column + torch.conj(column.flip(-1).roll(1, -1))
    return out


def input_pixel_scale(n: int, device: torch.device) -> torch.Tensor:
    """``(n, n // 2 + 1)`` weight of each input rfft pixel in an insertion.

    Mirrors ``_input_pixel_scale`` in ``_mojo/_common.mojo``.
    """
    scale = torch.ones((n, n // 2 + 1), device=device)
    scale[n // 2 + 1 :, 0] = 0
    scale[0, 0] = 0.5
    if n % 2 == 0:
        scale[n // 2, 0] = 0.5
        scale[:, n // 2] = 0.5
    return scale


def input_line_scale(n: int, device: torch.device) -> torch.Tensor:
    """``(n // 2 + 1,)`` weight of each input line sample in an insertion.

    Mirrors ``_input_line_scale`` in ``_mojo/_common.mojo``.
    """
    scale = torch.ones(n // 2 + 1, device=device)
    for x in hermitian_planes(n):
        scale[x] = 0.5
    return scale
