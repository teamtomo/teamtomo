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


def symmetrise_kx0_plane(grad_volume: torch.Tensor) -> torch.Tensor:
    """Adjoint of the Hermitian double-insert on a volume's kx=0 plane.

    The insertion writes each kx=0 sample and its (-z, -y) conjugate mirror, so
    the transpose adds the mirror's cotangent back onto each sample. Self-mirror
    points (z, y each 0 or N/2) map to themselves and were inserted once, not
    doubled, so they are excluded.
    """
    out = grad_volume.contiguous().clone()
    plane = grad_volume[..., 0]
    mirror = torch.conj(plane.flip(dims=(-2, -1)).roll(shifts=(1, 1), dims=(-2, -1)))
    self_mask = torch.zeros_like(plane, dtype=torch.bool)
    d, h = plane.shape[-2], plane.shape[-1]
    for zi in (0, d // 2):
        for yi in (0, h // 2):
            self_mask[..., zi, yi] = True
    out[..., 0] = plane + torch.where(self_mask, plane.new_zeros(()), mirror)
    return out
