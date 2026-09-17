"""Normalization and background-subtraction helpers for tilt-series images."""

import einops
import torch


def normalize_on_central_crop(images: torch.Tensor) -> torch.Tensor:
    """Normalize each image to zero mean / unit variance on its central 25% crop."""
    h, w = images.shape[-2:]
    h_crop = slice(int(0.375 * h), int(0.625 * h))
    w_crop = slice(int(0.375 * w), int(0.625 * w))
    crop = images[..., h_crop, w_crop]
    mean = crop.mean(dim=(-2, -1), keepdim=True)
    std = crop.std(dim=(-2, -1), keepdim=True, correction=0)
    return (images - mean) / std


def subtract_plane(images: torch.Tensor) -> torch.Tensor:
    """Fit and subtract a linear background plane `z = a*y + b*x + c` per image.

    Removes low-order illumination/background gradients by least-squares
    fitting a plane to each `(h, w)` image and subtracting it. Batched over
    any leading dimensions; the plane is fit independently per image.
    """
    h, w = images.shape[-2:]
    device, dtype = images.device, images.dtype

    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones(h * w, device=device, dtype=dtype)
    design = torch.stack([y.reshape(-1), x.reshape(-1), ones], dim=-1)  # (h*w, 3)
    pinv = torch.linalg.pinv(design)  # (3, h*w)

    images_flat, ps = einops.pack([images], "* h w")
    n = images_flat.shape[0]
    targets = images_flat.reshape(n, h * w)
    coeffs = targets @ pinv.T  # (n, 3)
    plane = (coeffs @ design.T).reshape(n, h, w)

    result = images_flat - plane
    [result] = einops.unpack(result, ps, "* h w")
    return result
