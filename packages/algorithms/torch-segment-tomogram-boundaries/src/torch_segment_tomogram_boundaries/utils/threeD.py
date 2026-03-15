"""3D volume processing utilities for tomographic data.

This module provides utilities for 3D volume manipulation including
resizing, padding, and interpolation operations used in the tomographic
processing pipeline.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_and_pad_3d(
    tensor: torch.Tensor, target_shape: tuple, mode: str
) -> torch.Tensor:
    """
    Resize and pad 3D tensor to target shape with appropriate interpolation.

    Parameters
    ----------
    tensor : torch.Tensor
        Input 3D tensor of shape (D, H, W).
    target_shape : tuple
        Target shape as (depth, height, width).
    mode : str
        Interpolation mode - 'image' for trilinear (continuous data) or
        'label' for nearest (discrete/label data).

    Returns
    -------
    torch.Tensor
        Resized and padded tensor with exact target_shape.
        Padding uses median value for 'image' mode, 0 for 'label' mode.
    """
    is_label = mode == "label"
    input_dtype = tensor.dtype
    tensor = tensor.float()

    tensor_5d = tensor.unsqueeze(0).unsqueeze(0)
    resized_5d = F.interpolate(
        tensor_5d,
        size=target_shape,
        mode="trilinear" if not is_label else "nearest",
        align_corners=False if not is_label else None,
    )
    resized_tensor = resized_5d.squeeze(0).squeeze(0)

    if is_label:
        resized_tensor = resized_tensor.to(input_dtype)

    shape = resized_tensor.shape
    discrepancy = [max(0, ts - s) for ts, s in zip(target_shape, shape)]
    if not any(d > 0 for d in discrepancy):
        return resized_tensor

    padding = []
    for d in reversed(discrepancy):
        pad_1, pad_2 = d // 2, d - (d // 2)
        padding.extend([pad_1, pad_2])

    padding_value = torch.median(tensor).item() if not is_label else 0
    return F.pad(resized_tensor, tuple(padding), mode="constant", value=padding_value)


def gpu_gaussian_blur_3d(
    tensor: torch.Tensor, sigma: float, device: torch.device
) -> torch.Tensor:
    """
    Apply 3D Gaussian blur using GPU-accelerated convolution.

    Parameters
    ----------
    tensor : torch.Tensor
        Input 3D tensor to blur.
    sigma : float
        Standard deviation for Gaussian kernel.
    device : torch.device
        PyTorch device for computation.

    Returns
    -------
    torch.Tensor
        Blurred tensor of same shape as input.
    """
    kernel_size = 2 * math.ceil(3.5 * sigma) + 1
    coords = (
        torch.arange(kernel_size, device=tensor.device, dtype=tensor.dtype)
        - kernel_size // 2
    )
    g = coords.pow(2)
    kernel_1d = torch.exp(-g / (2 * sigma**2))
    kernel_3d = (
        kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    )
    kernel_3d /= kernel_3d.sum()
    kernel_5d = kernel_3d.unsqueeze(0).unsqueeze(0)
    conv = nn.Conv3d(1, 1, kernel_size, padding="same", bias=False).to(device)
    conv.weight.data = kernel_5d
    return conv(tensor.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
