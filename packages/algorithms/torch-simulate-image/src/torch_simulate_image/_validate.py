"""Shared input validation for micrograph simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def validate_exit_wave(exit_wave: torch.Tensor) -> tuple[int, int]:
    """Validate a complex exit wave tensor.

    Parameters
    ----------
    exit_wave : torch.Tensor
        Complex tensor with shape ``(..., H, W)``.

    Returns
    -------
    tuple[int, int]
        Image height and width.

    Raises
    ------
    ValueError
        If the tensor is not complex-valued or does not have at least two
        spatial dimensions.
    """
    if exit_wave.ndim < 2:
        msg = f"exit_wave must have shape (..., H, W); got ndim={exit_wave.ndim}"
        raise ValueError(msg)
    if not exit_wave.is_complex():
        msg = f"exit_wave must be complex-valued; got dtype={exit_wave.dtype}"
        raise ValueError(msg)
    height, width = int(exit_wave.shape[-2]), int(exit_wave.shape[-1])
    if height < 1 or width < 1:
        msg = f"exit_wave spatial dimensions must be positive; got {(height, width)}"
        raise ValueError(msg)
    return height, width


def validate_real_image(image: torch.Tensor) -> tuple[int, int]:
    """Validate a real-valued 2D image tensor.

    Parameters
    ----------
    image : torch.Tensor
        Real tensor with shape ``(..., H, W)``.

    Returns
    -------
    tuple[int, int]
        Image height and width.
    """
    if image.ndim < 2:
        msg = f"image must have shape (..., H, W); got ndim={image.ndim}"
        raise ValueError(msg)
    if image.is_complex():
        msg = "image must be real-valued"
        raise ValueError(msg)
    height, width = int(image.shape[-2]), int(image.shape[-1])
    if height < 1 or width < 1:
        msg = f"image spatial dimensions must be positive; got {(height, width)}"
        raise ValueError(msg)
    return height, width


def image_shape_from_tensor(tensor: torch.Tensor) -> tuple[int, int]:
    """Return ``(H, W)`` from a tensor with trailing spatial dimensions."""
    return int(tensor.shape[-2]), int(tensor.shape[-1])
