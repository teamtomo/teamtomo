"""Fourier shell correlation between two 2D or 3D images."""

from typing import Sequence

import torch
from torch_grid_utils import fftfreq_grid

from .utils import (
    _compute_frequency_bins_weighted,
    _compute_shell_correlations_weighted,
    _prepare_fft_data,
)


def fourier_ring_correlation(
    a: torch.Tensor, b: torch.Tensor, fft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier ring correlation between two 2D images with batching.

    Supports both square and rectangular images using weighted interpolation.

    Args:
        a: Input tensor of shape (..., h, w)
        b: Input tensor of shape (..., h, w)
        fft_mask: Optional mask for fft, shape should match fft output

    Returns
    -------
        Correlation values of shape (broadcast(...), min(h, w) // 2 + 1)
    """
    # Input validation
    if a.ndim < 2:
        raise ValueError("Input tensors must have at least 2 dimensions.")
    if b.ndim < 2:
        raise ValueError("Input tensors must have at least 2 dimensions.")

    # Enforce that spatial dimensions match
    if a.shape[-2:] != b.shape[-2:]:
        raise ValueError(
            f"Spatial dimensions must match: a.shape[-2:] = {a.shape[-2:]} "
            f"vs b.shape[-2:] = {b.shape[-2:]}"
        )

    # Compute FFT
    spatial_dims_list = [-2, -1]  # Last 2 dimensions
    a_fft = torch.fft.rfftn(a, dim=spatial_dims_list)
    b_fft = torch.fft.rfftn(b, dim=spatial_dims_list)

    # Get image shape (spatial dimensions)
    image_shape = a.shape[-2:]

    return fourier_correlation(a_fft, b_fft, image_shape, fft_mask, rfft=True)


def fourier_shell_correlation(
    a: torch.Tensor, b: torch.Tensor, fft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier shell correlation between two 3D images with batching.

    Supports both cubic and rectangular volumes using weighted interpolation.

    Args:
        a: Input tensor of shape (..., d, h, w)
        b: Input tensor of shape (..., d, h, w)
        fft_mask: Optional mask for fft, shape should match fft output

    Returns
    -------
        Correlation values of shape (broadcast(...), min(d, h, w) // 2 + 1)
    """
    # Input validation
    if a.ndim < 3:
        raise ValueError("Input tensors must have at least 3 dimensions.")
    if b.ndim < 3:
        raise ValueError("Input tensors must have at least 3 dimensions.")

    # Enforce that spatial dimensions match
    if a.shape[-3:] != b.shape[-3:]:
        raise ValueError(
            f"Spatial dimensions must match: a.shape[-3:] = {a.shape[-3:]} "
            f"vs b.shape[-3:] = {b.shape[-3:]}"
        )

    # Compute FFT
    spatial_dims_list = [-3, -2, -1]  # Last 3 dimensions
    a_fft = torch.fft.rfftn(a, dim=spatial_dims_list)
    b_fft = torch.fft.rfftn(b, dim=spatial_dims_list)

    # Get image shape (spatial dimensions)
    image_shape = a.shape[-3:]

    return fourier_correlation(a_fft, b_fft, image_shape, fft_mask, rfft=True)


def fourier_correlation(
    a_fft: torch.Tensor,
    b_fft: torch.Tensor,
    image_shape: Sequence[int],
    fft_mask: torch.Tensor | None = None,
    rfft: bool = True,
) -> torch.Tensor:
    """Compute fourier correlation from FFT data supporting rectangular shapes.

    Args:
        a_fft: (..., *fft_shape) tensor containing FFT of a (..., *image_shape) tensor
        b_fft: (..., *fft_shape) tensor containing FFT of a (..., *image_shape) tensor
        image_shape: Size of spatial dimensions before FFT (e.g. (h, w) or (d, h, w))
            Note: For real FFTs, fft_shape = (*image_shape[:-1], image_shape[-1]//2 + 1)
        fft_mask: Optional mask for fft indices
        rfft: Whether the FFT data is from rfft (True) or fft (False)

    Returns
    -------
        Fourier correlation values with shape (..., n_shells)
    """
    # Input validation - check that FFT tensors can be broadcast together
    try:
        # This will raise an error if tensors can't be broadcast
        torch.broadcast_shapes(a_fft.shape, b_fft.shape)
    except RuntimeError as e:
        raise ValueError(f"FFT tensors must be broadcastable: {e}") from e

    # Validate fft_mask
    fft_shape = a_fft.shape[-len(image_shape) :]
    if fft_mask is not None and fft_mask.shape != fft_shape:
        raise ValueError("fft_mask must have same shape as fft output.")

    # Compute frequency grid and prepare FFT data
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=False,
        norm=True,
        device=a_fft.device,
    )

    # Apply mask and flatten spatial dimensions
    a_fft_flat, b_fft_flat, frequencies = _prepare_fft_data(
        a_fft, b_fft, frequency_grid, fft_mask, len(image_shape)
    )

    # Compute frequency bins using weighted approach
    bin_centers = _compute_frequency_bins_weighted(image_shape, a_fft.device)

    # Compute broadcast batch dimensions for correlation computation
    broadcast_shape = torch.broadcast_shapes(a_fft.shape, b_fft.shape)
    batch_dims = broadcast_shape[: -len(image_shape)]

    # Compute correlations using weighted interpolation
    correlations = _compute_shell_correlations_weighted(
        a_fft_flat,
        b_fft_flat,
        frequencies,
        bin_centers,
        batch_dims,
    )

    return torch.real(correlations)


def fsc(
    a: torch.Tensor, b: torch.Tensor, fft_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Fourier ring/shell correlation between two images.

    .. deprecated::
        Use `fourier_ring_correlation` for 2D or `fourier_shell_correlation` for 3D.

    Args:
        a: Input tensor (2D or 3D)
        b: Input tensor (2D or 3D), same shape as a
        fft_mask: Optional mask for fft indices

    Returns
    -------
        Correlation values of shape (n_shells,)
    """
    if a.ndim == 2:
        return fourier_ring_correlation(a, b, fft_mask)
    elif a.ndim == 3:
        return fourier_shell_correlation(a, b, fft_mask)
    else:
        raise ValueError("images must be 2D or 3D.")
