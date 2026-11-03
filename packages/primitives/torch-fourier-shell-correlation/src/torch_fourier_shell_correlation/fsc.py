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

    Parameters
    ----------
    a : torch.Tensor
        `(..., h, w)` real-space image(s).
    b : torch.Tensor
        `(..., h, w)` real-space image(s). Batch dimensions must broadcast with
        those of `a`.
    fft_mask : torch.Tensor | None, optional
        Boolean mask selecting which Fourier components contribute, with the same
        shape as the (non-fftshifted) rfft of the spatial dimensions,
        `(h, w // 2 + 1)`. If `None` (default), all components are used.

    Returns
    -------
    torch.Tensor
        `(..., min(h, w) // 2 + 1)` array of correlation values for rings at
        frequencies `torch.fft.rfftfreq(min(h, w))` (cycles per pixel). The
        first (DC) value is always 1.

    Raises
    ------
    ValueError
        If the inputs have fewer than 2 dimensions, their last 2 dimensions
        differ, or `fft_mask` has the wrong shape.
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

    Parameters
    ----------
    a : torch.Tensor
        `(..., d, h, w)` real-space volume(s).
    b : torch.Tensor
        `(..., d, h, w)` real-space volume(s). Batch dimensions must broadcast
        with those of `a`.
    fft_mask : torch.Tensor | None, optional
        Boolean mask selecting which Fourier components contribute, with the same
        shape as the (non-fftshifted) rfft of the spatial dimensions,
        `(d, h, w // 2 + 1)`. If `None` (default), all components are used.

    Returns
    -------
    torch.Tensor
        `(..., min(d, h, w) // 2 + 1)` array of correlation values for shells
        at frequencies `torch.fft.rfftfreq(min(d, h, w))` (cycles per pixel).
        The first (DC) value is always 1.

    Raises
    ------
    ValueError
        If the inputs have fewer than 3 dimensions, their last 3 dimensions
        differ, or `fft_mask` has the wrong shape.
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

    Components contribute to the two nearest shells with linear interpolation
    weights based on their frequency magnitude.

    Parameters
    ----------
    a_fft : torch.Tensor
        `(..., *fft_shape)` (non-fftshifted) DFT of a `(..., *image_shape)`
        tensor. For an rfft, `fft_shape` is
        `(*image_shape[:-1], image_shape[-1] // 2 + 1)`.
    b_fft : torch.Tensor
        `(..., *fft_shape)` (non-fftshifted) DFT of a `(..., *image_shape)`
        tensor. Must be broadcastable with `a_fft`.
    image_shape : Sequence[int]
        Shape of the spatial dimensions before the FFT, `(h, w)` or `(d, h, w)`.
    fft_mask : torch.Tensor | None, optional
        Boolean mask with shape `fft_shape` selecting which Fourier components
        contribute. If `None` (default), all components are used.
    rfft : bool, optional
        Whether the DFTs are the output of an `rfft` (`True`, default) or a full
        `fft` (`False`).

    Returns
    -------
    torch.Tensor
        `(..., min(image_shape) // 2 + 1)` array of real-valued correlation
        values for shells at frequencies `torch.fft.rfftfreq(min(image_shape))`
        (cycles per pixel). The first (DC) value is always 1.

    Raises
    ------
    ValueError
        If `a_fft` and `b_fft` do not broadcast, `fft_mask` has the wrong shape,
        or `image_shape` is not 2D or 3D.
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
    if len(image_shape) == 2:
        grid_shape: tuple[int, int] | tuple[int, int, int] = (
            image_shape[0],
            image_shape[1],
        )
    elif len(image_shape) == 3:
        grid_shape = (image_shape[0], image_shape[1], image_shape[2])
    else:
        raise ValueError(
            f"fourier_correlation only supports 2D or 3D image_shape, got {image_shape}"
        )
    frequency_grid = fftfreq_grid(
        image_shape=grid_shape,
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

    Parameters
    ----------
    a : torch.Tensor
        `(h, w)` image or `(d, h, w)` volume.
    b : torch.Tensor
        Image or volume with the same shape as `a`.
    fft_mask : torch.Tensor | None, optional
        Boolean mask with the shape of the (non-fftshifted) rfft of `a`
        selecting which Fourier components contribute. If `None` (default), all
        components are used.

    Returns
    -------
    torch.Tensor
        `(min(a.shape) // 2 + 1, )` array of correlation values, one per
        ring/shell.

    Raises
    ------
    ValueError
        If `a` is not 2D or 3D.
    """
    if a.ndim == 2:
        return fourier_ring_correlation(a, b, fft_mask)
    elif a.ndim == 3:
        return fourier_shell_correlation(a, b, fft_mask)
    else:
        raise ValueError("images must be 2D or 3D.")
