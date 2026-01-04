"""Utility functions for fourier shell correlation computation."""

from typing import Sequence

import einops
import torch


def _prepare_fft_data(
    a_fft: torch.Tensor,
    b_fft: torch.Tensor,
    frequency_grid: torch.Tensor,
    fft_mask: torch.Tensor | None,
    ndim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare FFT data by applying mask and flattening spatial dimensions.

    Args:
        a_fft: FFT of first tensor
        b_fft: FFT of second tensor
        frequency_grid: Frequency grid
        fft_mask: Optional mask for fft indices
        ndim: Number of spatial dimensions

    Returns
    -------
        Tuple of (a_fft_flat, b_fft_flat, frequencies)
    """
    if fft_mask is not None:
        # Apply mask to FFTs and frequency grid
        a_fft_flat = a_fft[..., fft_mask]  # (..., n_masked_freqs)
        b_fft_flat = b_fft[..., fft_mask]
        frequencies = frequency_grid[fft_mask]
    else:
        # Flatten spatial dimensions while preserving batch dimensions using einops
        if ndim == 2:
            # 2D case: (..., h, w) -> (..., h*w)
            a_fft_flat = einops.rearrange(a_fft, "... h w -> ... (h w)")
            b_fft_flat = einops.rearrange(b_fft, "... h w -> ... (h w)")
        elif ndim == 3:
            # 3D case: (..., d, h, w) -> (..., d*h*w)
            a_fft_flat = einops.rearrange(a_fft, "... d h w -> ... (d h w)")
            b_fft_flat = einops.rearrange(b_fft, "... d h w -> ... (d h w)")
        else:
            raise ValueError(f"Unsupported ndim: {ndim}. Only 2D and 3D are supported.")

        # Flatten frequency grid
        frequencies = torch.flatten(frequency_grid)

    return a_fft_flat, b_fft_flat, frequencies


def _compute_frequency_bins_weighted(
    image_shape: Sequence[int], device: torch.device
) -> torch.Tensor:
    """Compute frequency bin centers for weighted shell/ring correlation.

    Args:
        image_shape: Shape of the spatial dimensions
        device: Device to create tensors on

    Returns
    -------
        Frequency bin centers with shape (n_bins,)
    """
    # Use minimum dimension to define number of shells
    # This ensures all frequency components can contribute via interpolation
    min_dim = min(image_shape)
    bin_centers = torch.fft.rfftfreq(min_dim, device=device)

    return bin_centers


def _frequency_to_bin_coordinates(
    frequencies: torch.Tensor, bin_centers: torch.Tensor
) -> torch.Tensor:
    """Convert frequencies to continuous bin coordinates using interpolation.

    Args:
        frequencies: Flattened frequency magnitudes
        bin_centers: Frequency bin centers

    Returns
    -------
        Continuous bin coordinates for each frequency
    """
    # Use searchsorted to find the bin each frequency would fall into
    bin_indices = torch.searchsorted(bin_centers, frequencies, right=True) - 1

    # Handle edge cases
    bin_indices = torch.clamp(bin_indices, 0, len(bin_centers) - 2)

    # Calculate fractional position within the bin
    left_centers = bin_centers[bin_indices]
    right_centers = bin_centers[bin_indices + 1]

    # Avoid division by zero for the last bin
    bin_width = right_centers - left_centers
    bin_width = torch.where(bin_width == 0, torch.ones_like(bin_width), bin_width)

    # Linear interpolation coordinate
    bin_coords = bin_indices.float() + (frequencies - left_centers) / bin_width

    return bin_coords


def _calculate_interpolation_weights(
    bin_coords: torch.Tensor, target_bin: int
) -> torch.Tensor:
    """Calculate linear interpolation weights for a target bin.

    Args:
        bin_coords: Continuous bin coordinates for each frequency
        target_bin: Target bin index

    Returns
    -------
        Interpolation weights for the target bin
    """
    # Distance from each frequency to the target bin
    distances = torch.abs(bin_coords - target_bin)

    # Linear interpolation: weight = max(0, 1 - distance)
    weights = torch.clamp(1.0 - distances, min=0.0)

    return weights


def _compute_shell_correlations_weighted(
    a_fft: torch.Tensor,
    b_fft: torch.Tensor,
    frequencies: torch.Tensor,
    bin_centers: torch.Tensor,
    batch_dims: Sequence[int],
) -> torch.Tensor:
    """Compute weighted normalized cross correlation for each shell using interpolation.

    Args:
        a_fft: Flattened FFT data for first tensor
        b_fft: Flattened FFT data for second tensor
        frequencies: Flattened frequency magnitudes
        bin_centers: Frequency bin centers
        batch_dims: Batch dimensions shape

    Returns
    -------
        Correlation values for each shell with shape (*batch_dims, n_shells)
    """
    n_bins = len(bin_centers)
    correlation_results = []

    # Convert frequencies to continuous bin coordinates
    bin_coords = _frequency_to_bin_coordinates(frequencies, bin_centers)

    # Process each shell using weighted interpolation
    for bin_idx in range(n_bins):
        if bin_idx == 0:
            # DC component always has FSC = 1.0
            correlation_results.append(torch.ones(batch_dims, device=a_fft.device))
        else:
            # Calculate weights for this bin
            weights = _calculate_interpolation_weights(bin_coords, bin_idx)

            # Only include components with non-zero weights
            mask = weights > 0
            if mask.sum() > 0:
                # Get weighted FFT values for all batch items
                a_masked = a_fft[..., mask]  # (..., n_nonzero_weights)
                b_masked = b_fft[..., mask]
                weights_nonzero = weights[mask]

                # Weighted normalized cross correlation
                correlation = _weighted_normalized_cc_complex(
                    a_masked, b_masked, weights_nonzero
                )
                correlation_results.append(correlation)
            else:
                # Empty shell
                correlation_results.append(torch.zeros(batch_dims, device=a_fft.device))

    # Stack results along last dimension
    return torch.stack(correlation_results, dim=-1)  # (..., n_shells)


def _weighted_normalized_cc_complex(
    a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Calculate weighted normalized cross correlation for batched tensors.

    Args:
        a: Complex tensor (..., n_freqs)
        b: Complex tensor (..., n_freqs)
        weights: Weight values (n_freqs,)

    Returns
    -------
        Weighted normalized correlation for each batch item
    """
    # Weighted correlation (sum over frequency dimension)
    correlation = torch.sum(weights * a * torch.conj(b), dim=-1)

    # Weighted norms
    norm_a = torch.sqrt(torch.sum(weights * torch.abs(a) ** 2, dim=-1))
    norm_b = torch.sqrt(torch.sum(weights * torch.abs(b) ** 2, dim=-1))

    # Normalize, handling potential division by zero
    denominator = norm_a * norm_b
    correlation = torch.where(
        denominator > 0, correlation / denominator, torch.zeros_like(correlation)
    )

    return correlation
