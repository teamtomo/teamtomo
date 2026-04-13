"""Bandpass filters using Fourier transform."""

from __future__ import annotations

import einops
import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid


def bandpass_filter(
    low: float,
    high: float,
    falloff: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a bandpass filter for the Fourier transform.

    Note that low and high are in frequency range 0-0.5 at Nyquist.

    Flat response at 1.0 on ``(low, high]``; quarter-cosine ramps in
    ``(low - falloff, low]`` and ``(high, high + falloff]``. For a smooth
    analytic mask over the whole grid, see :func:`bandpass_filter_hyptan`.

    Parameters
    ----------
    low: float
        Lower cutoff frequency.
    high: float
        Higher cutoff frequency.
    falloff: float
        Falloff of the filter.
    image_shape: tuple
        Shape of the image.
    rfft: bool
        If the FFT is a real FFT.
    fftshift: bool
        If the FFT is shifted.
    device: torch.device
        Device to use.

    Returns
    -------
    band_float: torch.Tensor
        Bandpass filter.
    """
    # get the frequency grid
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=device,
    )
    # get the frequency band
    band = torch.logical_and(frequency_grid > low, frequency_grid <= high)
    band_float = band.float()

    # add cosine falloff
    band_with_falloff = torch.logical_and(
        frequency_grid > low - falloff, frequency_grid <= high + falloff
    )
    falloff_mask = torch.logical_and(band_with_falloff, ~band)
    cutoffs = torch.tensor([low, high], dtype=torch.float32, device=device)
    cutoffs = einops.rearrange(cutoffs, pattern="cutoffs -> cutoffs 1")
    distance = torch.abs(frequency_grid[falloff_mask] - cutoffs)
    distance = einops.reduce(distance, "cutoffs b -> b", reduction="min")
    softened_values = torch.cos((distance / falloff) * (torch.pi / 2))
    band_float[falloff_mask] = softened_values
    return band_float


def low_pass_filter(
    cutoff: float,
    falloff: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a low pass filter for the Fourier transform.

    Parameters
    ----------
    cutoff: float
        Cutoff frequency.
    falloff: float
        Falloff of the filter.
    image_shape: tuple
        Shape of the image.
    rfft: bool
        If the FFT is a real FFT.
    fftshift: bool
        If the FFT is shifted.
    device: torch.device
        Device to use.

    Returns
    -------
    filter: torch.Tensor
        Low pass filter.
    """
    filter_lp = bandpass_filter(
        low=0,
        high=cutoff,
        falloff=falloff,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        device=device,
    )
    return filter_lp


def high_pass_filter(
    cutoff: float,
    falloff: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a high pass filter for the Fourier transform.

    Parameters
    ----------
    cutoff: float
        Cutoff frequency.
    falloff: float
        Falloff of the filter.
    image_shape: tuple
        Shape of the image.
    rfft: bool
        If the FFT is a real FFT.
    fftshift: bool
        If the FFT is shifted.
    device: torch.device
        Device to use.

    Returns
    -------
    filter: torch.Tensor
        High pass filter.
    """
    filter_hp = bandpass_filter(
        low=cutoff,
        high=1,
        falloff=falloff,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        device=device,
    )
    return filter_hp


def bandpass_filter_hyptan(
    low: float,
    high: float,
    falloff: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool,
    device: torch.device = None,
) -> torch.Tensor:
    """Hyperbolic-tangent bandpass on the frequency grid.

    Builds the mask from four shifted ``tanh`` terms, then divides by the
    global maximum so the peak is 1. The response is smooth everywhere (not
    flat in the passband like :func:`bandpass_filter`); edge steepness scales
    with ``falloff * (high - low)``.

    Parameters
    ----------
    low: float
        Lower cutoff frequency (same normalisation as :func:`bandpass_filter`).
    high: float
        Higher cutoff frequency.
    falloff: float
        Controls edge steepness (larger gives softer edges).
    image_shape: tuple
        Shape of the image.
    rfft: bool
        If the FFT is a real FFT.
    fftshift: bool
        If the FFT is shifted.
    device: torch.device
        Device to use.

    Returns
    -------
    bandpass: torch.Tensor
        Values in ``[0, 1]``, same shape as the frequency grid.
    """
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=device,
    )
    span = high - low
    scale = falloff * span
    d = frequency_grid
    bandpass = 0.5 * (
        torch.tanh(torch.pi * (d + high) / scale)
        - torch.tanh(torch.pi * (d - high) / scale)
        - torch.tanh(torch.pi * (d + low) / scale)
        + torch.tanh(torch.pi * (d - low) / scale)
    )
    bandpass = bandpass / bandpass.max()
    return bandpass
