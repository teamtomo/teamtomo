"""Utilities for motion correction."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from torch_fourier_filter.bandpass import bandpass_filter
from torch_fourier_filter.envelopes import b_envelope
from torch_grid_utils import circle

if TYPE_CHECKING:
    from torch_motion_correction.types import FourierFilterConfig


def make_early_stopper(
    patience: int = 5,
    window_size: int = 3,
    tolerance: float = 1e-5,
) -> Callable[[float], bool]:
    """Return a stateful callable that signals when optimization should stop.

    Early stopping is triggered (returns True) when an average of the loss history (of
    'window_size') has a relative change (in absolute terms) less than 'tolerance' for
    'patience' consecutive checks. Otherwise, continue optimization (returns False).
    """
    loss_history: list[float] = []
    wait = 0

    def update(loss: float) -> bool:
        nonlocal loss_history, wait

        loss_history.append(loss)
        if len(loss_history) < window_size:
            return False

        smoothed_this = sum(loss_history[-window_size:]) / window_size
        smoothed_prev = sum(loss_history[-window_size - 1 : -1]) / window_size
        relative_diff = (smoothed_this - smoothed_prev) / (abs(smoothed_prev) + 1e-12)

        if abs(relative_diff) > tolerance:
            wait = 0
        else:
            wait += 1

        return wait >= patience

    return update


def array_to_grid_sample(
    array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate grids for `torch.nn.functional.grid_sample` from array coordinates.

    These coordinates should be used with `align_corners=True` in
    `torch.nn.functional.grid_sample`.


    Parameters
    ----------
    array_coordinates : torch.Tensor
        `(..., d)` array of d-dimensional coordinates.
        Coordinates are in the range `[0, N-1]` for the `N` elements in each dimension.
    array_shape : Sequence[int]
        shape of the array being sampled at `array_coordinates`.
    """
    dtype, device = array_coordinates.dtype, array_coordinates.device
    array_shape_tensor = torch.as_tensor(array_shape, dtype=dtype, device=device)
    grid_sample_coordinates = (array_coordinates / (0.5 * array_shape_tensor - 0.5)) - 1
    grid_sample_coordinates = torch.flip(grid_sample_coordinates, dims=(-1,))
    return grid_sample_coordinates


def fftfreq_to_spatial_frequency(
    frequencies: torch.Tensor, spacing: float
) -> torch.Tensor:
    """Convert frequencies in cycles per pixel to cycles per unit distance."""
    # cycles/px * px/distance = cycles/distance
    return torch.as_tensor(frequencies, dtype=torch.float32) * (1 / spacing)


def spatial_frequency_to_fftfreq(
    frequencies: torch.Tensor, spacing: float
) -> torch.Tensor:
    """Convert frequencies in cycles per unit distance to cycles per pixel."""
    # cycles/distance * distance/px = cycles/px
    return torch.as_tensor(frequencies, dtype=torch.float32) * spacing


def normalize_image(
    image: torch.Tensor, frac_low: float = 0.25, frac_high: float = 0.75
) -> torch.Tensor:
    """Normalizes the image by mean and std of a central box.

    Parameters
    ----------
    image : torch.Tensor
        (t, h, w) image to be normalized where t is the number of frames,
        h is the height, and w is the width.
    frac_low : float
        Fractional lower bound of the central box in both height and width. Default is
        0.25 (central 50%).
    frac_high : float
        Fractional upper bound of the central box in both height and width. Default is
        0.75 (central 50%).

    Returns
    -------
    normalized_image: torch.Tensor
        (t, h, w) normalized image.
    """
    # grab image dimensions
    _t, h, w = image.shape

    # calculate limits of central box
    hl, hu = int(frac_low * h), int(frac_high * h)
    wl, wu = int(frac_low * w), int(frac_high * w)

    # calculate mean and std of central 50%
    center = image[:, hl:hu, wl:wu]
    std, mean = torch.std_mean(center, dim=(-3, -2, -1))

    # normalize and return
    image = (image - mean) / std
    return image


def prepare_patch_filters(
    shape: tuple[int, int],
    pixel_spacing: float,
    fourier_filter: "FourierFilterConfig",
    mask_smoothing_fraction: float = 0.5,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the three reusable image filters (real-space and Fourier) for a patch.

    Parameters
    ----------
    shape : tuple[int, int]
        (h, w) spatial dimensions of the region to filter.
    pixel_spacing : float
        Pixel spacing in Angstroms.
    fourier_filter : FourierFilterConfig
        B-factor and frequency-range parameters.
    mask_smoothing_fraction : float
        Sets the circle-mask smoothing radius as a fraction of the mask radius.
        `smoothing_radius = (min(h, w) / 4) * mask_smoothing_fraction`.
        Use 0.5 for a gentle roll-off (default, matches XC estimator) or 1.0
        for the sharper roll-off used by the optimizer estimator.
    device : torch.device, optional
        Device for the output tensors.

    Returns
    -------
    circle_mask : torch.Tensor
        Real-space circular apodisation mask with shape (h, w).
    b_factor_envelope : torch.Tensor
        rFFT-space B-factor envelope with shape (h, w//2 + 1).
    bandpass : torch.Tensor
        rFFT-space bandpass filter with shape (h, w//2 + 1).
    """
    h, w = shape
    radius = min(h, w) / 4
    smoothing_radius = radius * mask_smoothing_fraction

    circle_mask = circle(
        radius=radius,
        image_shape=shape,
        smoothing_radius=smoothing_radius,
        device=device,
    )
    b_factor_envelope = b_envelope(
        B=fourier_filter.b_factor,
        image_shape=shape,
        pixel_size=pixel_spacing,
        rfft=True,
        fftshift=False,
        device=device,
    )
    bandpass = prepare_bandpass_filter(
        frequency_range=fourier_filter.frequency_range,
        patch_shape=shape,
        pixel_spacing=pixel_spacing,
        device=device,
    )
    return circle_mask, b_factor_envelope, bandpass


def prepare_bandpass_filter(
    frequency_range: tuple[float, float],  # angstroms
    patch_shape: tuple[int, int],
    pixel_spacing: float,  # angstroms
    refinement_fraction: float = 1.0,  # [0, 1]
    device: torch.device = None,
) -> torch.Tensor:
    """Prepare bandpass filter for cross-correlation (fixed, no refinement)."""
    ph, pw = patch_shape

    # Use the higher resolution cutoff (smaller angstrom value)
    cuton, cutoff_max = torch.as_tensor(frequency_range).float()  # angstroms
    cutoff = torch.lerp(cuton, cutoff_max, refinement_fraction)
    low_fftfreq = spatial_frequency_to_fftfreq(1 / cuton, spacing=pixel_spacing)
    high_fftfreq = spatial_frequency_to_fftfreq(1 / cutoff, spacing=pixel_spacing)

    # Prepare bandpass
    bandpass = bandpass_filter(
        low=low_fftfreq,
        high=high_fftfreq,
        falloff=0,
        image_shape=(ph, pw),
        rfft=True,
        fftshift=False,
        device=device,
    )

    return bandpass
