import math

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize_scalar
from torch_fourier_filter.bandpass import bandpass_filter

from torch_tiltxcorr.utils import (
    calculate_cross_correlation,
    get_shift_from_correlation_image,
    transform_shifts_from_stretched_images,
    taper_image_edges,
    apply_stretch_perpendicular_to_tilt_axis,
)


def tiltxcorr_with_sample_tilt_estimation(
    tilt_series: torch.Tensor,  # (b, h, w)
    tilt_angles: torch.Tensor,  # (b, )
    tilt_axis_angle: float,
    pixel_spacing_angstroms: float | None = None,
    lowpass_angstroms: float | None = None,
    sample_tilt_range: tuple[float, float] = (-30.0, 30.0),  # search range in degrees
    max_iter: int = 10,  # max iterations for Brent's method
) -> tuple[torch.Tensor, float]:  # (b, 2) yx shifts and optimal sample tilt
    """
    Estimate stage shifts and sample tilt by maximizing sum of inter-image tilted cross correlations.

    The sample tilt estimate represents sample tilt about the stage in the microscope.
    E.g. if the sample is physically tilted +5°, then at nominal 0° the beam
    sees the sample at +5°, and estimate_sample_tilt = +5°.

    Uses scipy's Brent's method (bounded) for optimization.

    Args:
        tilt_series: Stack of tilt images (b, h, w)
        tilt_angles: Nominal stage tilt angles for each image (b,)
        tilt_axis_angle: Angle of tilt axis in degrees
        pixel_spacing_angstroms: Pixel spacing in angstroms
        lowpass_angstroms: Low-pass filter cutoff in angstroms
        sample_tilt_range: (min, max) range of sample tilt angles to search in degrees
        max_iter: Maximum iterations for Brent's method optimizer

    Returns:
        shifts: Optimal shifts for each image (b, 2) yx coords
        sample_tilt: Optimal sample tilt angle in degrees
    """
    sample_tilt_min, sample_tilt_max = sample_tilt_range

    # Track history for correlation curve
    sample_tilt_history = []
    correlation_history = []

    def objective(sample_tilt: float) -> float:
        """Objective function: negative correlation (to minimize)."""
        _, total_correlation = _compute_shifts_with_sample_tilt(
            tilt_series=tilt_series,
            tilt_angles=tilt_angles,
            tilt_axis_angle=tilt_axis_angle,
            pixel_spacing_angstroms=pixel_spacing_angstroms,
            lowpass_angstroms=lowpass_angstroms,
            sample_tilt=sample_tilt,
        )

        # Convert to float and store history
        correlation_val = float(total_correlation.item())
        sample_tilt_history.append(sample_tilt)
        correlation_history.append(correlation_val)

        # Return negative for minimization
        return -correlation_val

    # Run Brent's method optimization
    result = minimize_scalar(
        objective,
        bounds=(sample_tilt_min, sample_tilt_max),
        method="bounded",
        options={"maxiter": max_iter},
    )

    estimated_sample_tilt = float(result.x)

    # Get final shifts with optimal sample tilt
    final_shifts, final_correlation = _compute_shifts_with_sample_tilt(
        tilt_series=tilt_series,
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        lowpass_angstroms=lowpass_angstroms,
        sample_tilt=estimated_sample_tilt,
    )

    return final_shifts, estimated_sample_tilt


def _compute_shifts_with_sample_tilt(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angle: float,
    pixel_spacing_angstroms: float | None,
    lowpass_angstroms: float | None,
    sample_tilt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute shifts for a given sample tilt angle and return total correlation.

    Args:
        tilt_series: Stack of tilt images (b, h, w)
        tilt_angles: Nominal tilt angles (b,)
        tilt_axis_angle: Tilt axis angle in degrees
        pixel_spacing_angstroms: Pixel spacing in angstroms (optional)
        lowpass_angstroms: Low-pass filter cutoff in angstroms (optional)
        sample_tilt: Sample tilt angle in degrees

    Returns:
        shifts: Computed shifts (b, 2)
        total_correlation: Sum of all inter-image correlation peaks (as tensor)
    """
    # extract shape
    b, h, w = tilt_series.shape

    # sort input data by ORIGINAL tilt angle to maintain consistent reference
    tilt_angles = torch.as_tensor(tilt_angles).float()
    sorted_indices = torch.argsort(tilt_angles)
    sorted_tilt_series = tilt_series[sorted_indices]
    sorted_tilt_angles = tilt_angles[sorted_indices]

    # rfft & filter
    sorted_tilt_series_rfft = torch.fft.rfft2(sorted_tilt_series)
    if lowpass_angstroms is None or pixel_spacing_angstroms is None:
        lowpass_cycles_per_pixel = 0.5
    else:  # (Å px⁻¹) / (Å cycle⁻¹) = cycles px⁻¹
        lowpass_cycles_per_pixel = pixel_spacing_angstroms / lowpass_angstroms
    filter = bandpass_filter(
        low=0.025,
        high=lowpass_cycles_per_pixel,
        falloff=0.025,
        rfft=True,
        fftshift=False,
        image_shape=(h, w),
        device=tilt_series.device,
    )
    sorted_tilt_series_rfft *= filter
    sorted_tilt_series = torch.fft.irfft2(sorted_tilt_series_rfft, s=(h, w))

    # find index where tilt angle is closest to 0 (transition point)
    transition_idx = torch.argmin(torch.abs(sorted_tilt_angles))

    # apply sample tilt to get true tilt angles
    # true_tilt_angle = nominal_stage_angle + sample_tilt
    true_tilt_angles = sorted_tilt_angles + sample_tilt

    # create index arrays for positive and negative branches
    idx_positive = torch.arange(transition_idx, b, device=tilt_series.device)
    idx_negative = torch.arange(0, transition_idx + 1, device=tilt_series.device)

    # process positive branch: from least positive -> most positive (ascending abs angles)
    positive_branch_shifts, positive_correlation = _find_shifts_for_branch(
        tilt_series=sorted_tilt_series[idx_positive],
        tilt_angles=true_tilt_angles[idx_positive],
        tilt_axis_angle=tilt_axis_angle,
    )

    # process negative branch: reverse so abs angles ascend, then reverse result
    idx_negative_reversed = torch.flip(idx_negative, dims=[0])
    negative_branch_shifts, negative_correlation = _find_shifts_for_branch(
        tilt_series=sorted_tilt_series[idx_negative_reversed],
        tilt_angles=true_tilt_angles[idx_negative_reversed],
        tilt_axis_angle=tilt_axis_angle,
    )

    # Total correlation is sum from both branches
    total_correlation = positive_correlation + negative_correlation

    # cumsum to get absolute shifts from reference (first image in each branch)
    positive_branch_shifts = torch.cumsum(positive_branch_shifts, dim=0)
    negative_branch_shifts = torch.cumsum(negative_branch_shifts, dim=0)

    # assemble shifts for sorted tilt series
    sorted_shifts = torch.zeros(size=(b, 2), device=tilt_series.device)
    sorted_shifts[idx_positive] = positive_branch_shifts
    sorted_shifts[idx_negative_reversed] = negative_branch_shifts

    # put shifts back in original order
    shifts = torch.zeros_like(sorted_shifts)
    shifts[sorted_indices] = sorted_shifts

    return shifts, total_correlation


def _find_shifts_for_branch(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angle: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # grab dims
    h, w = tilt_series.shape[-2:]

    # Initialize shifts tensor
    leaf_shifts = torch.zeros(
        size=(len(tilt_series), 2), dtype=torch.float32, device=tilt_series.device
    )

    if len(tilt_series) < 2:
        return leaf_shifts, torch.tensor(0.0, device=tilt_series.device)

    # Extract all adjacent pairs at once using slicing
    imgs1 = tilt_series[:-1]  # (n_pairs, h, w)
    imgs2 = tilt_series[1:]  # (n_pairs, h, w)
    angles1 = tilt_angles[:-1]  # (n_pairs,)
    angles2 = tilt_angles[1:]  # (n_pairs,)

    # Compute scale factors for all pairs
    abs_angles1 = torch.abs(angles1)
    abs_angles2 = torch.abs(angles2)
    scale_factors = torch.cos(torch.deg2rad(abs_angles1)) / torch.cos(
        torch.deg2rad(abs_angles2)
    )

    # Stretch all img2s (using list comprehension as requested, not batched)
    imgs2_stretched = torch.stack(
        [
            apply_stretch_perpendicular_to_tilt_axis(
                img2, tilt_axis_angle=tilt_axis_angle, scale_factor=float(sf)
            )
            for img2, sf in zip(imgs2, scale_factors)
        ]
    )

    # taper image edges
    imgs1_tapered = taper_image_edges(imgs1)
    imgs2_tapered = taper_image_edges(imgs2_stretched)

    # zero pad images
    p = int(0.5 * min(h, w))
    imgs1_padded = F.pad(imgs1_tapered, [p] * 4)
    imgs2_padded = F.pad(imgs2_tapered, [p] * 4)

    # Batch FFT and cross-correlation
    h_pad, w_pad = imgs1_padded.shape[-2:]
    fft1 = torch.fft.rfftn(imgs1_padded, dim=(-2, -1))  # (n_pairs, h_pad, w_pad//2+1)
    fft2 = torch.fft.rfftn(imgs2_padded, dim=(-2, -1))  # (n_pairs, h_pad, w_pad//2+1)

    correlation_images = fft1 * torch.conj(fft2)
    correlation_images = torch.fft.irfftn(
        correlation_images, dim=(-2, -1), s=(h_pad, w_pad)
    )
    correlation_images = torch.fft.ifftshift(correlation_images, dim=(-2, -1))
    correlation_images /= h_pad * w_pad

    # Remove padding from correlation images
    correlation_images = F.pad(correlation_images, [-p] * 4)

    # Find shifts for each pair
    shifts = torch.stack(
        [get_shift_from_correlation_image(corr_img) for corr_img in correlation_images]
    )

    # Get correlation peaks for each pair
    correlation_peaks = correlation_images.max(dim=-1)[0].max(dim=-1)[0]  # (n_pairs,)
    total_correlation = correlation_peaks.sum()

    # Transform shifts to account for stretching (batched)
    transformed_shifts = transform_shifts_from_stretched_images(
        shift=shifts, tilt_axis_angle=tilt_axis_angle, scale_factor=scale_factors
    )

    # Store transformed shifts
    leaf_shifts[1:] = transformed_shifts

    return leaf_shifts, total_correlation


def _find_shift_between_adjacent_tilt_images(
    img1: torch.Tensor,
    img2: torch.Tensor,
    tilt_angle1: float,
    tilt_angle2: float,
    tilt_axis_angle: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Get absolute tilt angles
    abs_tilt_angle1, abs_tilt_angle2 = abs(tilt_angle1), abs(tilt_angle2)

    # Stretch image with larger absolute tilt angle (always img2)
    scale_factor = math.cos(np.deg2rad(abs_tilt_angle1)) / math.cos(
        np.deg2rad(abs_tilt_angle2)
    )
    img2_stretched = apply_stretch_perpendicular_to_tilt_axis(
        img2, tilt_axis_angle=tilt_axis_angle, scale_factor=scale_factor
    )
    img1, img2_stretched = (taper_image_edges(img1), taper_image_edges(img2_stretched))

    # pad images for cross-correlation
    p = int(0.5 * min(img1.shape[-2:]))
    img1 = F.pad(img1, [p] * 4, value=img1.mean())
    img2_stretched = F.pad(img2_stretched, [p] * 4, value=img2_stretched.mean())

    # Calculate correlation and get shift
    correlation_image = calculate_cross_correlation(img1, img2_stretched)
    # remove padding from the result
    correlation_image = F.pad(correlation_image, [-p] * 4)

    shift = get_shift_from_correlation_image(correlation_image)

    # Get the peak correlation value as a quality metric
    correlation_peak = correlation_image.max()

    # Transform shift to account for the fact that img2 was stretched
    transformed_shift = transform_shifts_from_stretched_images(
        shift=shift, tilt_axis_angle=tilt_axis_angle, scale_factor=scale_factor
    )

    return transformed_shift, correlation_peak
