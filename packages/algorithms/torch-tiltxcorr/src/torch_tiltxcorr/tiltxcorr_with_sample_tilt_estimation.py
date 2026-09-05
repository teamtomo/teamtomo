import math

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize_scalar
from torch_tilt_series import preprocess_tilt_series_images

from torch_tiltxcorr.utils import (
    calculate_cross_correlation,
    get_shift_from_correlation_image,
    taper_image_edges,
    transform_shifts_from_stretched_images,
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
    preprocess: bool = True,
) -> tuple[torch.Tensor, float]:  # (b, 2) yx shifts and optimal sample tilt
    """
    Estimate stage shifts and sample tilt by maximizing sum of inter-image tilted cross correlations.

    The sample tilt estimate represents sample tilt about the stage in the microscope.
    E.g. if the sample is physically tilted +5°, then at nominal 0° the beam
    sees the sample at +5°, and estimate_sample_tilt = +5°.

    Uses scipy's Brent's method (bounded) for optimization.

    `preprocess` can be turned off if the caller has already preprocessed
    `tilt_series` themselves (e.g. once, for reuse across multiple calls).
    `torch_tiltxcorr.utils.taper_image_edges` is always applied to each image
    pair before cross-correlating and is not configurable.

    Args:
        tilt_series: Stack of tilt images (b, h, w)
        tilt_angles: Nominal stage tilt angles for each image (b,)
        tilt_axis_angle: Angle of tilt axis in degrees
        pixel_spacing_angstroms: Pixel spacing in angstroms
        lowpass_angstroms: Low-pass filter cutoff in angstroms
        sample_tilt_range: (min, max) range of sample tilt angles to search in degrees
        max_iter: Maximum iterations for Brent's method optimizer
        preprocess: If True (default), preprocess `tilt_series` via
            `torch_tilt_series.preprocess_tilt_series_images` (background
            plane subtraction, a bandpass filter with `low` fixed at 0.025
            and `high` from `pixel_spacing_angstroms`/`lowpass_angstroms`
            above, and central-crop normalization) before finding shifts.

    Returns:
        shifts: Optimal shifts for each image (b, 2) yx coords
        sample_tilt: Optimal sample tilt angle in degrees
    """
    sample_tilt_min, sample_tilt_max = sample_tilt_range

    if lowpass_angstroms is None or pixel_spacing_angstroms is None:
        lowpass_cycles_per_pixel = 0.5
    else:  # (Å px⁻¹) / (Å cycle⁻¹) = cycles px⁻¹
        lowpass_cycles_per_pixel = pixel_spacing_angstroms / lowpass_angstroms

    if preprocess:
        tilt_series = preprocess_tilt_series_images(
            tilt_series, low=0.025, high=lowpass_cycles_per_pixel, falloff=0.025
        )

    # Track history for correlation curve
    sample_tilt_history = []
    correlation_history = []

    def objective(sample_tilt: float) -> float:
        """Objective function: negative correlation (to minimize)."""
        _, total_correlation = _compute_shifts_with_sample_tilt(
            tilt_series=tilt_series,
            tilt_angles=tilt_angles,
            tilt_axis_angle=tilt_axis_angle,
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
        sample_tilt=estimated_sample_tilt,
    )

    return final_shifts, estimated_sample_tilt


def _compute_shifts_with_sample_tilt(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angle: float,
    sample_tilt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute shifts for a given sample tilt angle and return total correlation.

    Args:
        tilt_series: Stack of tilt images (b, h, w), already preprocessed
        tilt_angles: Nominal tilt angles (b,)
        tilt_axis_angle: Tilt axis angle in degrees
        sample_tilt: Sample tilt angle in degrees

    Returns:
        shifts: Computed shifts (b, 2)
        total_correlation: Sum of all inter-image correlation peaks (as tensor)
    """
    # extract shape
    b = tilt_series.shape[0]

    # sort input data by ORIGINAL tilt angle to maintain consistent reference
    tilt_angles = torch.as_tensor(tilt_angles).float()
    sorted_indices = torch.argsort(tilt_angles)
    sorted_tilt_series = tilt_series[sorted_indices]
    sorted_tilt_angles = tilt_angles[sorted_indices]

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
    # Initialize shifts tensor
    leaf_shifts = torch.zeros(
        size=(len(tilt_series), 2), dtype=torch.float32, device=tilt_series.device
    )
    total_correlation = torch.tensor(0.0, device=tilt_series.device)

    # Iterate over adjacent pairs one at a time, rather than batching them into
    # a single large FFT, to keep peak memory usage manageable for large images.
    for i in range(1, len(tilt_series)):
        shift, correlation_peak = _find_shift_between_adjacent_tilt_images(
            img1=tilt_series[i - 1],
            img2=tilt_series[i],
            tilt_angle1=float(tilt_angles[i - 1]),
            tilt_angle2=float(tilt_angles[i]),
            tilt_axis_angle=tilt_axis_angle,
        )
        leaf_shifts[i] = shift
        total_correlation = total_correlation + correlation_peak

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
    img1 = F.pad(img1, [p] * 4)
    img2_stretched = F.pad(img2_stretched, [p] * 4)

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
