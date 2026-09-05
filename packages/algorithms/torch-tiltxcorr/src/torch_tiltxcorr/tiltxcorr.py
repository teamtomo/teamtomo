import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_tilt_series import preprocess_tilt_series_images

from torch_tiltxcorr.utils import (
    apply_stretch_perpendicular_to_tilt_axis,
    calculate_cross_correlation,
    get_shift_from_correlation_image,
    taper_image_edges,
    transform_shifts_from_stretched_images,
)


def tiltxcorr(
    tilt_series: torch.Tensor,  # (b, h, w)
    tilt_angles: torch.Tensor,  # (b, )
    tilt_axis_angle: float,
    pixel_spacing_angstroms: float | None = None,
    lowpass_angstroms: float | None = None,
    preprocess: bool = True,
) -> torch.Tensor:  # (b, 2) yx shifts
    """Estimate tilt-series shifts by cross-correlating stretch-corrected adjacent tilts.

    `preprocess` can be turned off if the caller has already preprocessed
    `tilt_series` themselves (e.g. once, for reuse across multiple calls).
    `torch_tiltxcorr.utils.taper_image_edges` is always applied to each image
    pair before cross-correlating and is not configurable.

    Parameters
    ----------
    preprocess : bool
        If True (default), preprocess `tilt_series` via
        `torch_tilt_series.preprocess_tilt_series_images` (background plane
        subtraction, a bandpass filter with `low` fixed at 0.025 and `high`
        from `pixel_spacing_angstroms`/`lowpass_angstroms` below, and
        central-crop normalization) before finding shifts.
    """
    # extract shape
    b = tilt_series.shape[0]

    # sort input data by tilt angle
    tilt_angles = torch.as_tensor(tilt_angles).float()
    sorted_indices = torch.argsort(tilt_angles)
    sorted_tilt_series = tilt_series[sorted_indices]
    sorted_tilt_angles = tilt_angles[sorted_indices]

    if lowpass_angstroms is None or pixel_spacing_angstroms is None:
        lowpass_cycles_per_pixel = 0.5
    else:  # (Å px⁻¹) / (Å cycle⁻¹) = cycles px⁻¹
        lowpass_cycles_per_pixel = pixel_spacing_angstroms / lowpass_angstroms

    if preprocess:
        sorted_tilt_series = preprocess_tilt_series_images(
            sorted_tilt_series, low=0.025, high=lowpass_cycles_per_pixel, falloff=0.025
        )

    # find index where tilt angle is closest to 0 (transition point)
    transition_idx = torch.argmin(torch.abs(sorted_tilt_angles))

    # create index arrays for positive and negative branches
    idx_positive = torch.arange(transition_idx, b, device=tilt_series.device)
    idx_negative = torch.arange(0, transition_idx + 1, device=tilt_series.device)

    # process positive branch: from least positive -> most positive (ascending abs angles)
    positive_branch_shifts = _find_shifts_for_branch(
        tilt_series=sorted_tilt_series[idx_positive],
        tilt_angles=sorted_tilt_angles[idx_positive],
        tilt_axis_angle=tilt_axis_angle,
    )

    # process negative branch: reverse so abs angles ascend, then reverse result
    idx_negative_reversed = torch.flip(idx_negative, dims=[0])
    negative_branch_shifts = _find_shifts_for_branch(
        tilt_series=sorted_tilt_series[idx_negative_reversed],
        tilt_angles=sorted_tilt_angles[idx_negative_reversed],
        tilt_axis_angle=tilt_axis_angle,
    )

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
    return shifts


def _find_shifts_for_branch(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,
    tilt_axis_angle: float,
) -> torch.Tensor:
    # Initialize shifts tensor
    leaf_shifts = torch.zeros(
        size=(len(tilt_series), 2), dtype=torch.float32, device=tilt_series.device
    )

    # Iterate over adjacent pairs one at a time, rather than batching them into
    # a single large FFT, to keep peak memory usage manageable for large images.
    for i in range(1, len(tilt_series)):
        leaf_shifts[i] = _find_shift_between_adjacent_tilt_images(
            img1=tilt_series[i - 1],
            img2=tilt_series[i],
            tilt_angle1=float(tilt_angles[i - 1]),
            tilt_angle2=float(tilt_angles[i]),
            tilt_axis_angle=tilt_axis_angle,
        )

    return leaf_shifts


def _find_shift_between_adjacent_tilt_images(
    img1: torch.Tensor,
    img2: torch.Tensor,
    tilt_angle1: float,
    tilt_angle2: float,
    tilt_axis_angle: float,
) -> torch.Tensor:
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

    # Transform shift to account for the fact that img2 was stretched
    transformed_shift = transform_shifts_from_stretched_images(
        shift=shift, tilt_axis_angle=tilt_axis_angle, scale_factor=scale_factor
    )

    return transformed_shift
