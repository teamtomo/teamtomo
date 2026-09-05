import torch
from torch.nn import functional as F
from torch_tilt_series import preprocess_tilt_series_images

from torch_tiltxcorr.utils import (
    calculate_cross_correlation,
    get_shift_from_correlation_image,
    taper_image_edges,
)


def tiltxcorr_no_stretch(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,  # (b, )
    pixel_spacing_angstroms: float | None = None,
    lowpass_angstroms: float | None = None,
    preprocess: bool = True,
) -> torch.Tensor:
    """Find coarse shifts of images without stretching along tilt axis.

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
    positive_branch_shifts = _find_shifts_for_branch_no_stretch(
        tilt_series=sorted_tilt_series[idx_positive]
    )

    # process negative branch: reverse so abs angles ascend, then reverse result
    idx_negative_reversed = torch.flip(idx_negative, dims=[0])
    negative_branch_shifts = _find_shifts_for_branch_no_stretch(
        tilt_series=sorted_tilt_series[idx_negative_reversed]
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


def _find_shifts_for_branch_no_stretch(
    tilt_series: torch.Tensor,
) -> torch.Tensor:
    # Initialize shifts tensor
    leaf_shifts = torch.zeros(
        size=(len(tilt_series), 2), dtype=torch.float32, device=tilt_series.device
    )

    # Iterate over adjacent pairs one at a time, rather than batching them into
    # a single large FFT, to keep peak memory usage manageable for large images.
    for i in range(1, len(tilt_series)):
        leaf_shifts[i] = _find_shift_between_adjacent_tilt_images_no_stretch(
            img1=tilt_series[i - 1],
            img2=tilt_series[i],
        )

    return leaf_shifts


def _find_shift_between_adjacent_tilt_images_no_stretch(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> torch.Tensor:
    img1, img2 = (taper_image_edges(img1), taper_image_edges(img2))
    # pad images for cross-correlation
    p = int(0.5 * min(img1.shape[-2:]))
    img1 = F.pad(img1, [p] * 4)
    img2 = F.pad(img2, [p] * 4)
    correlation_image = calculate_cross_correlation(img1, img2)
    # remove padding from the result
    correlation_image = F.pad(correlation_image, [-p] * 4)
    shift = get_shift_from_correlation_image(correlation_image)
    return shift
