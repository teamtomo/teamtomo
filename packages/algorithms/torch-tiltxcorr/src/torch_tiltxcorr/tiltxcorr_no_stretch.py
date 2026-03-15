import torch
from torch.nn import functional as F
from torch_fourier_filter.bandpass import bandpass_filter

from torch_tiltxcorr.utils import (
    taper_image_edges,
    calculate_cross_correlation,
    get_shift_from_correlation_image,
)


def tiltxcorr_no_stretch(
    tilt_series: torch.Tensor,
    tilt_angles: torch.Tensor,  # (b, )
    pixel_spacing_angstroms: float | None = None,
    lowpass_angstroms: float | None = None,
) -> torch.Tensor:
    """Find coarse shifts of images without stretching along tilt axis."""
    # extract shape
    b, h, w = tilt_series.shape

    # sort input data by tilt angle
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
    # grab dims
    h, w = tilt_series.shape[-2:]

    # Initialize shifts tensor
    leaf_shifts = torch.zeros(
        size=(len(tilt_series), 2), dtype=torch.float32, device=tilt_series.device
    )

    if len(tilt_series) < 2:
        return leaf_shifts

    # Extract all adjacent pairs at once using slicing
    imgs1 = tilt_series[:-1]  # (n_pairs, h, w)
    imgs2 = tilt_series[1:]  # (n_pairs, h, w)

    # Batch taper edges
    imgs1_tapered = taper_image_edges(imgs1)
    imgs2_tapered = taper_image_edges(imgs2)

    # Batch pad images
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

    # Store shifts
    leaf_shifts[1:] = shifts

    return leaf_shifts


def _find_shift_between_adjacent_tilt_images_no_stretch(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> torch.Tensor:
    img1, img2 = (taper_image_edges(img1), taper_image_edges(img2))
    # pad images for cross-correlation
    p = int(0.5 * min(img1.shape[-2:]))
    img1 = F.pad(img1, [p] * 4, value=img1.mean())
    img2 = F.pad(img2, [p] * 4, value=img2.mean())
    correlation_image = calculate_cross_correlation(img1, img2)
    # remove padding from the result
    correlation_image = F.pad(correlation_image, [-p] * 4)
    shift = get_shift_from_correlation_image(correlation_image)
    return shift
