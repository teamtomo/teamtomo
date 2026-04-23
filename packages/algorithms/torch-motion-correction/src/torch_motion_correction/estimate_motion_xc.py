"""Estimate motion to a deformation field using cross correlation."""

import einops
import torch
from scipy.signal import savgol_filter
from torch_grid_utils.patch_grid import patch_grid_lazy

from torch_motion_correction.correct_motion import correct_motion, correct_motion_fast
from torch_motion_correction.deformation_field import DeformationField
from torch_motion_correction.types import (
    FourierFilterConfig,
    PatchSamplingConfig,
    XCRefinementConfig,
)
from torch_motion_correction.utils import normalize_image, prepare_patch_filters


def estimate_global_motion(
    image: torch.Tensor,  # (t, h, w)
    pixel_spacing: float,  # angstroms
    reference_frame: int | None = None,
    fourier_filter: FourierFilterConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> DeformationField:
    """Estimate motion using cross-correlation for the whole image.

    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    pixel_spacing: float
        Pixel spacing in angstroms
    reference_frame: int, optional
        Frame index to use as reference. If None, uses middle frame.
    fourier_filter: FourierFilterConfig, optional
        Fourier-space filtering parameters. Defaults to ``FourierFilterConfig()``
        when None.
    device: torch.device, optional
        Device for computation
    verbose: bool
        Whether to print progress information. Default is False.

    Returns
    -------
    DeformationField
        Global deformation field with shape (2, t, 1, 1), one shift per frame.
    """
    if fourier_filter is None:
        fourier_filter = FourierFilterConfig()

    # b_factor = fourier_filter.b_factor
    # frequency_range = fourier_filter.frequency_range

    if device is None:
        device = image.device
    else:
        image = image.to(device)

    t, h, w = image.shape

    # Use middle frame as reference if not specified
    if reference_frame is None:
        reference_frame = t // 2

    if verbose:
        print(
            f"Cross-correlation whole image: using frame {reference_frame} as reference"
        )

    # Normalize image
    image = normalize_image(image)

    mask, b_factor_envelope, bandpass = prepare_patch_filters(
        shape=(h, w),
        pixel_spacing=pixel_spacing,
        fourier_filter=fourier_filter,
        device=device,
    )

    # Apply mask, FFT, and filters to all frames at once
    fft_images = torch.fft.rfftn(image * mask, dim=(-2, -1))
    filtered_fft = fft_images * bandpass * b_factor_envelope

    # Reference frame
    reference_fft = filtered_fft[reference_frame]

    # Calculate shifts for each frame
    shifts = torch.zeros((t, 2), device=device)  # (y, x) shifts

    for frame_idx in range(t):
        if frame_idx == reference_frame:
            continue  # Reference frame has zero shift

        # Cross-correlation in frequency domain
        frame_fft = filtered_fft[frame_idx]
        cross_corr_fft = torch.conj(reference_fft) * frame_fft
        cross_corr = torch.fft.irfftn(cross_corr_fft, s=(h, w))

        # Find peak position
        peak_idx = torch.argmax(cross_corr.flatten())
        peak_y, peak_x = divmod(peak_idx.item(), w)

        # Convert to shifts (handle wraparound)
        shift_y = peak_y if peak_y <= h // 2 else peak_y - h
        shift_x = peak_x if peak_x <= w // 2 else peak_x - w

        shifts[frame_idx] = torch.tensor([shift_y, shift_x], device=device)

    if verbose:
        print(
            f"Estimated shifts range: "
            f"y=[{shifts[:, 0].min():.1f}, {shifts[:, 0].max():.1f}], "
            f"x=[{shifts[:, 1].min():.1f}, {shifts[:, 1].max():.1f}]"
        )

    return DeformationField.from_frame_shifts(
        shifts=shifts, pixel_spacing=pixel_spacing, device=device
    )


def estimate_motion_cross_correlation_patches(
    image: torch.Tensor,  # (t, h, w)
    pixel_spacing: float,  # angstroms
    patch_sampling: PatchSamplingConfig,
    reference_frame: int | None = None,
    reference_strategy: str = "mean_except_current",
    fourier_filter: FourierFilterConfig | None = None,
    refinement: XCRefinementConfig | None = None,
    initial_deformation_field: DeformationField | None = None,
    device: torch.device | None = None,
    verbose: bool = False,
) -> tuple[DeformationField, torch.Tensor]:
    """Estimate motion using cross-correlation for the patches.

    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    pixel_spacing: float
        Pixel spacing in angstroms
    patch_sampling: PatchSamplingConfig
        Patch extraction configuration, including patch shape and overlap fraction.
    reference_frame: int, optional
        Frame index to use as reference. If None, uses middle frame.
    reference_strategy: str
        Strategy for reference frame selection:
        - "middle_frame": Use the middle frame as reference for all frames
        - "mean_except_current": Use mean of all frames except current frame
        as reference
    fourier_filter: FourierFilterConfig, optional
        Fourier-space filtering parameters. Defaults to ``FourierFilterConfig()``
        when None.
    refinement: XCRefinementConfig, optional
        Post-processing configuration (sub-pixel refinement, temporal smoothing,
        outlier rejection). Defaults to ``XCRefinementConfig()`` when None.
    initial_deformation_field: DeformationField | None
        Optional deformation field to apply to the image before motion estimation.
        If provided, the image will be corrected first, and the newly calculated
        motion will be added to this field (cumulative correction).
        If shape[-2:] == (1, 1), uses ``correct_motion_fast``; otherwise uses
        ``correct_motion``.
    device: torch.device, optional
        Device for computation
    verbose: bool
        Whether to print progress information. Default is False.

    Returns
    -------
    DeformationField
        Estimated deformation field with shape (2, t, gh, gw).
    data_patch_positions: torch.Tensor
        (t, gh, gw, 3) patch center positions
    """
    if fourier_filter is None:
        fourier_filter = FourierFilterConfig()
    if refinement is None:
        refinement = XCRefinementConfig()

    # Deconstruct config objects
    # b_factor = fourier_filter.b_factor
    # frequency_range = fourier_filter.frequency_range
    ph, pw = patch_sampling.patch_shape
    step_h, step_w = patch_sampling.patch_step
    sub_pixel_refinement = refinement.sub_pixel_refinement
    temporal_smoothing = refinement.temporal_smoothing
    smoothing_window_size = refinement.smoothing_window_size
    outlier_rejection = refinement.outlier_rejection
    outlier_threshold = refinement.outlier_threshold

    if device is None:
        device = image.device
    else:
        image = image.to(device)

    t, _h, _w = image.shape

    # Use middle frame as reference if not specified
    if reference_frame is None:
        reference_frame = t // 2

    image = normalize_image(image)

    if verbose:
        if reference_strategy == "middle_frame":
            print(
                f"Cross-correlation patches: using frame {reference_frame} as reference"
            )
        else:
            print(
                "Cross-correlation patches: using mean of all frames except "
                "current frame as reference"
            )

    # Apply initial deformation field if provided
    if initial_deformation_field is not None:
        initial_deformation_field = initial_deformation_field.to(device)

        # Check if it's a single patch deformation field (last two dims are 1, 1)
        if initial_deformation_field.data.shape[-2:] == (1, 1):
            if verbose:
                print(
                    "Applying single patch deformation field using correct_motion_fast"
                )
            image = correct_motion_fast(
                image=image,
                deformation_field=initial_deformation_field,
                device=device,
                verbose=verbose,
            )
        else:
            if verbose:
                print("Applying full deformation field using correct_motion")
            image = correct_motion(
                image=image,
                deformation_field=initial_deformation_field,
                pixel_spacing=pixel_spacing,
                device=device,
            )

    # Split into patch grid using PatchSamplingConfig
    lazy_patch_grid, data_patch_positions = patch_grid_lazy(
        images=image,
        patch_shape=(1, ph, pw),
        patch_step=(1, step_h, step_w),
        distribute_patches=patch_sampling.distribute_patches,
    )
    gh, gw = data_patch_positions.shape[1:3]
    if verbose:
        print(f"Number of patches per frame: {gh * gw}")

    # Prepare filters (only need to do this once)
    mask, b_factor_envelope, bandpass = prepare_patch_filters(
        shape=(ph, pw),
        pixel_spacing=pixel_spacing,
        fourier_filter=fourier_filter,
        device=device,
    )

    # Initialize or update raw deformation field tensor
    if initial_deformation_field is None:
        raw_field = torch.zeros((2, t, gh, gw), device=device)
    else:
        raw_field = initial_deformation_field.resample((t, gh, gw)).data
        if verbose:
            print(
                "Using existing deformation field as base for cumulative motion "
                "correction"
            )

    # Use local variable for the raw tensor during frame processing
    deformation_field_tensor = raw_field

    # Process each frame individually to manage memory
    for frame_idx in range(t):
        if reference_strategy == "middle_frame" and frame_idx == reference_frame:
            continue  # Reference frame has zero shift

        if verbose:
            print(f"Processing frame {frame_idx}/{t - 1}")

        # Determine reference patches based on strategy
        if reference_strategy == "middle_frame":
            # Use middle frame as reference
            ref_patches = lazy_patch_grid[reference_frame]  # (1, gh, gw, 1, ph, pw)
            ref_patches = einops.rearrange(
                ref_patches, "1 gh gw 1 ph pw -> gh gw ph pw"
            )
        elif reference_strategy == "mean_except_current":
            # Use mean of all frames except current frame (computed
            # incrementally to save memory)
            ref_patches = None
            count = 0
            for other_frame_idx in range(t):
                if other_frame_idx != frame_idx:
                    other_patches = lazy_patch_grid[
                        other_frame_idx
                    ]  # (1, gh, gw, 1, ph, pw)
                    other_patches = einops.rearrange(
                        other_patches, "1 gh gw 1 ph pw -> gh gw ph pw"
                    )
                    if ref_patches is None:
                        ref_patches = other_patches.clone()
                    else:
                        ref_patches += other_patches
                    count += 1
            ref_patches = ref_patches / count
        else:
            raise ValueError(f"Unknown reference_strategy: {reference_strategy}")

        # Get patches for current frame only
        frame_patches = lazy_patch_grid[frame_idx]  # (1, gh, gw, 1, ph, pw)
        frame_patches = einops.rearrange(
            frame_patches, "1 gh gw 1 ph pw -> gh gw ph pw"
        )

        # Apply mask and filters to reference patches
        ref_patches *= mask
        ref_patches_fft = torch.fft.rfftn(ref_patches, dim=(-2, -1))
        ref_patches_fft = ref_patches_fft * bandpass * b_factor_envelope

        # Apply mask and filters to frame patches
        frame_patches *= mask
        frame_patches_fft = torch.fft.rfftn(frame_patches, dim=(-2, -1))
        frame_patches_fft = frame_patches_fft * bandpass * b_factor_envelope

        # Vectorized cross-correlation for all patches at once
        cross_corr_fft = torch.conj(ref_patches_fft) * frame_patches_fft
        cross_corr = torch.fft.irfftn(cross_corr_fft, s=(ph, pw))

        # Vectorized peak finding for all patches
        # Reshape to (gh*gw, ph*pw) for vectorized argmax
        cross_corr_flat = cross_corr.view(gh * gw, ph * pw)
        peak_indices = torch.argmax(cross_corr_flat, dim=1)  # (gh*gw,)

        if sub_pixel_refinement:
            # Use sub-pixel peak finding
            peak_y, peak_x = _apply_sub_pixel_refinement(
                cross_corr_flat, peak_indices, ph, pw
            )
        else:
            # Use integer peak finding
            peak_y = peak_indices // pw
            peak_x = peak_indices % pw

        # Convert to shifts (handle wraparound)
        shift_y = torch.where(peak_y <= ph // 2, peak_y, peak_y - ph)
        shift_x = torch.where(peak_x <= pw // 2, peak_x, peak_x - pw)

        # Reshape back to grid
        shift_y = shift_y.view(gh, gw)
        shift_x = shift_x.view(gh, gw)

        # Apply outlier rejection if enabled
        if outlier_rejection:
            shift_y, shift_x = _apply_outlier_rejection(
                shift_y, shift_x, outlier_threshold, frame_idx, verbose=verbose
            )

        # Add shifts to existing deformation field (cumulative motion correction)
        # Convert pixels to Angstroms
        shift_y = shift_y * pixel_spacing
        shift_x = shift_x * pixel_spacing
        deformation_field_tensor[0, frame_idx, :, :] += shift_y
        deformation_field_tensor[1, frame_idx, :, :] += shift_x

    # Apply temporal smoothing if enabled
    if temporal_smoothing:
        if verbose:
            print(
                f"Applying temporal smoothing with window size {smoothing_window_size}"
            )
        deformation_field_tensor = _apply_temporal_smoothing(
            deformation_field_tensor, smoothing_window_size, device
        )
        if verbose:
            def_min_y = deformation_field_tensor[0].min()
            def_max_y = deformation_field_tensor[0].max()
            def_min_x = deformation_field_tensor[1].min()
            def_max_x = deformation_field_tensor[1].max()
            print(
                f"After temporal smoothing - range: "
                f"y=[{def_min_y:.1f}, {def_max_y:.1f}], "
                f"x=[{def_min_x:.1f}, {def_max_x:.1f}]"
            )

    if verbose:
        def_min_y = deformation_field_tensor[0].min()
        def_max_y = deformation_field_tensor[0].max()
        def_min_x = deformation_field_tensor[1].min()
        def_max_x = deformation_field_tensor[1].max()
        print(
            f"Estimated deformation field range: "
            f"y=[{def_min_y:.1f}, {def_max_y:.1f}], "
            f"x=[{def_min_x:.1f}, {def_max_x:.1f}]"
        )
        print(f"Estimated deformation field shape: {deformation_field_tensor.shape}")

    deformation_field_tensor = deformation_field_tensor - torch.mean(
        deformation_field_tensor
    )
    return DeformationField(data=deformation_field_tensor), data_patch_positions


def _apply_sub_pixel_refinement(
    cross_corr: torch.Tensor,
    peak_indices: torch.Tensor,
    patch_height: int,
    patch_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply sub-pixel refinement to peak positions using parabolic fitting.

    Parameters
    ----------
    cross_corr: torch.Tensor
        Cross-correlation values (gh*gw, ph*pw)
    peak_indices: torch.Tensor
        Integer peak indices (gh*gw,)
    patch_height: int
        Height of patches
    patch_width: int
        Width of patches

    Returns
    -------
    peak_y: torch.Tensor
        Refined y coordinates (gh*gw,)
    peak_x: torch.Tensor
        Refined x coordinates (gh*gw,)
    """
    num_patches = cross_corr.shape[0]

    # Convert flat indices to 2D coordinates
    peak_y_int = peak_indices // patch_width
    peak_x_int = peak_indices % patch_width

    # Initialize refined coordinates
    peak_y_refined = peak_y_int.float()
    peak_x_refined = peak_x_int.float()

    # Reshape cross_corr to 3D for easier neighborhood access
    cross_corr_3d = cross_corr.view(num_patches, patch_height, patch_width)

    # Apply sub-pixel refinement to each patch using parabolic fitting
    for i in range(num_patches):
        y_int = peak_y_int[i].item()
        x_int = peak_x_int[i].item()

        # Check bounds for parabolic fit (need 3x3 neighborhood)
        if 1 <= y_int < patch_height - 1 and 1 <= x_int < patch_width - 1:
            # Extract 3x3 neighborhood around peak
            y_vals = cross_corr_3d[i, y_int - 1 : y_int + 2, x_int]
            x_vals = cross_corr_3d[i, y_int, x_int - 1 : x_int + 2]

            # Parabolic fit for y direction
            if y_vals[2] != y_vals[0]:
                y_offset = (
                    0.5
                    * (y_vals[0] - y_vals[2])
                    / (y_vals[0] - 2 * y_vals[1] + y_vals[2])
                )
                peak_y_refined[i] += y_offset

            # Parabolic fit for x direction
            if x_vals[2] != x_vals[0]:
                x_offset = (
                    0.5
                    * (x_vals[0] - x_vals[2])
                    / (x_vals[0] - 2 * x_vals[1] + x_vals[2])
                )
                peak_x_refined[i] += x_offset

    return peak_y_refined, peak_x_refined


def _apply_temporal_smoothing(
    deformation_field: torch.Tensor, window_size: int, device: torch.device
) -> torch.Tensor:
    """Apply temporal smoothing to deformation field across frames for each patch.

    Parameters
    ----------
    deformation_field: torch.Tensor
        Deformation field (2, t, gh, gw)
    window_size: int
        Window size for smoothing (must be odd)
    device: torch.device
        Device for computation

    Returns
    -------
    smoothed_deformation_field: torch.Tensor
        Smoothed deformation field (2, t, gh, gw)
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size

    # Ensure window size doesn't exceed number of frames
    t = deformation_field.shape[1]
    window_size = min(window_size, t)

    if window_size < 3:
        return deformation_field  # No smoothing for very small windows

    # Apply Savitzky-Golay smoothing to each patch independently
    smoothed_field = deformation_field.clone()

    # Process each patch position
    for gy in range(deformation_field.shape[2]):
        for gx in range(deformation_field.shape[3]):
            # Extract time series for this patch
            y_series = deformation_field[0, :, gy, gx].detach().cpu().numpy()
            x_series = deformation_field[1, :, gy, gx].detach().cpu().numpy()

            # Apply Savitzky-Golay smoothing
            if len(y_series) >= window_size:
                smoothed_y = savgol_filter(y_series, window_size, 1)
                smoothed_x = savgol_filter(x_series, window_size, 1)

                # Convert back to torch and update
                smoothed_field[0, :, gy, gx] = torch.from_numpy(smoothed_y).to(device)
                smoothed_field[1, :, gy, gx] = torch.from_numpy(smoothed_x).to(device)

    return smoothed_field


def _apply_outlier_rejection(
    shift_y: torch.Tensor,  # (gh, gw) y shifts
    shift_x: torch.Tensor,  # (gh, gw) x shifts
    outlier_threshold: float,  # standard deviations from median
    frame_idx: int,  # For logging
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply outlier rejection to patch shifts using standard deviation from median.

    If either X or Y shift for a patch is an outlier, both X and Y shifts
    for that patch are replaced with the mean of valid patches.

    Parameters
    ----------
    shift_y: torch.Tensor
        Y shifts for all patches in the frame (gh, gw)
    shift_x: torch.Tensor
        X shifts for all patches in the frame (gh, gw)
    outlier_threshold: float
        Threshold in standard deviations from median for outlier detection
    frame_idx: int
        Frame index for logging purposes
    verbose: bool
        Whether to print outlier counts. Default is False.

    Returns
    -------
    corrected_shift_y: torch.Tensor
        Y shifts with outliers replaced by mean (gh, gw)
    corrected_shift_x: torch.Tensor
        X shifts with outliers replaced by mean (gh, gw)
    """
    # Flatten for easier processing
    shift_y_flat = shift_y.flatten()
    shift_x_flat = shift_x.flatten()

    # Calculate medians
    median_y = torch.median(shift_y_flat)
    median_x = torch.median(shift_x_flat)

    # Calculate standard deviations
    std_y = torch.std(shift_y_flat)
    std_x = torch.std(shift_x_flat)

    # Avoid division by zero
    std_y = torch.max(std_y, torch.tensor(1e-6, device=shift_y.device))
    std_x = torch.max(std_x, torch.tensor(1e-6, device=shift_x.device))

    # Calculate z-scores (standard deviations from median)
    z_score_y = torch.abs(shift_y_flat - median_y) / std_y
    z_score_x = torch.abs(shift_x_flat - median_x) / std_x

    # Identify outliers in either direction
    outliers_y = z_score_y > outlier_threshold
    outliers_x = z_score_x > outlier_threshold

    # If either X or Y is an outlier, mark the entire patch as outlier
    outliers_combined = outliers_y | outliers_x

    # Count outliers
    num_outliers_y = outliers_y.sum().item()
    num_outliers_x = outliers_x.sum().item()
    num_outliers_combined = outliers_combined.sum().item()
    total_patches = shift_y_flat.numel()

    if verbose and num_outliers_combined > 0:
        print(
            f"Frame {frame_idx}: Found {num_outliers_y} Y outliers, "
            f"{num_outliers_x} X outliers, {num_outliers_combined} patches "
            f"rejected (either X or Y outlier) out of {total_patches} patches"
        )

    # Calculate means for replacement (excluding patches that are outliers
    # in either direction)
    valid_y = shift_y_flat[~outliers_combined]
    valid_x = shift_x_flat[~outliers_combined]

    mean_y = torch.mean(valid_y) if len(valid_y) > 0 else median_y
    mean_x = torch.mean(valid_x) if len(valid_x) > 0 else median_x

    # Replace outliers with means (replace both X and Y if either is an outlier)
    corrected_shift_y_flat = shift_y_flat.clone()
    corrected_shift_x_flat = shift_x_flat.clone()

    corrected_shift_y_flat[outliers_combined] = mean_y
    corrected_shift_x_flat[outliers_combined] = mean_x

    # Reshape back to grid
    corrected_shift_y = corrected_shift_y_flat.view(shift_y.shape)
    corrected_shift_x = corrected_shift_x_flat.view(shift_x.shape)

    return corrected_shift_y, corrected_shift_x
