"""Dose weighting functions for Fourier filtering.

This module provides dose weighting implementations with different
memory efficiency characteristics:

1. `dose_weight_movie()` - Main function with optional memory-efficient processing
2. `dose_weight_movie_memory_efficient_2d()` - Memory-efficient chunked processing
3. `dose_weight_2d()` - Core dose weighting for individual frames/stacks

Memory Usage Comparison:
- Original: ~5 * n_frames * h * w * 8 bytes (all frames in memory)
- Memory Efficient: ~2 * chunk_size * h * w * 8 bytes (chunked processing)

For large movies that don't fit in memory, use `memory_efficient=True` in
`dose_weight_movie()` or call `dose_weight_movie_memory_efficient_2d()` directly.
"""

import einops
import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid


def critical_exposure(
    fft_freq: torch.Tensor,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
) -> torch.Tensor:
    """
    Calculate the critical exposure using the Grant and Grigorieff 2015 formula.

    Ne = a * fft_freq^b + c

    Parameters
    ----------
    fft_freq: torch.Tensor
        The frequency grid of the Fourier transform.
    a: float
        The a parameter for the critical exposure formula. Default is 0.245.
    b: float
        The b parameter for the critical exposure formula. Default is -1.665.
    c: float
        The c parameter for the critical exposure formula. Default is 2.81.

    Returns
    -------
        The critical exposure for the given frequency grid.
    """
    eps = 1e-10
    Ne = a * torch.pow(fft_freq.clamp(min=eps), b) + c
    return Ne


def critical_exposure_bfactor(fft_freq: torch.Tensor, bfactor: float) -> torch.Tensor:
    """
    Calculate the critical exposure using a user defined B-factor.

    Parameters
    ----------
    fft_freq: torch.Tensor
        The frequency grid of the Fourier transform.
    bfactor: float
        The B-factor to use.

    Returns
    -------
        The critical exposure for the given frequency grid.
    """
    eps = 1e-10
    Ne = 4 / (bfactor * fft_freq.clamp(min=eps) ** 2)
    return Ne


def _critical_exposure_grid(
    image_shape: tuple[int, int] | tuple[int, int, int],
    pixel_size: float,
    crit_exposure_bfactor: int | float = -1,
    rfft: bool = True,
    fftshift: bool = False,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Per-frequency critical exposure Ne(k), unscaled (no Eq. 5 factor of 2)."""
    fft_freq_px = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=device,
    )
    fft_freq_px = fft_freq_px / pixel_size  # Convert to Angstrom^-1

    if crit_exposure_bfactor == -1:
        return critical_exposure(fft_freq=fft_freq_px, a=a, b=b, c=c)
    elif crit_exposure_bfactor >= 0:
        return critical_exposure_bfactor(
            fft_freq=fft_freq_px, bfactor=crit_exposure_bfactor
        )
    raise ValueError("B-factor must be positive or -1.")


def _apply_voltage_correction(dose: torch.Tensor, voltage: float) -> torch.Tensor:
    """Apply voltage-dependent damage correction, consistent with RELION."""
    if abs(voltage - 200) <= 2:
        return dose / 0.8
    elif abs(voltage - 100) <= 2:
        return dose / 0.64
    return dose


def _validate_and_reshape_dose(
    dose: torch.Tensor | float, image_dft: torch.Tensor
) -> torch.Tensor | float:
    """Check a per-batch-element dose tensor matches image_dft's batch shape."""
    if isinstance(dose, torch.Tensor) and dose.numel() > 1 and image_dft.ndim > 2:
        image_batch_shape = image_dft.shape[:-2]
        dose_shape = dose.shape
        if dose_shape != image_batch_shape:
            raise ValueError(
                f"dose tensor shape {dose_shape} "
                f"does not match image batch dimensions {image_batch_shape}"
            )
        dose = einops.rearrange(dose, "... -> ... 1 1")
    return dose


def _dose_weights(dose: torch.Tensor, Ne: torch.Tensor) -> torch.Tensor:
    """Per-frequency, per-frame weight from Grant & Grigorieff (2015) Eq. 5."""
    return torch.exp(-dose / Ne)


def _frame_doses(
    n_frames: int,
    pre_exposure: float,
    dose_per_frame: float,
    start_idx: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Cumulative dose after each frame in [start_idx, start_idx + n_frames)."""
    frame_indices = torch.arange(
        start_idx, start_idx + n_frames, dtype=torch.float32, device=device
    )
    return pre_exposure + dose_per_frame * (frame_indices + 1)


def dose_weight_2d(
    image_dft: torch.Tensor,  # shape (..., h, w)
    image_shape: tuple[int, int],  # shape (h, w)
    pixel_size: float,
    dose: torch.Tensor | float,  # shape (..., ) or float
    voltage: float = 300.0,
    crit_exposure_bfactor: int | float = -1,
    rfft: bool = True,
    fftshift: bool = False,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Apply dose weighting to an image or stack.

    This function implements the dose weighting algorithm following Grant and
    Grigorieff 2015, applying different weights to each frame based on cumulative
    dose and then normalizing across frames.

    Parameters
    ----------
    image_dft : torch.Tensor
        Complex tensor containing imagesin Fourier space with shape
        (..., h, w) for rfft=True or (..., h, w) for full fft.
    image_shape : tuple[int, int]
        The shape of the real space images (h, w).
    pixel_size : float
        The pixel size of the images, in Angstroms.
    dose : torch.Tensor | float, optional
        The dose, in e-/A^2.
    voltage : float, optional
        The acceleration voltage in kV. Affects damage correction for 100kV and 200kV.
        Default is 300.0.
    crit_exposure_bfactor : int | float, optional
        The B factor for dose weighting based on critical exposure. If -1,
        then use Grant and Grigorieff (2015) values. Default is -1.
    rfft : bool, optional
        Whether the input DFT is from a real FFT. Default is True.
    fftshift : bool, optional
        Whether the input DFT is fftshifted. Default is False.
    a : float, optional
        The a parameter for the critical exposure formula. Default is 0.245.
    b : float, optional
        The b parameter for the critical exposure formula. Default is -1.665.
    c : float, optional
        The c parameter for the critical exposure formula. Default is 2.81.
    device : torch.device | None, optional
        The device to use for the calculation. If None, infers device from movie_dft.
        Default is None.

    Returns
    -------
    torch.Tensor
        The dose-weighted movie frames with the same shape as input.
    """
    dose = _validate_and_reshape_dose(dose, image_dft)

    # Determine device
    if device is None:
        device = image_dft.device

    # Move movie_dft to specified device if needed
    image_dft = image_dft.to(device)

    dose = _apply_voltage_correction(dose, voltage)

    eps = 1e-10
    Ne = (
        _critical_exposure_grid(
            image_shape=image_shape,
            pixel_size=pixel_size,
            crit_exposure_bfactor=crit_exposure_bfactor,
            rfft=rfft,
            fftshift=fftshift,
            a=a,
            b=b,
            c=c,
            device=device,
        )
        * 2  # Apply factor of 2 from Eq. 5 (factoring out 0.5)
    ).clamp(min=eps)

    # Calculate weights for each frame at each frequency
    # Reshape for broadcasting: dose (..., 1, 1) and Ne (1, h, w)
    # Expand Ne to match the number of leading dimensions in dose/image_dft
    n_leading = len(image_dft.shape) - 2  # number of ... dims
    Ne_expanded = einops.rearrange(Ne, "h w -> " + " ".join(["1"] * n_leading) + " h w")
    weights = _dose_weights(dose, Ne_expanded)  # Shape: (..., h, w)

    # Apply weights to each image
    weighted_images = image_dft * weights

    # Calculate sum of squared weights for normalization (Eq. 9)
    sum_weight_sq = einops.reduce(weights**2, "... h w -> h w", "sum")
    sum_weight_sq = torch.sqrt(sum_weight_sq.clamp(min=eps))

    # Normalize all frames by the sum of squared weights
    sum_weight_sq_expanded = einops.rearrange(
        sum_weight_sq, "h w -> " + " ".join(["1"] * n_leading) + " h w"
    )
    normalized_frames = weighted_images / sum_weight_sq_expanded

    return normalized_frames


def dose_weight_movie_memory_efficient_2d(
    image_dft: torch.Tensor,
    image_shape: tuple[int, int],
    pixel_size: float,
    dose: torch.Tensor,
    voltage: float = 300.0,
    crit_exposure_bfactor: int | float = -1,
    rfft: bool = True,
    fftshift: bool = False,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    device: torch.device | None = None,
    chunk_size: int = 100,
) -> torch.Tensor:
    """
    Memory-efficient dose weighting for movies using chunked processing.

    This function processes movies in chunks to avoid memory issues by calling
    dose_weight_2d on chunks and handling normalization separately.

    Parameters
    ----------
    image_dft : torch.Tensor
        Complex tensor containing movie frames in Fourier space with shape
        (n_frames, h, w) for rfft=True or (n_frames, h, w) for full fft.
    image_shape : tuple[int, int]
        The shape of the real space images (h, w).
    pixel_size : float
        The pixel size of the images, in Angstroms.
    dose : torch.Tensor
        Pre-calculated doses for each frame, in e-/A^2.
    voltage : float, optional
        The acceleration voltage in kV. Affects damage correction for 100kV and 200kV.
        Default is 300.0.
    crit_exposure_bfactor : int | float, optional
        The B factor for dose weighting based on critical exposure. If -1,
        then use Grant and Grigorieff (2015) values. Default is -1.
    rfft : bool, optional
        Whether the input DFT is from a real FFT. Default is True.
    fftshift : bool, optional
        Whether the input DFT is fftshifted. Default is False.
    a : float, optional
        The a parameter for the critical exposure formula. Default is 0.245.
    b : float, optional
        The b parameter for the critical exposure formula. Default is -1.665.
    c : float, optional
        The c parameter for the critical exposure formula. Default is 2.81.
    device : torch.device | None, optional
        The device to use for the calculation. If None, infers device from movie_dft.
        Default is None.
    chunk_size : int, optional
        The number of frames to process in each chunk. Default is 100.

    Returns
    -------
    torch.Tensor
        The dose-weighted movie frames with the same shape as input.
    """
    dose = _validate_and_reshape_dose(dose, image_dft)

    # Determine device
    if device is None:
        device = image_dft.device

    # Move image_dft to specified device if needed
    image_dft = image_dft.to(device)

    dose = _apply_voltage_correction(dose, voltage)

    n_frames = image_dft.shape[0]

    eps = 1e-10
    Ne = (
        _critical_exposure_grid(
            image_shape=image_shape,
            pixel_size=pixel_size,
            crit_exposure_bfactor=crit_exposure_bfactor,
            rfft=rfft,
            fftshift=fftshift,
            a=a,
            b=b,
            c=c,
            device=device,
        )
        * 2  # Apply factor of 2 from Eq. 5 (factoring out 0.5)
    ).clamp(min=eps)
    Ne_expanded = einops.rearrange(Ne, "h w -> 1 h w")

    # FIRST PASS: Compute normalization factors by accumulating weight sums
    sum_weight_sq = torch.zeros_like(Ne)

    for start_idx in range(0, n_frames, chunk_size):
        end_idx = min(start_idx + chunk_size, n_frames)
        chunk_doses = dose[start_idx:end_idx]

        # Calculate weights for this chunk
        weights = _dose_weights(chunk_doses, Ne_expanded)  # (chunk_size, h, w)

        # Accumulate sum of squared weights
        sum_weight_sq += einops.reduce(weights**2, "... h w -> h w", "sum")

        # Clear chunk from memory
        del weights, chunk_doses

    # Compute normalization factor
    sum_weight_sq = torch.sqrt(sum_weight_sq.clamp(min=eps))
    sum_weight_sq_expanded = einops.rearrange(sum_weight_sq, "h w -> 1 h w")

    # SECOND PASS: Apply weighting and normalization in chunks
    result = torch.zeros_like(image_dft)

    for start_idx in range(0, n_frames, chunk_size):
        end_idx = min(start_idx + chunk_size, n_frames)
        chunk_doses = dose[start_idx:end_idx]
        chunk_movie = image_dft[start_idx:end_idx]

        # Calculate weights for this chunk
        weights = _dose_weights(chunk_doses, Ne_expanded)  # (chunk_size, h, w)

        # Apply weights to chunk and normalize by sum of squared weights
        weighted_chunk = chunk_movie * weights
        normalized_chunk = weighted_chunk / sum_weight_sq_expanded

        # Store result
        result[start_idx:end_idx] = normalized_chunk

    return result


def dose_weight_normalization_grid(
    image_shape: tuple[int, int],
    pixel_size: float,
    n_frames: int,
    pre_exposure: float = 0.0,
    dose_per_frame: float = 1.0,
    voltage: float = 300.0,
    crit_exposure_bfactor: int | float = -1,
    rfft: bool = True,
    fftshift: bool = False,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    device: torch.device | None = None,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the per-frequency critical exposure and normalization grids.

    Parameters
    ----------
    image_shape : tuple[int, int]
        The shape of the real space images (h, w).
    pixel_size : float
        The pixel size of the images, in Angstroms.
    n_frames : int
        The total number of frames in the movie.
    pre_exposure : float, optional
        The pre-exposure before the first frame, in e-/A^2. Default is 0.0.
    dose_per_frame : float, optional
        The dose per frame, in e-/A^2. Default is 1.0.
    voltage : float, optional
        The acceleration voltage in kV. Affects damage correction for 100kV and 200kV.
        Default is 300.0.
    crit_exposure_bfactor : int | float, optional
        The B factor for dose weighting based on critical exposure. If -1,
        then use Grant and Grigorieff (2015) values. Default is -1.
    rfft : bool, optional
        Whether the grid should match a real FFT. Default is True.
    fftshift : bool, optional
        Whether the grid should be fftshifted. Default is False.
    a : float, optional
        The a parameter for the critical exposure formula. Default is 0.245.
    b : float, optional
        The b parameter for the critical exposure formula. Default is -1.665.
    c : float, optional
        The c parameter for the critical exposure formula. Default is 2.81.
    device : torch.device | None, optional
        The device to compute the grids on. Default is None.
    chunk_size : int, optional
        Number of frames' doses to accumulate at a time while summing
        squared weights across all `n_frames`. Default is 16.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        `(Ne, normalization)`, each shaped like the frequency grid for
        `image_shape` (i.e. `(h, w // 2 + 1)` for `rfft=True`). `Ne` is the
        critical exposure grid; `normalization` is the sqrt of the summed
        squared weights across all `n_frames`, to be divided into each
        frame's weight.
    """
    doses = _frame_doses(n_frames, pre_exposure, dose_per_frame, device=device)
    doses = _apply_voltage_correction(doses, voltage)

    eps = 1e-10
    Ne = (
        _critical_exposure_grid(
            image_shape=image_shape,
            pixel_size=pixel_size,
            crit_exposure_bfactor=crit_exposure_bfactor,
            rfft=rfft,
            fftshift=fftshift,
            a=a,
            b=b,
            c=c,
            device=device,
        )
        * 2  # Apply factor of 2 from Eq. 5 (factoring out 0.5)
    ).clamp(min=eps)

    sum_weight_sq = torch.zeros_like(Ne)
    for start_idx in range(0, n_frames, chunk_size):
        end_idx = min(start_idx + chunk_size, n_frames)
        chunk_doses = doses[start_idx:end_idx]
        weights = _dose_weights(chunk_doses[:, None, None], Ne[None])
        sum_weight_sq += (weights**2).sum(dim=0)

    normalization = torch.sqrt(sum_weight_sq.clamp(min=eps))
    return Ne, normalization


def dose_weight_frame_chunk(
    chunk_dft: torch.Tensor,
    frame_start_idx: int,
    Ne: torch.Tensor,
    normalization: torch.Tensor,
    pre_exposure: float = 0.0,
    dose_per_frame: float = 1.0,
    voltage: float = 300.0,
    in_place: bool = False,
) -> torch.Tensor:
    """Dose weight one contiguous chunk of a movie's frames' Fourier transforms."""
    n_chunk = chunk_dft.shape[0]
    device = chunk_dft.device
    doses = _frame_doses(
        n_chunk, pre_exposure, dose_per_frame, start_idx=frame_start_idx, device=device
    )
    doses = _apply_voltage_correction(doses, voltage)

    weights = _dose_weights(doses[:, None, None], Ne[None]) / normalization[None]
    if in_place:
        chunk_dft *= weights
        return chunk_dft
    return chunk_dft * weights


def dose_weight_movie(
    movie_dft: torch.Tensor,
    image_shape: tuple[int, int],
    pixel_size: float,
    pre_exposure: float = 0.0,
    dose_per_frame: float = 1.0,
    voltage: float = 300.0,
    crit_exposure_bfactor: int | float = -1,
    rfft: bool = True,
    fftshift: bool = False,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    device: torch.device | None = None,
    memory_efficient: bool = False,
    chunk_size: int = 10,
) -> torch.Tensor:
    """
    Apply per-frame dose weighting to a movie in Fourier space.

    This function implements the dose weighting algorithm following Grant and
    Grigorieff 2015, applying different weights to each frame based on cumulative
    dose and then normalizing across frames.

    Parameters
    ----------
    movie_dft : torch.Tensor
        Complex tensor containing movie frames in Fourier space with shape
        (n_frames, h, w) for rfft=True or (n_frames, h, w) for full fft.
    image_shape : tuple[int, int]
        The shape of the real space images (h, w).
    pixel_size : float
        The pixel size of the images, in Angstroms.
    pre_exposure : float, optional
        The pre-exposure before the first frame, in e-/A^2. Default is 0.0.
    dose_per_frame : float, optional
        The dose per frame, in e-/A^2. Default is 1.0.
    voltage : float, optional
        The acceleration voltage in kV. Affects damage correction for 100kV and 200kV.
        Default is 300.0.
    crit_exposure_bfactor : int | float, optional
        The B factor for dose weighting based on critical exposure. If -1,
        then use Grant and Grigorieff (2015) values. Default is -1.
    rfft : bool, optional
        Whether the input DFT is from a real FFT. Default is True.
    fftshift : bool, optional
        Whether the input DFT is fftshifted. Default is False.
    a : float, optional
        The a parameter for the critical exposure formula. Default is 0.245.
    b : float, optional
        The b parameter for the critical exposure formula. Default is -1.665.
    c : float, optional
        The c parameter for the critical exposure formula. Default is 2.81.
    device : torch.device | None, optional
        The device to use for the calculation. If None, infers device from movie_dft.
        Default is None.
    memory_efficient : bool, optional
        Whether to use memory-efficient chunked processing. Default is False.
    chunk_size : int, optional
        The number of frames to process in each chunk for memory-efficient mode.
        Default is 100.

    Returns
    -------
    torch.Tensor
        The dose-weighted movie frames with the same shape as input.
    """
    if movie_dft.ndim != 3:
        raise ValueError(
            f"movie_dft must be 3D tensor with shape (n_frames, h, w),"
            f" got {movie_dft.shape}"
        )

    # Determine device
    if device is None:
        device = movie_dft.device

    # Move movie_dft to specified device if needed
    movie_dft = movie_dft.to(device)
    # Calculate doses for each frame (dose AFTER each frame)
    doses = _frame_doses(
        movie_dft.shape[0], pre_exposure, dose_per_frame, device=device
    )

    if memory_efficient:
        # Use memory-efficient chunked processing
        return dose_weight_movie_memory_efficient_2d(
            image_dft=movie_dft,
            image_shape=image_shape,
            pixel_size=pixel_size,
            dose=doses,
            voltage=voltage,
            crit_exposure_bfactor=crit_exposure_bfactor,
            rfft=rfft,
            fftshift=fftshift,
            a=a,
            b=b,
            c=c,
            device=device,
            chunk_size=chunk_size,
        )
    else:
        # Use original method
        return dose_weight_2d(
            image_dft=movie_dft,
            image_shape=image_shape,
            pixel_size=pixel_size,
            dose=doses,
            voltage=voltage,
            crit_exposure_bfactor=crit_exposure_bfactor,
            rfft=rfft,
            fftshift=fftshift,
            a=a,
            b=b,
            c=c,
            device=device,
        )


def cumulative_dose_filter_3d(
    volume_shape: tuple[int, int, int] | tuple[int, int],
    pixel_size: float = 1,
    start_exposure: float = 0.0,
    end_exposure: float = 30.0,
    crit_exposure_bfactor: int | float = -1,
    rfft: bool = True,
    fftshift: bool = False,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Dose weight a 3D volume using Grant and Grigorieff 2015.

    Use integration to speed up.

    Parameters
    ----------
    volume_shape : tuple[int, int, int]
        The shape of the filter to calculate (real space). Rfft is
        automatically calculated from this.
    pixel_size : float
        The pixel size of the volume, in Angstrom.
    start_exposure : float
        The start exposure for dose weighting, in e-/A^2. Default is 0.0.
    end_exposure : float
        The end exposure for dose weighting, in e-/A^2. Default is 30.0.
    crit_exposure_bfactor : int | float
        The B factor for dose weighting based on critical exposure. If '-1',
        then use Grant and Grigorieff (2015) values.
    rfft : bool
        If the FFT is a real FFT.
    fftshift : bool
        If the FFT is shifted.
    a: float
        The a parameter for the critical exposure formula. Default is 0.245.
    b: float
        The b parameter for the critical exposure formula. Default is -1.665.
    c: float
        The c parameter for the critical exposure formula. Default is 2.81.
    device : torch.device
        The device to use for the calculation.

    Returns
    -------
    torch.Tensor
        The dose weighting filter.
    """
    eps = 1e-10
    Ne = _critical_exposure_grid(
        image_shape=volume_shape,
        pixel_size=pixel_size,
        crit_exposure_bfactor=crit_exposure_bfactor,
        rfft=rfft,
        fftshift=fftshift,
        a=a,
        b=b,
        c=c,
        device=device,
    ).clamp(min=eps)

    return (
        2
        * Ne
        * (
            torch.exp((-0.5 * start_exposure) / Ne)
            - torch.exp((-0.5 * end_exposure) / Ne)
        )
        / end_exposure
    )
