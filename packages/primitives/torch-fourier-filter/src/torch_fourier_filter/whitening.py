"""Noise whitening functions."""

from typing import Optional

import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from torch_fourier_filter.utils import (
    _handle_dim,
    bin_1dim_with_lerp,
    curve_1dim_to_ndim,
)


def calculate_num_freq_bins(image_shape: tuple[int, ...]) -> int:
    """Calculate the number of frequency bins for a power spectrum given image shape.

    Parameters
    ----------
    image_shape: tuple[int, ...]
        Shape of the image.

    Returns
    -------
    int
        Number of frequency bins.
    """
    largest_dim = max(image_shape)

    return int((largest_dim // 2 + 1) * len(image_shape) ** 0.5 + 1)


def real_space_shape_from_dft_shape(
    dft_shape: tuple[int, ...], rfft: bool
) -> tuple[int, ...]:
    """Calculate the real-space shape from a DFT shape and rfft argument.

    Parameters
    ----------
    dft_shape: tuple[int, ...]
        Shape of the DFT.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False). The default is
        True.

    Returns
    -------
    tuple[int, ...]
        Shape of the real-space image.
    """
    real_space_shape = list(dft_shape)
    if rfft:
        last_dim = 2 * (real_space_shape[-1] - 1)
        real_space_shape[-1] = last_dim

    return tuple(real_space_shape)


def power_spectral_density(
    image_dft: torch.Tensor,
    rfft: bool = True,
    fftshift: bool = False,
    dim: int | tuple[int, ...] = (-2, -1),
    num_freq_bins: Optional[int] = None,
    max_freq: Optional[float] = None,
    do_power_spectrum: Optional[bool] = True,
) -> torch.Tensor:
    """Calculates the power spectral density of an image in 1-dimension.

    Note that the power spectrum can either be calculated over the intensities or
    amplitudes of the Fourier components; the parameter `do_power_spectrum` controls
    this.

    Parameters
    ----------
    image_dft: torch.Tensor
        The DFT of the images/volumes.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
    fftshift: bool
        Whether the input is fftshifted.
    dim: int | tuple[int, ...], optional
        Which dimensions to calculate the power spectrum over. The default is (-2, -1)
        which corresponds to the spatial dimensions of an image.
    num_freq_bins: int, optional
        Number of frequency bins to calculate the power spectrum over. If None, the
        the number is inferred from the input shape automatically.
    max_freq: float, optional
        Optional maximum frequency to calculate the power spectrum to, in terms of the
        Nyquist frequency. The default is None and corresponds to going past the Nyquist
        frequency into the corner of Fourier space. For example, a 2D input will have
        frequencies up to sqrt(2)/2. Frequencies above this are set to zero.
    do_power_spectrum: bool, optional
        Weather to use the power (intensity) spectrum or the amplitude spectrum. If
        True, the power spectrum is calculated, otherwise the amplitude spectrum is
        calculated. The default is True.

    Returns
    -------
    torch.Tensor
        The power spectral density of the input image. Includes batch dimensions, if
        present.

    Raises
    ------
    ValueError
        Currently *does not* support batching. Raises error if the input image is larger
        than the number of dimensions.
    """
    dim = _handle_dim(dim, image_dft.ndim)
    shape_over_dim = tuple(image_dft.shape[d] for d in dim)

    real_space_shape = real_space_shape_from_dft_shape(
        dft_shape=shape_over_dim, rfft=rfft
    )

    # Construct 1-dimensional grid of frequencies for binning
    if num_freq_bins is None:
        num_freq_bins = calculate_num_freq_bins(shape_over_dim)
    binning_freqs = torch.linspace(
        start=0.0,
        end=(len(dim) ** 0.5) / 2,  # corner of Fourier space
        steps=num_freq_bins,
        device=image_dft.device,
    )

    # Calculate the amplitude (or intensity) of all Fourier components
    power_spectrum = torch.abs(image_dft)
    if do_power_spectrum:
        power_spectrum = power_spectrum**2

    # Construct FFT frequency grid
    freq_grid = fftfreq_grid(
        image_shape=real_space_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=image_dft.device,
    )

    # Bin the power spectrum into 1-dimension
    power_spectrum_1d = bin_1dim_with_lerp(
        x=freq_grid,
        y=power_spectrum,
        xp=binning_freqs,
        normalize_by_count=True,
    )

    if do_power_spectrum:
        power_spectrum_1d = torch.sqrt(power_spectrum_1d)

    # Set frequencies above max_freq to zero, if specified
    if max_freq is not None:
        power_spectrum_1d = torch.where(
            binning_freqs <= max_freq,
            power_spectrum_1d,
            torch.zeros_like(power_spectrum_1d),
        )

    return power_spectrum_1d


def whitening_filter(
    image_dft: torch.Tensor,
    rfft: bool = True,
    fftshift: bool = False,
    dim: int | tuple[int, ...] = (-2, -1),
    num_freq_bins: Optional[int] = None,
    max_freq: Optional[float] = None,
    do_power_spectrum: bool = True,
    output_shape: Optional[tuple[int, ...]] = None,
    output_rfft: Optional[bool] = None,
    output_fftshift: Optional[bool] = None,
) -> torch.Tensor:
    """Create a whitening filter the discrete Fourier transform of an input.

    Parameters
    ----------
    image_dft: torch.Tensor
        The DFT of the images/volumes.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
    fftshift: bool
        Whether the input is fftshifted.
    dim: int | tuple[int, ...], optional
        Which dimensions to calculate the power spectrum over. The default is (-2, -1)
        which corresponds to the spatial dimensions of an image.
    num_freq_bins: int, optional
        Number of frequency bins to calculate the power spectrum over. If None, the
        the number is inferred from the input shape automatically.
    max_freq: float, optional
        Maximum frequency to calculate the whitening filter to, in terms of the
        Nyquist frequency. The default is None and corresponds to going past the Nyquist
        frequency into the corner of Fourier space. For example, a 2D input will have
        frequencies up to sqrt(2)/2. Frequencies above this are set to one in the
        returned whitening filter.
    do_power_spectrum: bool, optional
        Weather to use the power (intensity) spectrum or the amplitude spectrum. If
        True, the power spectrum is calculated, otherwise the amplitude spectrum is
        calculated. The default is True.
    output_shape: tuple[int, ...], optional
        The shape of the output filter. If None, then the shape is the same as
        `image_dft`. The default is None. NOTE: This shape is in real space.
    output_rfft: bool, optional
        Whether to return the filter in rfft form. If None, then the same as `rfft`.
        The default is None.
    output_fftshift: bool, optional
        Whether to return the filter fftshifted. If None, then the same as `fftshift`.
        The default is None.

    Returns
    -------
    torch.Tensor
        The whitening filter in Fourier space.
    """
    device = image_dft.device
    dim = _handle_dim(dim, image_dft.ndim)

    power_spec_1d = power_spectral_density(
        image_dft=image_dft,
        rfft=rfft,
        fftshift=fftshift,
        dim=dim,
        num_freq_bins=num_freq_bins,
        do_power_spectrum=do_power_spectrum,
    )

    # Create the filter from the power spectrum
    psd_zeros_mask = power_spec_1d == 0
    whitening_filter_1d = torch.where(
        psd_zeros_mask, torch.zeros_like(power_spec_1d), 1 / power_spec_1d
    )

    # Calculate the last frequency value
    last_freq = (len(dim) ** 0.5) / 2

    # Generate a 1D frequency grid
    frequency_1d = torch.linspace(
        0, last_freq, steps=len(whitening_filter_1d), device=device
    )

    # Create a mask for frequencies above the max frequency, if specified
    if max_freq is not None:
        above_freq_mask = frequency_1d > max_freq
        whitening_filter_1d[above_freq_mask] = 1.0
    else:
        above_freq_mask = torch.zeros_like(whitening_filter_1d, dtype=torch.bool)

    # Normalize the whitening filter by the maximum value of the valid frequencies
    max_valid_value = whitening_filter_1d[~above_freq_mask].max()
    whitening_filter_1d /= max_valid_value

    # Set the values above the max frequency to 1
    whitening_filter_1d = torch.where(
        above_freq_mask, torch.ones_like(whitening_filter_1d), whitening_filter_1d
    )

    # Construct FFT frequency grid
    if output_shape is None:
        output_shape = real_space_shape_from_dft_shape(image_dft.shape, rfft=rfft)

    if output_rfft is None:
        output_rfft = rfft

    if output_fftshift is None:
        output_fftshift = fftshift

    # real_space_shape = real_space_shape_from_dft_shape(
    #     dft_shape=output_shape, rfft=output_rfft
    # )

    freq_grid = fftfreq_grid(
        image_shape=output_shape,
        rfft=output_rfft,
        fftshift=output_fftshift,
        norm=True,
        device=image_dft.device,
    )

    # Interpolate the filter to the frequency grid
    whitening_filter_ndim = curve_1dim_to_ndim(
        frequency_1d=frequency_1d,
        values_1d=whitening_filter_1d,
        frequency_grid=freq_grid,
        fill_lower=1.0,  # Fill oob areas with ones (no scaling)
        fill_upper=1.0,
    )

    return whitening_filter_ndim


# def gaussian_smoothing(
#     tensor: torch.Tensor,
#     dim: int | tuple[int, ...] = -1,
#     kernel_size: int = 5,
#     sigma: float = 1.0,
# ) -> torch.Tensor:
#     """
#     Apply Gaussian smoothing over specified dimensions of a tensor.

#     Parameters
#     ----------
#     tensor: torch.Tensor
#         The input tensor to be smoothed.
#     dim: int | tuple[int, ...]
#         Dimensions over which to apply smoothing. Can be a single int or tuple of
#         ints. Negative dimensions are indexed from the end.
#     kernel_size: int
#         The size of the Gaussian kernel.
#     sigma: float
#         The standard deviation of the Gaussian kernel.

#     Returns
#     -------
#     torch.Tensor
#         The smoothed tensor with same shape as input.
#     """
#     # Convert single dim to tuple
#     if isinstance(dim, int):
#         dim = (dim,)

#     # Convert negative dims to positive
#     dim = tuple(d if d >= 0 else tensor.dim() + d for d in dim)

#     # Validate dimensions
#     if not all(0 <= d < tensor.dim() for d in dim):
#         raise ValueError(
#             f"Invalid dimensions {dim} for tensor of rank {tensor.dim()}"
#          )
#     if len(dim) > 2:
#         raise ValueError("Gaussian smoothing only supports 1D or 2D operations")

#     # Create coordinate grid for kernel
#     x = torch.arange(
#         -kernel_size // 2 + 1,
#         kernel_size // 2 + 1,
#         dtype=tensor.dtype,
#         device=tensor.device,
#     )

#     if len(dim) == 1:  # 1D smoothing
#         kernel = torch.exp(-0.5 * (x / sigma) ** 2)
#         kernel = kernel / kernel.sum()

#         # Create conv kernel with singleton dimensions
#         shape = [1] * (tensor.dim())
#         shape[dim[0]] = kernel_size
#         kernel = kernel.view(*shape)

#         # Apply 1D convolution
#         tensor = einops.rearrange(tensor, "... n -> ... 1 1 n")
#         kernel = einops.rearrange(kernel, "... n -> ... 1 1 n")
#         smoothed_tensor = F.conv1d(tensor, kernel, padding=kernel_size // 2)
#         return einops.rearrange(smoothed_tensor, "... 1 1 n -> ... n")

#     else:  # 2D smoothing
#         x, y = torch.meshgrid(x, x, indexing="ij")
#         kernel = torch.exp(-0.5 * ((x / sigma) ** 2 + (y / sigma) ** 2))
#         kernel = kernel / kernel.sum()

#         # Create conv kernel with singleton dimensions
#         shape = [1] * (tensor.dim())
#         shape[dim[0]] = kernel_size
#         shape[dim[1]] = kernel_size
#         kernel = kernel.view(*shape)
#         kernel = einops.rearrange(kernel, "... h w -> ... 1 1 h w")
#         # Apply 2D convolution
#         tensor = einops.rearrange(tensor, "... h w -> ... 1 1 h w")
#         smoothed_tensor = F.conv2d(
#             tensor, kernel, padding=kernel_size // 2, stride=(1, 1)
#         )
#         return einops.rearrange(smoothed_tensor, "... 1 1 h w -> ... h w")
