"""Functions to construct grids of DFT sample frequencies."""

from typing import Sequence

import einops
import torch

from torch_grid_utils.fft_shape import rfft_shape


# no lru_cache as it interferes with gradient calculation (see torch-fourier-shift PR#5)
def fftfreq_grid(
    image_shape: tuple[int] | tuple[int, int] | tuple[int, int, int],
    rfft: bool,
    fftshift: bool = False,
    spacing: float
    | tuple[float]
    | tuple[float, float]
    | tuple[float, float, float] = 1,
    norm: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct a 1D, 2D or 3D grid of DFT sample frequencies.

    For a 2D image with shape `(h, w)` and `rfft=False` this function will produce
    a `(h, w, 2)` array of DFT sample frequencies in the `h` and `w` dimensions.
    If `norm` is True the Euclidean norm will be calculated over the last dimension
    leaving a `(h, w)` grid.

    Parameters
    ----------
    image_shape: tuple[int] | tuple[int, int] | tuple[int, int, int]
        Shape of the 1D, 2D or 3D image before computing the DFT.
    rfft: bool
        Whether the output should contain frequencies for a real-valued DFT.
    fftshift: bool
        Whether to fftshift the output grid.
    spacing: float | tuple[float] | tuple[float, float] | tuple[float, float, float]
        Spacing between samples in each dimension. Sampling is considered to be
        isotropic if a single value is passed.
    norm: bool
        Whether to compute the Euclidean norm over the last dimension.
    device: torch.device | None
        PyTorch device on which the returned grid will be stored.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(*image_shape, ndim)` array of DFT sample frequencies in each
        image dimension if `norm` is `False` else `(*image_shape, )`.
    """
    if len(image_shape) == 1:
        frequency_grid = _construct_fftfreq_grid_1d(
            image_shape=image_shape,  # type: ignore[arg-type]
            rfft=rfft,
            spacing=spacing,  # type: ignore[arg-type]
            device=device,
        )
        if fftshift is True:
            frequency_grid = einops.rearrange(frequency_grid, "... freq -> freq ...")
            frequency_grid = fftshift_1d(frequency_grid, rfft=rfft)
            frequency_grid = einops.rearrange(frequency_grid, "freq ... -> ... freq")
    elif len(image_shape) == 2:
        frequency_grid = _construct_fftfreq_grid_2d(
            image_shape=image_shape,
            rfft=rfft,
            spacing=spacing,  # type: ignore[arg-type]
            device=device,
        )
        if fftshift is True:
            frequency_grid = einops.rearrange(frequency_grid, "... freq -> freq ...")
            frequency_grid = fftshift_2d(frequency_grid, rfft=rfft)
            frequency_grid = einops.rearrange(frequency_grid, "freq ... -> ... freq")
    elif len(image_shape) == 3:
        frequency_grid = _construct_fftfreq_grid_3d(
            image_shape=image_shape,
            rfft=rfft,
            spacing=spacing,  # type: ignore[arg-type]
            device=device,
        )
        if fftshift is True:
            frequency_grid = einops.rearrange(frequency_grid, "... freq -> freq ...")
            frequency_grid = fftshift_3d(frequency_grid, rfft=rfft)
            frequency_grid = einops.rearrange(frequency_grid, "freq ... -> ... freq")
    else:
        raise NotImplementedError(
            "Construction of fftfreq grids is currently only supported for "
            "1D, 2D and 3D images."
        )
    if norm is True:
        frequency_grid = (
            einops.reduce(
                frequency_grid**2, "... squared_freqs -> ...", reduction="sum"
            )
            ** 0.5
        )
    return frequency_grid


def transform_fftfreq_grid(
    frequency_grid: torch.Tensor,
    real_space_matrix: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Apply transformation matrix to an N-D spatial frequency grid.

    Parameters
    ----------
    frequency_grid : torch.Tensor
        Frequency grid with shape (..., D), where D is 2 or 3.
    real_space_matrix : torch.Tensor
        Real-space matrix with shape (..., D, D).
        Can be reprensatiative of an anisotropic magnification.
        Must be broadcastable to frequency_grid.
        Real space transforms as x -> A x.
    device: torch.device | None
        PyTorch device for the transformation matrix.

    Returns
    -------
    transformed_grid : torch.Tensor
        Frequency grid with transformation applied.
        Same shape and layout as input.
    """
    # Move tensors to a common device
    if device is not None:
        frequency_grid = frequency_grid.to(device)
        real_space_matrix = real_space_matrix.to(device)
    else:
        real_space_matrix = real_space_matrix.to(frequency_grid.device)

    ndim = frequency_grid.shape[-1]

    if real_space_matrix.shape[-2:] != (ndim, ndim):
        raise ValueError(
            f"real_space_matrix must have shape (..., {ndim}, {ndim}), "
            f"got {real_space_matrix.shape}"
        )

    real_space_mat = real_space_matrix.to(
        dtype=frequency_grid.dtype,
        device=frequency_grid.device,
    )

    # Fourier-space transform matrix: (A^-1)^T
    fourier_space_mat = torch.linalg.inv(real_space_mat).transpose(-2, -1)

    # Apply transform:
    # [k_new]^T = fourier_space_mat @ [k]^T
    frequency_grid = einops.rearrange(frequency_grid, "... d -> ... d 1")
    frequency_grid = fourier_space_mat @ frequency_grid
    transformed_grid = einops.rearrange(frequency_grid, "... d 1 -> ... d")

    return transformed_grid


def _construct_fftfreq_grid_1d(
    image_shape: tuple[int],
    rfft: bool,
    spacing: float | tuple[float] = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct a `(w, 1)` grid of DFT sample freqs for a 1D signal `(w, )`."""
    (dw,) = spacing if isinstance(spacing, Sequence) else [spacing]
    frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    (w,) = image_shape
    return frequency_func(w, d=dw, device=device).unsqueeze(-1)


def _construct_fftfreq_grid_2d(
    image_shape: tuple[int, int],
    rfft: bool,
    spacing: float | tuple[float, float] = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """Construct a grid of DFT sample freqs for a 2D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 2D shape `(h, w)` of the input image for which a grid of DFT sample freqs
        should be calculated.
    rfft: bool
        Whether the frequency grid is for a real fft (rfft).
    spacing: float | tuple[float, float]
        Sample spacing in `h` and `w` dimensions of the grid.
    device: torch.device
        Torch device for the resulting grid.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 2)` array of DFT sample freqs.
        Order of freqs in the last dimension corresponds to the order of
        the two dimensions of the grid.
    """
    dh, dw = spacing if isinstance(spacing, Sequence) else [spacing] * 2
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    h, w = image_shape
    freq_y = torch.fft.fftfreq(h, d=dh, device=device)
    freq_x = last_axis_frequency_func(w, d=dw, device=device)
    h, w = rfft_shape(image_shape) if rfft is True else image_shape
    freq_yy = einops.repeat(freq_y, "h -> h w", w=w)
    freq_xx = einops.repeat(freq_x, "w -> h w", h=h)
    return einops.rearrange([freq_yy, freq_xx], "freq h w -> h w freq")


def _construct_fftfreq_grid_3d(
    image_shape: Sequence[int],
    rfft: bool,
    spacing: float | tuple[float, float, float] = 1,
    device: torch.device = None,
) -> torch.Tensor:
    """Construct a grid of DFT sample freqs for a 3D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 3D shape `(d, h, w)` of the input image for which a grid of DFT sample freqs
        should be calculated.
    rfft: bool
        Controls Whether the frequency grid is for a real fft (rfft).
    spacing: float | tuple[float, float, float]
        Sample spacing in `d`, `h` and `w` dimensions of the grid.
    device: torch.device
        Torch device for the resulting grid.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 3)` array of DFT sample freqs.
        Order of freqs in the last dimension corresponds to the order of dimensions
        of the grid.
    """
    dd, dh, dw = spacing if isinstance(spacing, Sequence) else [spacing] * 3
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    d, h, w = image_shape
    freq_z = torch.fft.fftfreq(d, d=dd, device=device)
    freq_y = torch.fft.fftfreq(h, d=dh, device=device)
    freq_x = last_axis_frequency_func(w, d=dw, device=device)
    d, h, w = rfft_shape(image_shape) if rfft is True else image_shape
    freq_zz = einops.repeat(freq_z, "d -> d h w", h=h, w=w)
    freq_yy = einops.repeat(freq_y, "h -> d h w", d=d, w=w)
    freq_xx = einops.repeat(freq_x, "w -> d h w", d=d, h=h)
    return einops.rearrange([freq_zz, freq_yy, freq_xx], "freq ... -> ... freq")


def fftfreq_shell_centers(
    sidelength: int, device: torch.device | None = None
) -> torch.Tensor:
    """Centers of the ``sidelength // 2 + 1`` frequency shells of a DFT.

    Notes
    -----
    Shells are spaced ``1 / sidelength`` apart, from DC to the highest frequency sampled
    along a dimension of length ``sidelength``. That is ``(n // 2) / n`` for both rfft
    and full DFTs: Nyquist (0.5) for even ``n`` and ``0.5 - 1 / (2n)`` for odd ``n``.
    The values match ``torch.fft.rfftfreq(sidelength)``.

    Parameters
    ----------
    sidelength: int
        Real-space length ``n`` of the dimension setting the shell spacing, usually
        the smallest dimension of the image.
    device: torch.device | None
        PyTorch device on which the returned tensor will be stored.

    Returns
    -------
    shell_centers: torch.Tensor
        ``(sidelength // 2 + 1, )`` array of shell centers in cycles per pixel.
    """
    n_shells = sidelength // 2 + 1
    return torch.linspace(
        0, (sidelength // 2) / sidelength, steps=n_shells, device=device
    )


def fftfreq_to_dft_coordinates(
    frequencies: torch.Tensor,
    image_shape: Sequence[int],
    rfft: bool,
    fftshift: bool = False,
) -> torch.Tensor:
    """Convert DFT sample frequencies into continuous array coordinates of a DFT.

    Adjacent DFT samples are `1 / n` apart along a dimension of length `n`, so
    frequency `f` (cycles per pixel) is `f * n` samples away from DC:

    - `fftshift=True`: DC sits at index `n // 2` of each full dimension (`n / 2` for
       even `n`, `(n - 1) / 2` for odd `n`), so coordinates are
      `f * n + n // 2`, in `[0, n - 1]`.
    - `fftshift=False`: DC sits at index 0 and negative frequencies of each full
      dimension wrap to the end of the array, so coordinates are `(f * n) mod n`, in
      `[0, n)`. Values in `(n - 1, n)` lie between the last sample and sample 0, so
      interpolation there must wrap around.
    - `rfft=True`: the last dimension is the non-redundant half from an rfft, which
      starts at DC (index 0) with or without fftshift and is not wrapped.

    `fftfreq(n) * n` isn't exact in float32, which can put edge samples just
    outside the array (e.g. -2.4e-7 for the lowest frequency at n = 7). Samplers
    zero, and insertion drops, anything outside the array, so coordinates within
    1e-3 of an edge are snapped onto it.

    Parameters
    ----------
    frequencies: torch.Tensor
        `(..., d)` array of multidimensional DFT sample frequencies in cycles per
        pixel.
    image_shape: Sequence[int]
        Length `d` real-space shape of the image. For an rfft, the last dimension is
        the length of the full (real space) dimension, not the half transform.
    rfft: bool
        Whether the DFT is the non-redundant half transform from `rfft`.
    fftshift: bool
        Whether the DFT is fftshifted.

    Returns
    -------
    coordinates: torch.Tensor
        `(..., d)` array of continuous coordinates into the DFT.
    """
    n = torch.as_tensor(
        tuple(image_shape), device=frequencies.device, dtype=frequencies.dtype
    )
    coordinates = frequencies * n
    tolerance = 1e-3

    # highest in-array coordinate along the rfft dimension, from DC at index 0
    rfft_upper = torch.div(n[-1], 2, rounding_mode="floor")

    if fftshift is True:
        dc_index = torch.div(n, 2, rounding_mode="floor")
        if rfft is True:
            dc_index[-1] = 0
        coordinates = coordinates + dc_index
        upper = n - 1
        if rfft is True:
            upper[-1] = rfft_upper
        lower = torch.zeros_like(upper)
        coordinates = torch.where(
            (coordinates < lower) & (coordinates > lower - tolerance),
            lower,
            coordinates,
        )
        coordinates = torch.where(
            (coordinates > upper) & (coordinates < upper + tolerance),
            upper,
            coordinates,
        )
        return coordinates

    # DC at index 0: snap tiny negative values onto DC before wrapping
    coordinates = torch.where(
        (coordinates < 0) & (coordinates > -tolerance),
        torch.zeros_like(coordinates),
        coordinates,
    )
    if rfft is True:
        full = coordinates[..., :-1]
        half = coordinates[..., -1:]
        half = torch.where(
            (half > rfft_upper) & (half < rfft_upper + tolerance), rfft_upper, half
        )
        return torch.cat([torch.remainder(full, n[:-1]), half], dim=-1)
    return torch.remainder(coordinates, n)


def dft_center(
    image_shape: tuple[int, ...],
    rfft: bool,
    fftshift: bool,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Return the position of the DFT center for a given input shape."""
    fft_center = torch.zeros(size=(len(image_shape),), device=device)
    image_shape = torch.as_tensor(image_shape, device=device).float()
    if rfft is True:
        image_shape = torch.tensor(rfft_shape(image_shape), device=device)
    if fftshift is True:
        fft_center = torch.divide(image_shape, 2, rounding_mode="floor")
    if rfft is True:
        fft_center[-1] = 0
    return fft_center.long()


def fftshift_1d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply fftshift to a 1D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, no shift is
        applied.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-1))
    else:
        output = input
    return output


def ifftshift_1d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply ifftshift to a 1D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, no shift is
        applied.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-1))
    else:
        output = input
    return output


def fftshift_2d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply fftshift to a 2D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, only the first
        dimension is shifted.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-2, -1))
    else:
        output = torch.fft.fftshift(input, dim=(-2,))
    return output


def ifftshift_2d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply ifftshift to a 2D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, only the first
        dimension is shifted.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-2, -1))
    else:
        output = torch.fft.ifftshift(input, dim=(-2,))
    return output


def fftshift_3d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply fftshift to a 3D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, only the first
        two dimensions are shifted.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-3, -2, -1))
    else:
        output = torch.fft.fftshift(input, dim=(-3, -2))
    return output


def ifftshift_3d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    """Apply ifftshift to a 3D tensor.

    Parameters
    ----------
    input: torch.Tensor
        Input tensor to shift.
    rfft: bool
        Whether the input is from a real-valued FFT. If True, only the first
        two dimensions are shifted.

    Returns
    -------
    output: torch.Tensor
        Shifted tensor.
    """
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-3, -2, -1))
    else:
        output = torch.fft.ifftshift(input, dim=(-3, -2))
    return output


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
