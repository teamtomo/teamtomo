"""Internal helpers for preparing CTF calculation inputs."""

import einops
import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid, transform_fftfreq_grid


def infer_device(reference: float | torch.Tensor) -> torch.device:
    """Infer computation device from a reference input.

    Parameters
    ----------
    reference : float | torch.Tensor
        Reference input used to infer the target device.

    Returns
    -------
    device : torch.device
        Device to use for tensor creation.
    """
    if isinstance(reference, torch.Tensor):
        return reference.device
    return torch.device("cpu")


def as_float_tensor(value: float | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert a scalar-like input to a float tensor on the requested device.

    Parameters
    ----------
    value : float | torch.Tensor
        Input value to convert to a float tensor.
    device : torch.device
        Device to use for the resulting tensor.

    Returns
    -------
    tensor : torch.Tensor
        The input value converted to a float tensor on the specified device.
    """
    return torch.as_tensor(value, dtype=torch.float, device=device)


def prepare_frequency_grid_2d(
    image_shape: tuple[int, int] | torch.Tensor,
    pixel_size: float | torch.Tensor,
    rfft: bool,
    fftshift: bool,
    device: torch.device,
    transform_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a 2D frequency grid in cycles/Angstrom and its squared norm.

    Parameters
    ----------
    image_shape : tuple[int, int] | torch.Tensor
        Shape of the image, in real-space, for which to build the frequency grid.
    pixel_size : float | torch.Tensor
        Pixel size(s) in Angstroms.
    rfft : bool
        Whether the frequency grid is for an real-valued FFT (rfft) or a complex-valued
        FFT (fft). If True, then assumes a real-valued FFT.
    fftshift : bool
        Whether to apply an FFT shift to the frequency grid.
    device : torch.device
        Device to use for tensor creation.
    transform_matrix : torch.Tensor | None, optional
        Optional 2x2 transformation matrix to apply to the frequency grid.

    Returns
    -------
    fft_freq_grid : torch.Tensor
        The 2D frequency grid in cycles/Angstrom, with shape (..., Hf, Wf, 2) where
        Hf and Wf are the frequency dimensions in Fourier space and 2 corresponds to the
        (fx, fy) frequency components. The leading dimension can be batched if there are
        multiple pixel sizes.
    fft_freq_grid_squared : torch.Tensor
        The squared norm of the frequency grid, with shape (..., Hf, Wf), where Hf and
        Wf are the frequency dimensions in Fourier space.
    """
    pixel_size_tensor = as_float_tensor(pixel_size, device=device)

    fft_freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=False,
        device=device,
    )
    if transform_matrix is not None:
        fft_freq_grid = transform_fftfreq_grid(
            frequency_grid=fft_freq_grid,
            real_space_matrix=transform_matrix,
            device=device,
        )

    fft_freq_grid = fft_freq_grid / einops.rearrange(
        pixel_size_tensor, "... -> ... 1 1 1"
    )
    fft_freq_grid_squared = einops.reduce(
        fft_freq_grid**2, "... f->...", reduction="sum"
    )
    return fft_freq_grid, fft_freq_grid_squared
