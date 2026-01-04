"""Phase randomization function."""

import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid


def phase_randomize(
    dft: torch.Tensor,
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool = False,
    cuton: float = 0,
    fftshift: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """Phase randomize a 2/3D Fourier transform while preserving magnitude spectrum.

    Parameters
    ----------
    dft: torch.Tensor
        Complex tensor containing 2D Fourier transform(s). Can be batched with shape
        (batch, h, w) or unbatched (h, w).
    image_shape: tuple[int, ...]
        Shape of the input image
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False)
    cuton: float
        Fraction of Nyquist frequency to cut on
    fftshift: bool
        Whether the input is fftshifted
    device: torch.device
        Device to place tensors on

    Returns
    -------
    torch.Tensor
        Phase randomized version of input with same shape and dtype
    """
    # Get magnitude spectrum
    magnitudes = torch.abs(dft)

    # Create frequency grid
    # image_shape = dft.shape[-2:]
    freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=device,
    )

    # Create mask for frequencies above cuton
    freq_mask = freq_grid > cuton

    # Generate random phases between -π and π only where freq > cuton
    random_phases = torch.zeros_like(dft, dtype=torch.float32)
    random_phases[..., freq_mask] = (
        torch.rand(freq_mask.sum(), device=dft.device) * (2 * torch.pi) - torch.pi
    )

    # Convert to complex numbers (e^(iθ))
    phase_factors = torch.complex(torch.cos(random_phases), torch.sin(random_phases))

    # Keep original phases where freq <= cuton
    original_phases = torch.angle(dft)
    original_phase_factors = torch.complex(
        torch.cos(original_phases), torch.sin(original_phases)
    )
    phase_factors[..., ~freq_mask] = original_phase_factors[..., ~freq_mask]

    # Combine with original magnitudes
    randomized = magnitudes * phase_factors

    return randomized
