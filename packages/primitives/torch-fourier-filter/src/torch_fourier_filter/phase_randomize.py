"""Phase randomization and permutation (fixed magnitude spectrum)."""

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
        Complex tensor containing 2/3D Fourier transform.
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

    if cuton <= 0:
        # Full spectrum: random phases everywhere (no frequency grid needed)
        random_phases = (
            torch.rand(dft.shape, device=dft.device, dtype=torch.float32)
            * (2 * torch.pi)
            - torch.pi
        )
        phase_factors = torch.complex(
            torch.cos(random_phases), torch.sin(random_phases)
        )
        return magnitudes * phase_factors

    # Create frequency grid
    # image_shape = dft.shape[-2:]
    freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=device,
    )

    # Create mask for frequencies at or above cuton (using absolute value)
    freq_mask = torch.abs(freq_grid) >= cuton

    # Generate random phases between -π and π only where |freq| >= cuton
    random_phases = torch.zeros_like(dft, dtype=torch.float32)
    random_phases[..., freq_mask] = (
        torch.rand(freq_mask.sum(), device=dft.device) * (2 * torch.pi) - torch.pi
    )

    # Convert to complex numbers (e^(iθ))
    phase_factors = torch.complex(torch.cos(random_phases), torch.sin(random_phases))

    # Keep original phases where |freq| < cuton
    original_phases = torch.angle(dft)
    original_phase_factors = torch.complex(
        torch.cos(original_phases), torch.sin(original_phases)
    )
    phase_factors[..., ~freq_mask] = original_phase_factors[..., ~freq_mask]

    # Combine with original magnitudes
    randomized = magnitudes * phase_factors

    return randomized


def phase_permutation(
    dft: torch.Tensor,
    image_shape: tuple[int, int] | tuple[int, int, int],
    rfft: bool = False,
    cuton: float = 0,
    fftshift: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """Permute Fourier phases while preserving magnitude spectrum.

    For a single 2D spectrum ``(h, w)`` or 3D volume ``(d, h, w)``, phase angles at
    frequencies with ``|f| >= cuton`` are shuffled among those bins; phases below
    ``cuton`` are unchanged. There is no separate batch dimension: ``dft.ndim`` must
    match ``len(image_shape)``.

    Parameters
    ----------
    dft: torch.Tensor
        Complex tensor with the same spatial shape as ``image_shape`` (2D or 3D).
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
        Output with permuted phases, same shape and dtype as ``dft``
    """
    # Get magnitude spectrum
    magnitudes = torch.abs(dft)
    original_phases = torch.angle(dft)

    if cuton <= 0:
        # Permute all phases (no frequency grid needed)
        permuted_phases = _permute_all_phases(original_phases)
    else:
        # Create frequency grid
        freq_grid = fftfreq_grid(
            image_shape=image_shape,
            rfft=rfft,
            fftshift=fftshift,
            norm=True,
            device=device,
        )
        # Create mask for frequencies at or above cuton (using absolute value)
        freq_mask = torch.abs(freq_grid) >= cuton
        # Shuffle phase values only where |freq| >= cuton
        permuted_phases = _permute_phases_at_mask(original_phases, freq_mask)

    # Convert to complex numbers (e^(iθ))
    phase_factors = torch.complex(
        torch.cos(permuted_phases), torch.sin(permuted_phases)
    )
    # Combine with original magnitudes
    return magnitudes * phase_factors


def _permute_all_phases(phases: torch.Tensor) -> torch.Tensor:
    flat = phases.reshape(-1)
    perm = torch.randperm(flat.numel(), device=phases.device)
    return flat[perm].reshape(phases.shape)


def _permute_phases_at_mask(
    original_phases: torch.Tensor,
    freq_mask: torch.Tensor,
) -> torch.Tensor:
    result = original_phases.clone()
    masked_vals = original_phases[..., freq_mask]
    n = masked_vals.numel()
    perm = torch.randperm(n, device=original_phases.device)
    shuffled = masked_vals.reshape(-1)[perm].reshape(masked_vals.shape)
    result[..., freq_mask] = shuffled
    return result
