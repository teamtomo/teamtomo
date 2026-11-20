"""Phase randomization and permutation (fixed magnitude spectrum)."""

import torch
from torch_grid_utils.fftfreq_grid import (
    fftfreq_grid,
    fftshift_2d,
    fftshift_3d,
)


def _random_hermitian_phase_factors(
    image_shape: tuple[int, int] | tuple[int, int, int],
    batch_shape: tuple[int, ...],
    rfft: bool,
    fftshift: bool,
    device: torch.device,
) -> torch.Tensor:
    """Unit-magnitude random phase factors with Hermitian symmetry.

    Taken from the DFT of real white noise, so the phases are uniformly random
    while ``F[-k] == conj(F[k])`` holds: the inverse transform stays real and,
    for an rfft, the self-conjugate samples (x=0 plane mirrors, DC and even-size
    Nyquist) are consistent, for odd and even sizes.
    """
    noise = torch.randn((*batch_shape, *image_shape), device=device)
    dims = tuple(range(-len(image_shape), 0))
    noise_dft = (
        torch.fft.rfftn(noise, dim=dims) if rfft else torch.fft.fftn(noise, dim=dims)
    )
    phase_factors = noise_dft / noise_dft.abs().clamp_min(1e-12)
    if fftshift:
        shift = fftshift_2d if len(image_shape) == 2 else fftshift_3d
        phase_factors = shift(phase_factors, rfft=rfft)
    return phase_factors


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

    random_phase_factors = _random_hermitian_phase_factors(
        image_shape=image_shape,
        batch_shape=tuple(dft.shape[: dft.ndim - len(image_shape)]),
        rfft=rfft,
        fftshift=fftshift,
        device=dft.device,
    )

    if cuton <= 0:
        # Full spectrum: random phases everywhere (no frequency grid needed)
        return magnitudes * random_phase_factors

    # Create frequency grid
    freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=dft.device,
    )

    # Randomize phases only where |freq| >= cuton. |freq| is symmetric under
    # k -> -k, so mixing original and random phases keeps Hermitian symmetry.
    freq_mask = torch.abs(freq_grid) >= cuton
    original_phase_factors = torch.polar(torch.ones_like(magnitudes), torch.angle(dft))
    phase_factors = torch.where(freq_mask, random_phase_factors, original_phase_factors)

    return magnitudes * phase_factors


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
