"""Pure-math primitives for scattering algorithms."""

import math
from numbers import Real

import torch
from scipy import constants as C
from torch_ctf import calculate_relativistic_electron_wavelength
from torch_grid_utils import fftfreq_grid

_SUPPORTED_POTENTIAL_DTYPES = {
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
}


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype corresponding to a supported potential dtype."""
    if dtype in {torch.float32, torch.complex64}:
        return torch.complex64
    return torch.complex128


def _validate_scalar(
    value: float | torch.Tensor,
    name: str,
) -> None:
    """Validate a positive, finite scalar without replacing differentiable tensors."""
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a scalar")
        if value.is_complex():
            raise ValueError(f"{name} must be real")
        if not bool(torch.isfinite(value).item()) or not bool((value > 0).item()):
            raise ValueError(f"{name} must be positive and finite")
    elif (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive, finite scalar")


def _validate_propagation_inputs(
    potential: torch.Tensor,
    pixel_size: float | torch.Tensor,
    voltage: float | torch.Tensor,
) -> None:
    """Validate inputs shared by all high-level propagation modes."""
    if not isinstance(potential, torch.Tensor):
        raise TypeError("potential must be a torch.Tensor")
    if potential.ndim < 3:
        raise ValueError("potential must have shape (..., Z, H, W)")
    if any(size == 0 for size in potential.shape[-3:]):
        raise ValueError("potential spatial dimensions Z, H, and W must be nonempty")
    if potential.dtype not in _SUPPORTED_POTENTIAL_DTYPES:
        raise TypeError(
            "potential dtype must be float32, float64, complex64, or complex128"
        )
    _validate_scalar(pixel_size, "pixel_size")
    _validate_scalar(voltage, "voltage")


def fresnel_propagator(
    frequency_grid: torch.Tensor,
    wavelength: float | torch.Tensor,
    dz: float | torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Fresnel free-space propagator for one multislice step.

    Parameters
    ----------
    frequency_grid : torch.Tensor
        Real-valued grid of spatial frequency magnitudes (1/Angstrom), e.g.
        from ``torch_grid_utils.fftfreq_grid(..., norm=True)``.
    wavelength : float | torch.Tensor
        Relativistic electron wavelength in Angstroms.
    dz : float | torch.Tensor
        Propagation distance (slice thickness) in Angstroms.

    Returns
    -------
    torch.Tensor
        Complex-valued Fresnel propagator with the same shape as
        `frequency_grid`.

    Notes
    -----
    ``H(k) = exp(i * pi * wavelength * dz * k^2)``. This is a pure phase
    transfer function, so ``|H(k)| == 1`` for all `k`.

    References
    ----------
    .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy,
       Springer US, Boston, MA, 2010.
    """
    return torch.exp(1j * C.pi * wavelength * dz * frequency_grid**2)


def transmission_function(
    potential_slice: torch.Tensor,
    sigma: float | torch.Tensor,
    dz: float | torch.Tensor,
) -> torch.Tensor:
    """
    Compute the transmission function for one electrostatic-potential slice.

    Parameters
    ----------
    potential_slice : torch.Tensor
        Electrostatic potential in volts for a single slice. Real-valued for
        a non-absorbing specimen, or complex-valued with a positive imaginary
        part to model absorption.
    sigma : float | torch.Tensor
        Electron-specimen interaction parameter, e.g. from
        `interaction_parameter`, in rad/(V*Angstrom).
    dz : float | torch.Tensor
        Slice thickness in Angstroms.

    Returns
    -------
    torch.Tensor
        Complex-valued transmission function, same shape as
        `potential_slice`.

    Notes
    -----
    ``T = exp(i * sigma * dz * V)``. The phase shift imparted by a slice is
    the line integral of `sigma * V` along the beam direction through the
    slice; treating `V` as constant over the (thin) slice thickness reduces
    that integral to `sigma * V * dz`. For real `V` this makes `T` a pure
    phase mask (``|T| == 1``); a positive imaginary part of `V` attenuates
    the wave.
    """
    return torch.exp(1j * sigma * dz * potential_slice)


def multislice_step(
    wave: torch.Tensor,
    potential_slice: torch.Tensor,
    propagator: torch.Tensor,
    sigma: float | torch.Tensor,
    dz: float | torch.Tensor,
) -> torch.Tensor:
    """
    Advance the electron wave through one electrostatic-potential slice.

    Parameters
    ----------
    wave : torch.Tensor
        Complex-valued incident wave for this slice, shape (..., H, W).
    potential_slice : torch.Tensor
        Electrostatic potential in volts for this slice, shape (..., H, W).
    propagator : torch.Tensor
        Fresnel propagator for this slice thickness, e.g. from
        `fresnel_propagator`, shape (H, W).
    sigma : float | torch.Tensor
        Electron-specimen interaction parameter, in rad/(V*Angstrom).
    dz : float | torch.Tensor
        Slice thickness in Angstroms.

    Returns
    -------
    torch.Tensor
        Complex-valued wave after transmission through and propagation
        past this slice, same shape as `wave`.

    Notes
    -----
    Each step alternates transmission through the slice (real space) with
    Fresnel propagation to the next slice (Fourier space):
    ``wave_out = IFFT(FFT(wave * T) * H)``
    """
    transmitted = wave * transmission_function(potential_slice, sigma, dz)
    propagated_wave: torch.Tensor = torch.fft.ifft2(
        torch.fft.fft2(transmitted) * propagator
    )
    return propagated_wave


def chunk_slices(total_slices: int, n_chunks: int) -> list[int]:
    """
    Compute contiguous, back-loaded chunk sizes for coarsened multislice.

    Parameters
    ----------
    total_slices : int
        Total number of slices along the beam direction.
    n_chunks : int
        Desired number of chunks. Must satisfy ``0 < n_chunks <= total_slices``.

    Returns
    -------
    list[int]
        Chunk sizes summing to `total_slices`, length `n_chunks`, suitable
        for `torch.split(..., split_size_or_sections=sizes, dim=...)`. Every
        chunk has ``total_slices // n_chunks`` slices except the last,
        which absorbs the remainder ``total_slices % n_chunks``. This is a
        back-loaded remainder, unlike `torch.chunk`/`torch.tensor_split`,
        which front-load it.

    Raises
    ------
    ValueError
        If `n_chunks` is not in ``(0, total_slices]``.

    Examples
    --------
    >>> chunk_slices(10, 3)
    [3, 3, 4]
    """
    if not (0 < n_chunks <= total_slices):
        raise ValueError(
            "n_chunks must satisfy 0 < n_chunks <= total_slices "
            f"({total_slices}), got {n_chunks}"
        )
    base = total_slices // n_chunks
    sizes = [base] * n_chunks
    sizes[-1] += total_slices - base * n_chunks
    return sizes


def interaction_parameter(voltage: float | torch.Tensor) -> float | torch.Tensor:
    """
    Calculate the electron-specimen interaction parameter.

    Computes the interaction constant sigma for electron scattering,
    following Kirkland Eq. (5.6). Wavelength is computed internally from
    `voltage` via `torch_ctf.calculate_relativistic_electron_wavelength`.

    Parameters
    ----------
    voltage : float | torch.Tensor
        Electron beam acceleration voltage in kilovolts.

    Returns
    -------
    float | torch.Tensor
        Interaction parameter sigma in units of rad/(V*Angstrom). A tensor
        if `voltage` is a tensor, otherwise a plain float.

    References
    ----------
    .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy,
       Eq. (5.6), Springer US, Boston, MA, 2010.
    """
    wavelength_m = calculate_relativistic_electron_wavelength(voltage * 1.0e3)
    wavelength = wavelength_m * 1.0e10  # meters -> Angstroms
    # Electron rest mass energy, m0*c^2, in electronvolts.
    rest_ev = C.electron_mass * C.speed_of_light**2 / C.elementary_charge
    ev = voltage * 1.0e3  # [eV]
    sigma: float | torch.Tensor = (
        2.0 * C.pi / (wavelength * ev) * ((ev + rest_ev) / (ev + 2.0 * rest_ev))
    )
    return sigma


def _prepare_propagation_parameters(
    potential: torch.Tensor,
    pixel_size: float | torch.Tensor,
    voltage: float | torch.Tensor,
) -> tuple[torch.Tensor, float | torch.Tensor, torch.Tensor]:
    """
    Compute the wavelength, interaction parameter, and frequency grid.

    Shared setup for the high-level wrapper functions (`firstborn`,
    `rytov`, `multislice`).

    Parameters
    ----------
    potential : torch.Tensor
        Real or complex 3D electrostatic potential in volts, shape
        (..., Z, H, W).
    pixel_size : float | torch.Tensor
        Isotropic voxel spacing in Angstroms, used as both the in-plane pixel
        spacing and slice thickness.
    voltage : float | torch.Tensor
        Electron beam acceleration voltage in kilovolts.

    Returns
    -------
    tuple[torch.Tensor, float | torch.Tensor, torch.Tensor]
        `wavelength` in Angstroms, `sigma` (interaction parameter) in
        rad/(V*Angstrom), and `frequency_grid`, the real-valued grid of
        spatial frequency magnitudes (1/Angstrom), shape (H, W).
    """
    _validate_propagation_inputs(potential, pixel_size, voltage)
    wavelength_m = calculate_relativistic_electron_wavelength(voltage * 1.0e3)
    wavelength = wavelength_m * 1.0e10  # meters -> Angstroms
    sigma = interaction_parameter(voltage=voltage)

    height, width = potential.shape[-2:]
    spacing = (
        float(pixel_size.detach().cpu())
        if isinstance(pixel_size, torch.Tensor)
        else float(pixel_size)
    )
    frequency_grid = fftfreq_grid(
        image_shape=(height, width),
        rfft=False,
        spacing=spacing,
        norm=True,
        device=potential.device,
    ).to(dtype=potential.real.dtype)
    return wavelength, sigma, frequency_grid
