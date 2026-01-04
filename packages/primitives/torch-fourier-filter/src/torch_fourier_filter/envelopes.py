"""Envelope functions for the Fourier filter."""

import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from torch_fourier_filter.ctf import calculate_relativistic_electron_wavelength


def b_envelope(
    B: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    pixel_size: float,
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create a B-factor envelope for a Fourier transform.

    Parameters
    ----------
    B: float
        The B-factor value.
        Suggested value is 5 A^2 / e-/A^2
    image_shape: tuple[int, ...]
        Shape of the real space the dft is from input image.
    pixel_size: float
        The pixel size of the image.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
    fftshift: bool
        Whether the input is fftshifted.
    device: torch.device
        Device to place tensors on.

    Returns
    -------
    torch.Tensor
        B-factor envelope
    """
    frequency_grid_px = (
        fftfreq_grid(
            image_shape=image_shape,
            rfft=rfft,
            fftshift=fftshift,
            norm=True,
            device=device,
        )
        / pixel_size
    )

    divisor = 4  # this is 4 for amplitude, 2 for intensity
    b_tensor = torch.exp(-(B * frequency_grid_px**2) / divisor)
    return b_tensor


def dose_envelope(
    fluence: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
    pixel_size: float,
    rfft: bool = True,
    fftshift: bool = False,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Create Grant and Grigorieff 2015 dose envelope for a Fourier transform.

    Parameters
    ----------
    fluence: float
        The fluence of the electron beam in e-/A^2.
    image_shape: tuple[int, ...]
        Shape of the real space the dft is from input image.
    pixel_size: float
        The pixel size of the image.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
    fftshift: bool
        Whether the input is fftshifted.
    a: float
        The a parameter of the dose envelope.
    b: float
        The b parameter of the dose envelope.
    c: float
        The c parameter of the dose envelope.
    device: torch.device
        Device to place tensors on.

    Returns
    -------
    torch.Tensor
        Dose envelope
    """
    frequency_grid_px = (
        fftfreq_grid(
            image_shape=image_shape,
            rfft=rfft,
            fftshift=fftshift,
            norm=True,
            device=device,
        )
        / pixel_size
    )
    if fluence < c:
        fluence_env = torch.ones_like(frequency_grid_px)
    else:
        fluence_env = torch.exp(-(fluence - c) / (a * torch.pow(frequency_grid_px, b)))

    return fluence_env


def Cs_envelope(
    spherical_aberration: float,  # in mm
    defocus: float,  # units in microns, positive for underfocus
    image_shape: tuple[int, int] | tuple[int, int, int],
    pixel_size: float,  # in angstroms
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device = None,
    voltage: float = 300,  # in kV
    alpha: float = 0.005,  # semiangle in mrad
) -> torch.Tensor:
    """
    Create a Cs envelope for a Fourier transform.

    Parameters
    ----------
    spherical_aberration: float
        The Cs value in mm
    defocus: float
        The defocus value in microns. Positive for underfocus.
    image_shape: tuple[int, ...]
        Shape of the real space the dft is from input image.
    pixel_size: float
        The pixel size of the image.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
    fftshift: bool
        Whether the input is fftshifted.
    device: torch.device
        Device to place tensors on.
    voltage: float
        The voltage of the microscope in kV.
    alpha: float
        The semiangle in mrad.

    Returns
    -------
    torch.Tensor
        Cs envelope
    """
    voltage *= 1e3  # kV -> V
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    )  # wavelength meters -> angstroms
    Cs = spherical_aberration * 1e7  # mm -> angstroms
    defocus *= 1e4  # microns -> angstroms

    frequency_grid_px = (
        fftfreq_grid(
            image_shape=image_shape,
            rfft=rfft,
            fftshift=fftshift,
            norm=True,
            device=device,
        )
        / pixel_size
    )

    Cs_env = torch.exp(
        -(((torch.pi * (alpha / 1000)) / _lambda) ** 2)
        * (
            Cs * _lambda**3 * frequency_grid_px**3
            + _lambda * (defocus) * frequency_grid_px
        )
        ** 2
    )

    return Cs_env


def Cc_envelope(
    chromatic_aberration: float,  # in mm
    image_shape: tuple[int, int] | tuple[int, int, int],
    pixel_size: float,  # in angstroms
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device = None,
    voltage: float = 300,  # in kV
    energy_spread: float = 0.7,  # in eV
    deltaV_V: float = 0.06e-6,  #
    deltaI_I: float = 0.01e-6,  #
) -> torch.Tensor:
    """
    Create a Cc envelope for a Fourier transform.

    Parameters
    ----------
    chromatic_aberration: float
        The Cc value in mm
    image_shape: tuple[int, ...]
        Shape of the real space the dft is from input image.
    pixel_size: float
        The pixel size of the image.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
    fftshift: bool
        Whether the input is fftshifted.
    device: torch.device
        Device to place tensors on.
    voltage: float
        The voltage of the microscope in kV.
    energy_spread: float
        The FWHM of the energy spread in eV.
    deltaV_V: float
        The relative voltage spread.
    deltaI_I: float
        The relative current spread

    Returns
    -------
    torch.Tensor
        Cc envelope
    """
    voltage *= 1e3  # kV -> V
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    )  # wavelength meters -> angstroms
    Cc = chromatic_aberration * 1e7  # mm -> angstroms

    frequency_grid_px = (
        fftfreq_grid(
            image_shape=image_shape,
            rfft=rfft,
            fftshift=fftshift,
            norm=True,
            device=device,
        )
        / pixel_size
    )

    focus_spread = Cc * (
        ((energy_spread / voltage) ** 2 + deltaV_V**2 + (2 * deltaI_I) ** 2) ** 0.5
    )
    Cc_env = torch.exp(
        -0.5 * ((torch.pi * _lambda * focus_spread * (frequency_grid_px) ** 2) ** 2)
    )

    return Cc_env
