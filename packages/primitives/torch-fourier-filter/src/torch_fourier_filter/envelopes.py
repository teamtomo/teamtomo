"""Envelope functions for the Fourier filter."""

import torch
from torch_ctf import calculate_relativistic_electron_wavelength
from torch_grid_utils.fftfreq_grid import fftfreq_grid


def b_envelope(
    B: float,
    image_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    pixel_size: float | None = None,
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device = None,
    frequency_grid_px: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Create a B-factor envelope for a Fourier transform.

    Parameters
    ----------
    B: float
        The B-factor value.
        Suggested value is 5 A^2 / e-/A^2
    image_shape: tuple[int, ...] | None
        Shape of the real space the dft is from input image.
        Required when ``frequency_grid_px`` is not provided.
    pixel_size: float | None
        The pixel size of the image in Å.
        Required when ``frequency_grid_px`` is not provided.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
        Ignored when ``frequency_grid_px`` is provided.
    fftshift: bool
        Whether the input is fftshifted.
        Ignored when ``frequency_grid_px`` is provided.
    device: torch.device
        Device to place tensors on.
        Ignored when ``frequency_grid_px`` is provided.
    frequency_grid_px: torch.Tensor | None
        Pre-computed frequency grid in Å⁻¹, equivalent to
        ``fftfreq_grid(image_shape, norm=True) / pixel_size``.
        If provided, ``image_shape``, ``pixel_size``, ``rfft``, ``fftshift``,
        and ``device`` are ignored and no new grid is allocated.

    Returns
    -------
    torch.Tensor
        B-factor envelope
    """
    if frequency_grid_px is None:
        if image_shape is None or pixel_size is None:
            raise ValueError(
                "Provide either 'frequency_grid_px' (in Å⁻¹) or both 'image_shape' and 'pixel_size'."
            )
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
    image_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    pixel_size: float | None = None,
    rfft: bool = True,
    fftshift: bool = False,
    a: float = 0.245,
    b: float = -1.665,
    c: float = 2.81,
    device: torch.device = None,
    frequency_grid_px: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Create Grant and Grigorieff 2015 dose envelope for a Fourier transform.

    Parameters
    ----------
    fluence: float
        The fluence of the electron beam in e-/A^2.
    image_shape: tuple[int, ...] | None
        Shape of the real space the dft is from input image.
        Required when ``frequency_grid_px`` is not provided.
    pixel_size: float | None
        The pixel size of the image in Å.
        Required when ``frequency_grid_px`` is not provided.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
        Ignored when ``frequency_grid_px`` is provided.
    fftshift: bool
        Whether the input is fftshifted.
        Ignored when ``frequency_grid_px`` is provided.
    a: float
        The a parameter of the dose envelope.
    b: float
        The b parameter of the dose envelope.
    c: float
        The c parameter of the dose envelope.
    device: torch.device
        Device to place tensors on.
        Ignored when ``frequency_grid_px`` is provided.
    frequency_grid_px: torch.Tensor | None
        Pre-computed frequency grid in Å⁻¹, equivalent to
        ``fftfreq_grid(image_shape, norm=True) / pixel_size``.
        If provided, ``image_shape``, ``pixel_size``, ``rfft``, ``fftshift``,
        and ``device`` are ignored and no new grid is allocated.

    Returns
    -------
    torch.Tensor
        Dose envelope
    """
    if frequency_grid_px is None:
        if image_shape is None or pixel_size is None:
            raise ValueError(
                "Provide either 'frequency_grid_px' (in Å⁻¹) or both 'image_shape' and 'pixel_size'."
            )
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
    image_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    pixel_size: float | None = None,  # in angstroms
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device = None,
    voltage: float = 300,  # in kV
    alpha: float = 0.005,  # semiangle in mrad
    frequency_grid_px: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Create a Cs envelope for a Fourier transform.

    Parameters
    ----------
    spherical_aberration: float
        The Cs value in mm
    defocus: float
        The defocus value in microns. Positive for underfocus.
    image_shape: tuple[int, ...] | None
        Shape of the real space the dft is from input image.
        Required when ``frequency_grid_px`` is not provided.
    pixel_size: float | None
        The pixel size of the image in Å.
        Required when ``frequency_grid_px`` is not provided.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
        Ignored when ``frequency_grid_px`` is provided.
    fftshift: bool
        Whether the input is fftshifted.
        Ignored when ``frequency_grid_px`` is provided.
    device: torch.device
        Device to place tensors on.
        Ignored when ``frequency_grid_px`` is provided.
    voltage: float
        The voltage of the microscope in kV.
    alpha: float
        The semiangle in mrad.
    frequency_grid_px: torch.Tensor | None
        Pre-computed frequency grid in Å⁻¹, equivalent to
        ``fftfreq_grid(image_shape, norm=True) / pixel_size``.
        If provided, ``image_shape``, ``pixel_size``, ``rfft``, ``fftshift``,
        and ``device`` are ignored and no new grid is allocated.

    Returns
    -------
    torch.Tensor
        Cs envelope
    """
    if frequency_grid_px is None:
        if image_shape is None or pixel_size is None:
            raise ValueError(
                "Provide either 'frequency_grid_px' (in Å⁻¹) or both 'image_shape' and 'pixel_size'."
            )
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

    voltage *= 1e3  # kV -> V
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    )  # wavelength meters -> angstroms
    Cs = spherical_aberration * 1e7  # mm -> angstroms
    defocus *= 1e4  # microns -> angstroms

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
    image_shape: tuple[int, int] | tuple[int, int, int] | None = None,
    pixel_size: float | None = None,  # in angstroms
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device = None,
    voltage: float = 300,  # in kV
    energy_spread: float = 0.7,  # in eV
    deltaV_V: float = 0.06e-6,
    deltaI_I: float = 0.01e-6,
    frequency_grid_px: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Create a Cc envelope for a Fourier transform.

    Parameters
    ----------
    chromatic_aberration: float
        The Cc value in mm
    image_shape: tuple[int, ...] | None
        Shape of the real space the dft is from input image.
        Required when ``frequency_grid_px`` is not provided.
    pixel_size: float | None
        The pixel size of the image in Å.
        Required when ``frequency_grid_px`` is not provided.
    rfft: bool
        Whether the input is from an rfft (True) or full fft (False).
        Ignored when ``frequency_grid_px`` is provided.
    fftshift: bool
        Whether the input is fftshifted.
        Ignored when ``frequency_grid_px`` is provided.
    device: torch.device
        Device to place tensors on.
        Ignored when ``frequency_grid_px`` is provided.
    voltage: float
        The voltage of the microscope in kV.
    energy_spread: float
        The FWHM of the energy spread in eV.
    deltaV_V: float
        The relative voltage spread.
    deltaI_I: float
        The relative current spread
    frequency_grid_px: torch.Tensor | None
        Pre-computed frequency grid in Å⁻¹, equivalent to
        ``fftfreq_grid(image_shape, norm=True) / pixel_size``.
        If provided, ``image_shape``, ``pixel_size``, ``rfft``, ``fftshift``,
        and ``device`` are ignored and no new grid is allocated.

    Returns
    -------
    torch.Tensor
        Cc envelope
    """
    if frequency_grid_px is None:
        if image_shape is None or pixel_size is None:
            raise ValueError(
                "Provide either 'frequency_grid_px' (in Å⁻¹) or both 'image_shape' and 'pixel_size'."
            )
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

    voltage *= 1e3  # kV -> V
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    )  # wavelength meters -> angstroms
    Cc = chromatic_aberration * 1e7  # mm -> angstroms

    focus_spread = Cc * (
        ((energy_spread / voltage) ** 2 + deltaV_V**2 + (2 * deltaI_I) ** 2) ** 0.5
    )
    Cc_env = torch.exp(
        -0.5 * ((torch.pi * _lambda * focus_spread * (frequency_grid_px) ** 2) ** 2)
    )

    return Cc_env
