"""CTF module for Fourier filtering."""

import warnings

import torch
from torch_ctf import (
    calculate_additional_phase_shift as _calc_add_phase_shift,
)
from torch_ctf import (
    calculate_amplitude_contrast_equivalent_phase_shift as _calc_amp_contrast_phase_shift,  # noqa: E501
)
from torch_ctf import (
    calculate_ctf_1d as _calc_ctf_1d,
)
from torch_ctf import (
    calculate_ctf_2d as _calc_ctf_2d,
)
from torch_ctf import (
    calculate_defocus_phase_aberration as _calc_defocus_phase_aberration,
)
from torch_ctf import (
    calculate_relativistic_electron_wavelength as _calc_relativistic_wavelength,
)
from torch_ctf import (
    calculate_total_phase_shift as _calc_total_phase_shift,
)


def calculate_relativistic_electron_wavelength(
    energy: float | torch.Tensor,
) -> torch.Tensor:
    """Calculate the relativistic electron wavelength in SI units.

    For derivation see:
    1.  Kirkland, E. J. Advanced Computing in Electron Microscopy.
        (Springer International Publishing, 2020). doi:10.1007/978-3-030-33260-0.

    2.  https://en.wikipedia.org/wiki/Electron_diffraction#Relativistic_theory

    Parameters
    ----------
    energy : float | torch.Tensor
        Acceleration potential in volts.

    Returns
    -------
    wavelength : float | torch.Tensor
        Relativistic wavelength of the electron in meters.
    """
    warnings.warn(
        "This function is deprecated and will be removed in a future version. "
        "Please use torch_ctf.calculate_relativistic_electron_wavelength instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _calc_relativistic_wavelength(energy)


def calculate_defocus_phase_aberration(
    defocus_um: torch.Tensor,
    voltage_kv: torch.Tensor,
    spherical_aberration_mm: torch.Tensor,
    fftfreq_grid_angstrom_squared: torch.Tensor,
) -> torch.Tensor:
    """Calculate the phase aberration.

    Parameters
    ----------
    defocus_um : torch.Tensor
        Defocus in micrometers, positive is underfocused.
    voltage_kv : torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration_mm : torch.Tensor
        Spherical aberration in millimeters (mm).
    fftfreq_grid_angstrom_squared : torch.Tensor
        Precomputed squared frequency grid in Angstroms^-2.

    Returns
    -------
    phase_aberration : torch.Tensor
        The phase aberration for the given parameters.
    """
    warnings.warn(
        "This function is deprecated and will be removed in a future version. "
        "Please use torch_ctf.calculate_defocus_phase_aberration instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _calc_defocus_phase_aberration(
        defocus_um,
        voltage_kv,
        spherical_aberration_mm,
        fftfreq_grid_angstrom_squared,
    )


def calculate_additional_phase_shift(
    phase_shift_degrees: torch.Tensor,
) -> torch.Tensor:
    """Calculate additional phase shift from degrees to radians.

    Parameters
    ----------
    phase_shift_degrees : torch.Tensor
        Phase shift in degrees.

    Returns
    -------
    phase_shift_radians : torch.Tensor
        Phase shift in radians.
    """
    warnings.warn(
        "This function is deprecated and will be removed in a future version. "
        "Please use torch_ctf.calculate_additional_phase_shift instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _calc_add_phase_shift(phase_shift_degrees)


def calculate_amplitude_contrast_equivalent_phase_shift(
    amplitude_contrast_fraction: torch.Tensor,
) -> torch.Tensor:
    """Calculate the phase shift equivalent to amplitude contrast.

    Parameters
    ----------
    amplitude_contrast_fraction : torch.Tensor
        Amplitude contrast as a fraction (0 to 1).

    Returns
    -------
    phase_shift : torch.Tensor
        Phase shift equivalent to the given amplitude contrast.
    """
    warnings.warn(
        "This function is deprecated and will be removed in a future version. "
        "Please use torch_ctf.calculate_amplitude_contrast_equivalent_phase_shift "
        "instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _calc_amp_contrast_phase_shift(amplitude_contrast_fraction)


def calculate_total_phase_shift(
    defocus_um: torch.Tensor,
    voltage_kv: torch.Tensor,
    spherical_aberration_mm: torch.Tensor,
    phase_shift_degrees: torch.Tensor,
    amplitude_contrast_fraction: torch.Tensor,
    fftfreq_grid_angstrom_squared: torch.Tensor,
) -> torch.Tensor:
    """Calculate the total phase shift for the CTF.

    Parameters
    ----------
    defocus_um : torch.Tensor
        Defocus in micrometers, positive is underfocused.
    voltage_kv : torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration_mm : torch.Tensor
        Spherical aberration in millimeters (mm).
    phase_shift_degrees : torch.Tensor
        Phase shift in degrees.
    amplitude_contrast_fraction : torch.Tensor
        Amplitude contrast as a fraction (0 to 1).
    fftfreq_grid_angstrom_squared : torch.Tensor
        Precomputed squared frequency grid in Angstroms^-2.

    Returns
    -------
    total_phase_shift : torch.Tensor
        The total phase shift for the given parameters.
    """
    warnings.warn(
        "This function is deprecated and will be removed in a future version. "
        "Please use torch_ctf.calculate_total_phase_shift instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _calc_total_phase_shift(
        defocus_um,
        voltage_kv,
        spherical_aberration_mm,
        phase_shift_degrees,
        amplitude_contrast_fraction,
        fftfreq_grid_angstrom_squared,
    )


def calculate_ctf_2d(
    defocus: float | torch.Tensor,
    astigmatism: float | torch.Tensor,
    astigmatism_angle: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    image_shape: tuple[int, int],
    rfft: bool,
    fftshift: bool,
) -> torch.Tensor:
    """Calculate the Contrast Transfer Function (CTF) for a 2D image.

    NOTE: The device of the input tensors is inferred from the `defocus` tensor.

    Parameters
    ----------
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
        `(defocus_u + defocus_v) / 2`
    astigmatism : float | torch.Tensor
        Amount of astigmatism in micrometers.
        `(defocus_u - defocus_v) / 2`
    astigmatism_angle : float | torch.Tensor
        Angle of astigmatism in degrees. 0 places `defocus_u` along the y-axis.
    voltage : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration : float | torch.Tensor
        Spherical aberration in millimeters (mm).
    amplitude_contrast : float | torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    phase_shift : float | torch.Tensor
        Angle of phase shift applied to CTF in degrees.
    pixel_size : float | torch.Tensor
        Pixel size in Angströms per pixel (Å px⁻¹).
    image_shape : tuple[int, int]
        Shape of 2D images onto which CTF will be applied.
    rfft : bool
        Generate the CTF containing only the non-redundant half transform from a rfft.
    fftshift : bool
        Whether to apply fftshift on the resulting CTF images.

    Returns
    -------
    ctf : torch.Tensor
        The Contrast Transfer Function for the given parameters.
    """
    warnings.warn(
        "This function is deprecated and will be removed in a future version. "
        "Please use torch_ctf.calculate_ctf_2d instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _calc_ctf_2d(
        defocus=defocus,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift,
        pixel_size=pixel_size,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
    )


def calculate_ctf_1d(
    defocus: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    n_samples: int,
    oversampling_factor: int,
) -> torch.Tensor:
    """Calculate the Contrast Transfer Function (CTF) for a 1D signal.

    Parameters
    ----------
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
    voltage : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration : float | torch.Tensor
        Spherical aberration in millimeters (mm).
    amplitude_contrast : float | torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    phase_shift : float | torch.Tensor
        Angle of phase shift applied to CTF in degrees.
    pixel_size : float | torch.Tensor
        Pixel size in Angströms per pixel (Å px⁻¹).
    n_samples : int
        Number of samples in CTF.
    oversampling_factor : int
        Factor by which to oversample the CTF.

    Returns
    -------
    ctf : torch.Tensor
        The Contrast Transfer Function for the given parameters.
    """
    warnings.warn(
        "This function is deprecated and will be removed in a future version. "
        "Please use torch_ctf.calculate_ctf_1d instead.",
        FutureWarning,
        stacklevel=2,
    )
    return _calc_ctf_1d(
        defocus=defocus,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift,
        pixel_size=pixel_size,
        n_samples=n_samples,
        oversampling_factor=oversampling_factor,
    )
