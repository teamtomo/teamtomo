"""Core CTF calculation utility functions."""

import torch

from torch_ctf.ctf_aberrations import calculate_defocus_phase_aberration


def calculate_additional_phase_shift(
    phase_shift_degrees: torch.Tensor,
) -> torch.Tensor:
    """Convert additional phase shift from degrees to radians.

    Parameters
    ----------
    phase_shift_degrees : torch.Tensor
        Phase shift in degrees.

    Returns
    -------
    phase_shift_radians : torch.Tensor
        Phase shift in radians.
    """
    return torch.deg2rad(phase_shift_degrees)


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
    return torch.arctan(
        amplitude_contrast_fraction / torch.sqrt(1 - amplitude_contrast_fraction**2)
    )


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
    phase_aberration = calculate_defocus_phase_aberration(
        defocus_um, voltage_kv, spherical_aberration_mm, fftfreq_grid_angstrom_squared
    )

    additional_phase_shift = calculate_additional_phase_shift(phase_shift_degrees)

    amplitude_contrast_phase_shift = (
        calculate_amplitude_contrast_equivalent_phase_shift(amplitude_contrast_fraction)
    )

    return phase_aberration - additional_phase_shift - amplitude_contrast_phase_shift
