"""CTF aberration calculation functions."""

import warnings

import torch
from scipy import constants as C


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
    h = C.Planck
    c = C.speed_of_light
    m0 = C.electron_mass
    e = C.elementary_charge
    V = torch.as_tensor(energy, dtype=torch.float)
    eV = e * V

    numerator = h * c
    denominator = torch.sqrt(eV * (2 * m0 * c**2 + eV))
    return numerator / denominator


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
    # Unit conversions
    defocus = defocus_um * 1e4  # micrometers -> angstroms
    voltage = voltage_kv * 1e3  # kV -> V
    spherical_aberration = spherical_aberration_mm * 1e7  # mm -> angstroms

    # derived quantities used in CTF calculation
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    )  # meters -> angstroms

    k1 = -torch.pi * _lambda
    k2 = torch.pi / 2 * spherical_aberration * _lambda**3
    return (
        k1 * fftfreq_grid_angstrom_squared * defocus
        + k2 * fftfreq_grid_angstrom_squared**2
    )


def beam_tilt_to_zernike_coeffs(
    beam_tilt_mrad: torch.Tensor,  # (..., 2) [bx, by] in mrad
    voltage_kv: torch.Tensor,
    spherical_aberration_mm: torch.Tensor,
) -> dict:
    """
    Convert beam tilt to Zernike axial coma coefficients Z31c, Z31s.

    Parameters
    ----------
    beam_tilt_mrad : torch.Tensor
        Beam tilt in milliradians.
    voltage_kv : torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration_mm : torch.Tensor
        Spherical aberration in millimeters (mm).

    Returns
    -------
    zernike_coeffs : dict
        Zernike axial coma coefficients Z31c, Z31s in radians.
    """
    # Convert mrad → rad
    beam_tilt_rad = beam_tilt_mrad * 1e-3

    voltage = voltage_kv * 1e3
    Cs = spherical_aberration_mm * 1e7  # mm → Å
    lam = calculate_relativistic_electron_wavelength(voltage) * 1e10  # Å

    prefactor = 2 * C.pi * Cs * lam**2

    return {
        "Z31c": prefactor * beam_tilt_rad[..., 0],
        "Z31s": prefactor * beam_tilt_rad[..., 1],
    }


def resolve_odd_zernikes(
    beam_tilt_mrad: torch.Tensor | None,
    odd_zernike_coeffs: dict | None,
    voltage_kv: torch.Tensor,
    spherical_aberration_mm: torch.Tensor,
) -> dict | None:
    """
    Resolve odd Zernike coefficients, handling beam tilt precedence.

    Parameters
    ----------
    beam_tilt_mrad : torch.Tensor | None
        Beam tilt in milliradians.
    odd_zernike_coeffs : dict | None
        Zernike axial coma coefficients Z31c, Z31s in radians.
    voltage_kv : torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration_mm : torch.Tensor
        Spherical aberration in millimeters (mm).

    Returns
    -------
    zernikes : dict | None
        Zernike axial coma coefficients Z31c, Z31s in radians.
        None if no beam tilt or Zernike coefficients are provided.
    """
    if beam_tilt_mrad is None and odd_zernike_coeffs is None:
        return None

    zernikes = {}

    if beam_tilt_mrad is not None:
        zernikes.update(
            beam_tilt_to_zernike_coeffs(
                beam_tilt_mrad,
                voltage_kv,
                spherical_aberration_mm,
            )
        )

    if odd_zernike_coeffs is not None:
        if beam_tilt_mrad is not None and any(
            k in odd_zernike_coeffs for k in ("Z31c", "Z31s")
        ):
            warnings.warn(
                "Both beam tilt and Zernike beam-tilt coefficients provided. "
                "Using Zernike coefficients and ignoring beam tilt.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Zernikes override beam tilt
        zernikes.update(odd_zernike_coeffs)

    return zernikes


def apply_odd_zernikes(
    odd_zernikes: dict | None,
    rho: torch.Tensor,
    theta: torch.Tensor,
    voltage_kv: torch.Tensor,
    spherical_aberration_mm: torch.Tensor,
    beam_tilt_mrad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply odd Zernike coefficients to the total phase shift.

    Parameters
    ----------
    odd_zernikes: dict | None
        Odd Zernike coefficients. Values can be floats or tensors.
        Floats will be converted to tensors for differentiability.
    rho: torch.Tensor
        Radial coordinate.
    theta: torch.Tensor
        Angular coordinate.
    voltage_kv: torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration_mm: torch.Tensor
        Spherical aberration in millimeters (mm).
    beam_tilt_mrad: torch.Tensor | None
        Beam tilt in milliradians.

    Returns
    -------
    phase: torch.Tensor
        Phase shift with odd Zernike coefficients applied.
    """
    odd_zernikes = resolve_odd_zernikes(
        beam_tilt_mrad=beam_tilt_mrad,
        odd_zernike_coeffs=odd_zernikes,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )
    if odd_zernikes is None:
        return torch.zeros_like(rho)

    phase = torch.zeros_like(rho)
    device = rho.device
    dtype = rho.dtype

    for name, coeff in odd_zernikes.items():
        # Convert float to tensor for differentiability
        if isinstance(coeff, int | float):
            coeff = torch.tensor(coeff, dtype=dtype, device=device, requires_grad=True)
        elif isinstance(coeff, torch.Tensor):
            # Ensure tensor is on correct device and dtype
            coeff = coeff.to(device=device, dtype=dtype)
        else:
            raise TypeError(
                f"Zernike coefficient must be float or torch.Tensor, got {type(coeff)}"
            )

        if name == "Z31c":  # beam tilt / axial coma x
            phase += coeff * rho**3 * torch.cos(theta)
        elif name == "Z31s":  # beam tilt / axial coma y
            phase += coeff * rho**3 * torch.sin(theta)
        elif name == "Z33c":  # trefoil
            phase += coeff * rho**3 * torch.cos(3 * theta)
        elif name == "Z33s":
            phase += coeff * rho**3 * torch.sin(3 * theta)
        else:
            raise ValueError(f"Unknown odd Zernike: {name}")

    return phase


def apply_even_zernikes(
    even_zernikes: dict,
    total_phase_shift: torch.Tensor,
    rho: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Apply even Zernike coefficients to the total phase shift.

    Parameters
    ----------
    even_zernikes: dict
        Even Zernike coefficients. Values can be floats or tensors.
        Floats will be converted to tensors for differentiability.
    total_phase_shift: torch.Tensor
        Total phase shift.
    rho: torch.Tensor
        Radial coordinate.
    theta: torch.Tensor
        Angular coordinate.

    Returns
    -------
    chi: torch.Tensor
        Phase shift with even Zernike coefficients applied.
    """
    chi = total_phase_shift.clone()
    device = rho.device
    dtype = rho.dtype

    for name, coeff in even_zernikes.items():
        # Convert float to tensor for differentiability
        if isinstance(coeff, int | float):
            coeff = torch.tensor(coeff, dtype=dtype, device=device, requires_grad=True)
        elif isinstance(coeff, torch.Tensor):
            # Ensure tensor is on correct device and dtype
            coeff = coeff.to(device=device, dtype=dtype)
        else:
            raise TypeError(
                f"Zernike coefficient must be float or torch.Tensor, got {type(coeff)}"
            )

        if name == "Z44c":  # 4-fold astigmatism
            chi += coeff * rho**4 * torch.cos(4 * theta)
        elif name == "Z44s":
            chi += coeff * rho**4 * torch.sin(4 * theta)
        elif name == "Z60":  # 6th-order spherical
            chi += coeff * rho**6
        else:
            raise ValueError(f"Unknown even Zernike: {name}")
    return chi
