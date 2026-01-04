"""Laser Phase Plate (LPP) CTF calculation functions."""

import einops
import numpy as np
import torch
from scipy import constants as C
from torch_grid_utils.fftfreq_grid import fftfreq_grid, transform_fftfreq_grid

from torch_ctf.ctf_2d import _setup_ctf_2d
from torch_ctf.ctf_aberrations import (
    apply_even_zernikes,
    apply_odd_zernikes,
    calculate_relativistic_electron_wavelength,
)
from torch_ctf.ctf_utils import calculate_total_phase_shift


def calculate_relativistic_gamma(
    energy: float | torch.Tensor,
) -> torch.Tensor:
    """Calculate the relativistic Lorentz factor (gamma).

    The Lorentz factor is defined as:

    γ = 1 + eV/(m₀c²)

    where:

    - e is the elementary charge
    - V is the acceleration potential
    - m₀ is the electron rest mass
    - c is the speed of light

    Parameters
    ----------
    energy : float | torch.Tensor
        Acceleration potential in volts.

    Returns
    -------
    gamma : torch.Tensor
        Relativistic Lorentz factor (dimensionless).
    """
    c = C.speed_of_light
    m0 = C.electron_mass
    e = C.elementary_charge
    V = torch.as_tensor(energy, dtype=torch.float)

    # γ = 1 + eV/(m₀c²)
    gamma = 1 + (e * V) / (m0 * c**2)
    return gamma


def calculate_relativistic_beta(
    energy: float | torch.Tensor,
) -> torch.Tensor:
    """Calculate the relativistic beta factor (v/c).

    The beta factor is defined as:

    β = v/c = √(1 - 1/γ²)

    where γ is the Lorentz factor and v is the electron velocity.

    Parameters
    ----------
    energy : float | torch.Tensor
        Acceleration potential in volts.

    Returns
    -------
    beta : torch.Tensor
        Relativistic beta factor (dimensionless, v/c).
    """
    gamma = calculate_relativistic_gamma(energy)

    # β = √(1 - 1/γ²) = √((γ² - 1)/γ²)
    beta = torch.sqrt((gamma**2 - 1) / gamma**2)
    return beta


def initialize_laser_params(
    NA: float,
    laser_wavelength_angstrom: float,
) -> tuple[float, float]:
    """Initialize the laser parameters.

    Parameters
    ----------
    NA : float
        Numerical aperture of the objective lens.
    laser_wavelength_angstrom : float
        Wavelength of the laser in Angstroms.

    Returns
    -------
    beam_waist : float
        Beam waist of the laser in Angstroms.
    rayleigh_range : float
        Rayleigh range of the laser in Angstroms.
    """
    beam_waist = laser_wavelength_angstrom / (np.pi * NA)
    rayleigh_range = beam_waist / NA
    return beam_waist, rayleigh_range


def make_laser_coords(
    fft_freq_grid_angstrom: torch.Tensor,
    electron_wavelength_angstrom: float,
    focal_length_angstrom: float,
    laser_xy_angle_deg: float,
    laser_long_offset_angstrom: float,
    laser_trans_offset_angstrom: float,
    beam_waist_angstroms: float,
    rayleigh_range_angstroms: float,
) -> torch.Tensor:
    """Make the laser coordinates for the CTF.

    Parameters
    ----------
    fft_freq_grid_angstrom : torch.Tensor
        FFT frequency grid in Angstroms^-1, shape (..., H, W, 2) with last dim [x, y].
    electron_wavelength_angstrom : float
        Electron wavelength in Angstroms.
    focal_length_angstrom : float
        Focal length in Angstroms.
    laser_xy_angle_deg : float
        Laser rotation angle in degrees.
    laser_long_offset_angstrom : float
        Longitudinal offset in Angstroms.
    laser_trans_offset_angstrom : float
        Transverse offset in Angstroms.
    beam_waist_angstroms : float
        Beam waist (w0) in Angstroms.
    rayleigh_range_angstroms : float
        Rayleigh range (zR) in Angstroms.

    Returns
    -------
    laser_coords : torch.Tensor
        Dimensionless laser coordinates, shape (..., H, W, 2) with last dim [Lx, Ly].
    """
    # Convert frequency coordinates to physical coordinates [A]
    physical_freq_coords = (
        fft_freq_grid_angstrom * electron_wavelength_angstrom * focal_length_angstrom
    )

    # Create rotation matrix for xy_angle
    angle_rad = torch.deg2rad(
        torch.tensor(laser_xy_angle_deg, device=physical_freq_coords.device)
    )
    cos_angle = torch.cos(angle_rad)
    sin_angle = torch.sin(angle_rad)

    # Rotation matrix: [[cos, -sin], [sin, cos]]
    rotation_matrix = torch.tensor(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]],
        device=physical_freq_coords.device,
        dtype=physical_freq_coords.dtype,
    )

    # Apply rotation: R @ coords (broadcasting over spatial dimensions)
    # physical_freq_coords is (..., H, W, 2), rotation_matrix is (2, 2)
    rotated_coords = torch.einsum(
        "...ij,jk->...ik", physical_freq_coords, rotation_matrix
    )

    # Apply translation offsets
    offset_tensor = torch.tensor(
        [laser_long_offset_angstrom, laser_trans_offset_angstrom],
        device=physical_freq_coords.device,
        dtype=physical_freq_coords.dtype,
    )
    translated_coords = rotated_coords - offset_tensor

    # Make dimensionless coordinates
    scale_tensor = torch.tensor(
        [rayleigh_range_angstroms, beam_waist_angstroms],
        device=physical_freq_coords.device,
        dtype=physical_freq_coords.dtype,
    )
    laser_coords = translated_coords / scale_tensor

    return laser_coords


def get_eta(
    eta0: float | torch.Tensor,
    laser_coords: torch.Tensor,
    beta: float | torch.Tensor,
    NA: float | torch.Tensor,
    pol_angle_deg: float | torch.Tensor,
    xz_angle_deg: float | torch.Tensor,
    laser_phi_deg: float | torch.Tensor,
) -> torch.Tensor:
    """Calculate eta (phase modulation) due to laser standing wave.

    Parameters
    ----------
    eta0 : float | torch.Tensor
        Base eta value.
    laser_coords : torch.Tensor
        Dimensionless laser coordinates from make_laser_coords,
        shape (..., H, W, 2) where last dim is [Lx, Ly].
    beta : float | torch.Tensor
        Beta parameter from scope.
    NA : float | torch.Tensor
        Numerical aperture of the laser.
    pol_angle_deg : float | torch.Tensor
        Polarization angle in degrees.
    xz_angle_deg : float | torch.Tensor
        XZ angle in degrees.
    laser_phi_deg : float | torch.Tensor
        Laser phi in degrees.

    Returns
    -------
    eta : torch.Tensor
        Phase modulation due to laser standing wave.
    """
    # Extract Lx and Ly from laser coordinates tensor
    Lx = laser_coords[..., 0]  # (..., H, W)
    Ly = laser_coords[..., 1]  # (..., H, W)

    # Convert parameters to tensors with proper device/dtype
    device = laser_coords.device
    dtype = laser_coords.dtype

    eta0 = torch.as_tensor(eta0, device=device, dtype=dtype)
    beta = torch.as_tensor(beta, device=device, dtype=dtype)
    NA = torch.as_tensor(NA, device=device, dtype=dtype)
    pol_angle_rad = torch.deg2rad(
        torch.as_tensor(pol_angle_deg, device=device, dtype=dtype)
    )
    xz_angle_rad = torch.deg2rad(
        torch.as_tensor(xz_angle_deg, device=device, dtype=dtype)
    )
    laser_phi_rad = torch.deg2rad(
        torch.as_tensor(laser_phi_deg, device=device, dtype=dtype)
    )

    # Calculate intermediate terms
    Lx_squared_plus_1 = 1 + Lx**2
    Ly_squared = Ly**2

    # Main calculation following the original formula
    # eta0/2 * exp(-2*Ly^2/(1+Lx^2)) / sqrt(1+Lx^2)
    base_term = (
        (eta0 / 2)
        * torch.exp(-2 * Ly_squared / Lx_squared_plus_1)
        / torch.sqrt(Lx_squared_plus_1)
    )

    # Calculate the complex modulation term
    # (1-2*beta^2*cos^2(pol_angle))
    pol_modulation = 1 - 2 * beta**2 * torch.cos(pol_angle_rad) ** 2

    # exp(-xz_angle^2 * (2/NA^2) * (1+Lx^2))
    xz_exp_term = torch.exp(-(xz_angle_rad**2) * (2 / NA**2) * Lx_squared_plus_1)

    # (1+Lx^2)^(-1/4)
    power_term = Lx_squared_plus_1 ** (-0.25)

    # cos(2*Lx*Ly^2/(1+Lx^2) + 4*Lx/NA^2 - 1.5*arctan(Lx) - laser_phi)
    phase_arg = (
        2 * Lx * Ly_squared / Lx_squared_plus_1
        + 4 * Lx / NA**2
        - 1.5 * torch.arctan(Lx)
        - laser_phi_rad
    )
    cos_term = torch.cos(phase_arg)

    # Combine all terms
    modulation_term = 1 + pol_modulation * xz_exp_term * power_term * cos_term

    return base_term * modulation_term


def get_eta0_from_peak_phase_deg(
    peak_phase_deg: float | torch.Tensor,
    laser_coords: torch.Tensor,
    beta: float | torch.Tensor,
    NA: float | torch.Tensor,
    pol_angle_deg: float | torch.Tensor,
    xz_angle_deg: float | torch.Tensor,
    laser_phi_deg: float | torch.Tensor,
) -> torch.Tensor:
    """Calculate eta0 from desired peak phase in degrees.

    This function iteratively determines the eta0 value needed to achieve
    a desired peak phase, accounting for the fact that the maximum phase
    may not occur at the expected location due to tilts and other effects.

    Parameters
    ----------
    peak_phase_deg : float | torch.Tensor
        Desired peak phase in degrees.
    laser_coords : torch.Tensor
        Dimensionless laser coordinates from make_laser_coords, shape (..., H, W, 2).
    beta : float | torch.Tensor
        Beta parameter from scope.
    NA : float | torch.Tensor
        Numerical aperture of the laser.
    pol_angle_deg : float | torch.Tensor
        Polarization angle in degrees.
    xz_angle_deg : float | torch.Tensor
        XZ angle in degrees.
    laser_phi_deg : float | torch.Tensor
        Laser phi in degrees.

    Returns
    -------
    eta0 : torch.Tensor
        Calibrated eta0 value in radians.
    """
    # Convert peak phase to radians for initial guess
    device = laser_coords.device
    dtype = laser_coords.dtype
    peak_phase_deg = torch.as_tensor(peak_phase_deg, device=device, dtype=dtype)
    eta0_test = torch.deg2rad(peak_phase_deg)  # [rad]

    # Calculate eta with the test eta0 value
    eta_test = get_eta(
        eta0=eta0_test,
        laser_coords=laser_coords,
        beta=beta,
        NA=NA,
        pol_angle_deg=pol_angle_deg,
        xz_angle_deg=xz_angle_deg,
        laser_phi_deg=laser_phi_deg,
    )

    # Find the actual peak phase achieved with this eta0
    peak_phase_deg_test = torch.rad2deg(eta_test.max())  # [deg]

    # Scale eta0 to achieve the desired peak phase
    # eta0_corrected = eta0_test * (desired_peak / actual_peak)
    eta0 = eta0_test * peak_phase_deg / peak_phase_deg_test  # [rad]

    return eta0


def calc_LPP_phase(
    fft_freq_grid: torch.Tensor,
    NA: float,
    laser_wavelength_angstrom: float,
    focal_length_angstrom: float,
    laser_xy_angle_deg: float,
    laser_xz_angle_deg: float,
    laser_long_offset_angstrom: float,
    laser_trans_offset_angstrom: float,
    laser_polarization_angle_deg: float,
    peak_phase_deg: float,
    voltage: float | torch.Tensor,
) -> torch.Tensor:
    """Calculate the laser phase plate phase modulation.

    Parameters
    ----------
    fft_freq_grid : torch.Tensor
        FFT frequency grid in cycles/Å, shape (..., H, W, 2).
    NA : float
        Numerical aperture of the laser.
    laser_wavelength_angstrom : float
        Wavelength of the laser in Angstroms.
    focal_length_angstrom : float
        Focal length in Angstroms.
    laser_xy_angle_deg : float
        Laser rotation angle in the xy plane in degrees.
    laser_xz_angle_deg : float
        Laser angle in the xz plane in degrees.
    laser_long_offset_angstrom : float
        Longitudinal offset of the laser in Angstroms.
    laser_trans_offset_angstrom : float
        Transverse offset of the laser in Angstroms.
    laser_polarization_angle_deg : float
        Polarization angle of the laser in degrees.
    peak_phase_deg : float
        Desired peak phase in degrees.
    voltage : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).

    Returns
    -------
    eta : torch.Tensor
        Laser phase modulation in radians.
    """
    # Calculate laser parameters
    beam_waist_angstroms, rayleigh_range_angstroms = initialize_laser_params(
        NA, laser_wavelength_angstrom
    )

    voltage_v = torch.as_tensor(voltage, dtype=torch.float) * 1e3  # Convert kV to V
    beta = calculate_relativistic_beta(voltage_v)
    electron_wavelength_angstrom = (
        calculate_relativistic_electron_wavelength(voltage_v) * 1e10
    )  # Convert m to Å

    # Make laser coordinates
    laser_coords = make_laser_coords(
        fft_freq_grid,
        electron_wavelength_angstrom,
        focal_length_angstrom,
        laser_xy_angle_deg,
        laser_long_offset_angstrom,
        laser_trans_offset_angstrom,
        beam_waist_angstroms,
        rayleigh_range_angstroms,
    )

    # Calculate laser phase (antinode configuration)
    laser_phi = 0  # antinode, 90 is node
    eta0 = get_eta0_from_peak_phase_deg(
        peak_phase_deg,
        laser_coords,
        beta,
        NA,
        laser_polarization_angle_deg,
        laser_xz_angle_deg,
        laser_phi,
    )

    eta = get_eta(
        eta0,
        laser_coords,
        beta,
        NA,
        laser_polarization_angle_deg,
        laser_xz_angle_deg,
        laser_phi,
    )

    return eta


def calc_LPP_ctf_2D(
    defocus: float | torch.Tensor,
    astigmatism: float | torch.Tensor,
    astigmatism_angle: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    image_shape: tuple[int, int],
    rfft: bool,
    fftshift: bool,
    # Laser parameters
    NA: float,
    laser_wavelength_angstrom: float,
    focal_length_angstrom: float,
    laser_xy_angle_deg: float,
    laser_xz_angle_deg: float,
    laser_long_offset_angstrom: float,
    laser_trans_offset_angstrom: float,
    laser_polarization_angle_deg: float,
    peak_phase_deg: float,
    beam_tilt_mrad: torch.Tensor | None = None,
    even_zernike_coeffs: dict | None = None,
    odd_zernike_coeffs: dict | None = None,
    transform_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate the Laser Phase Plate (LPP) modified CTF for a 2D image.

    This function is similar to calculate_ctf_2d but uses laser parameters to generate
    a spatially varying phase shift instead of a uniform phase shift.

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
    pixel_size : float | torch.Tensor
        Pixel size in Angströms per pixel (Å px⁻¹).
    image_shape : tuple[int, int]
        Shape of 2D images onto which CTF will be applied.
    rfft : bool
        Generate the CTF containing only the non-redundant half transform from a rfft.
    fftshift : bool
        Whether to apply fftshift on the resulting CTF images.
    NA : float
        Numerical aperture of the laser.
    laser_wavelength_angstrom : float
        Wavelength of the laser in Angstroms.
    focal_length_angstrom : float
        Focal length in Angstroms.
    laser_xy_angle_deg : float
        Laser rotation angle in the xy plane in degrees.
    laser_xz_angle_deg : float
        Laser angle in the xz plane in degrees.
    laser_long_offset_angstrom : float
        Longitudinal offset of the laser in Angstroms.
    laser_trans_offset_angstrom : float
        Transverse offset of the laser in Angstroms.
    laser_polarization_angle_deg : float
        Polarization angle of the laser in degrees.
    peak_phase_deg : float
        Desired peak phase in degrees.
    beam_tilt_mrad : torch.Tensor | None
        Beam tilt in milliradians. [bx, by] in mrad
    even_zernike_coeffs : dict | None
        Even Zernike coefficients.
        Example: {"Z44c": 0.1, "Z44s": 0.2, "Z60": 0.3}
    odd_zernike_coeffs : dict | None
        Odd Zernike coefficients.
        Example: {"Z31c": 0.1, "Z31s": 0.2, "Z33c": 0.3, "Z33s": 0.4}
    transform_matrix : torch.Tensor | None
        Optional 2x2 transformation matrix for anisotropic magnification.
        This should be the real-space transformation matrix A. The frequency-space
        transformation (A^-1)^T is automatically computed and applied.

    Returns
    -------
    ctf : torch.Tensor
        The Laser Phase Plate modified Contrast Transfer Function.
    """
    # Use _setup_ctf_2d to get the frequency grid and setup parameters
    (
        defocus,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        _,
        fft_freq_grid_squared,
        rho,
        theta,
    ) = _setup_ctf_2d(
        defocus=defocus,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=torch.tensor(0.0),  # Not used, will be replaced by laser phase
        pixel_size=pixel_size,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        transform_matrix=transform_matrix,
    )

    # Get the frequency grid for laser calculations
    # We need to reconstruct it from the squared version or get it from _setup_ctf_2d
    # For now, let's get it directly
    if isinstance(defocus, torch.Tensor):
        device = defocus.device
    else:
        device = torch.device("cpu")

    pixel_size_tensor = torch.as_tensor(pixel_size, dtype=torch.float, device=device)
    image_shape_tensor = torch.as_tensor(image_shape, dtype=torch.int, device=device)

    fft_freq_grid = fftfreq_grid(
        image_shape=image_shape_tensor,
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

    # Calculate laser phase using the dedicated function
    laser_phase_radians = calc_LPP_phase(
        fft_freq_grid=fft_freq_grid,
        NA=NA,
        laser_wavelength_angstrom=laser_wavelength_angstrom,
        focal_length_angstrom=focal_length_angstrom,
        laser_xy_angle_deg=laser_xy_angle_deg,
        laser_xz_angle_deg=laser_xz_angle_deg,
        laser_long_offset_angstrom=laser_long_offset_angstrom,
        laser_trans_offset_angstrom=laser_trans_offset_angstrom,
        laser_polarization_angle_deg=laser_polarization_angle_deg,
        peak_phase_deg=peak_phase_deg,
        voltage=voltage,
    )

    # Convert laser phase from radians to degrees for compatibility
    laser_phase_degrees = torch.rad2deg(laser_phase_radians)

    # Calculate total phase shift using laser phase instead of uniform phase shift
    total_phase_shift = calculate_total_phase_shift(
        defocus_um=defocus,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        phase_shift_degrees=laser_phase_degrees,  # Use spatially varying phase
        amplitude_contrast_fraction=amplitude_contrast,
        fftfreq_grid_angstrom_squared=fft_freq_grid_squared,
    )

    # Apply even Zernike coefficients if provided
    if even_zernike_coeffs is not None:
        total_phase_shift = apply_even_zernikes(
            even_zernike_coeffs,
            total_phase_shift,
            rho,
            theta,
        )

    # Calculate CTF
    ctf = -torch.sin(total_phase_shift)

    # Apply odd Zernike coefficients if provided
    if odd_zernike_coeffs is None and beam_tilt_mrad is None:
        return ctf

    antisymmetric_phase_shift = apply_odd_zernikes(
        odd_zernikes=odd_zernike_coeffs,
        rho=rho,
        theta=theta,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        beam_tilt_mrad=beam_tilt_mrad,
    )
    return ctf * torch.exp(1j * antisymmetric_phase_shift)
