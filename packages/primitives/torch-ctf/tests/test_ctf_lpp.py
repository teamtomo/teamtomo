"""Tests for laser phase plate utilities and CTF."""

import pytest
import torch

from torch_ctf import (
    calc_LPP_ctf_2D,
    calc_LPP_phase,
    calculate_relativistic_beta,
    calculate_relativistic_gamma,
    get_eta,
    get_eta0_from_peak_phase_deg,
    initialize_laser_params,
    make_laser_coords,
)


def test_calculate_relativistic_gamma():
    """Test relativistic gamma (Lorentz factor) calculation."""
    result = calculate_relativistic_gamma(300e3)
    assert isinstance(result, torch.Tensor)
    assert result > 1.0  # Gamma should always be > 1 for relativistic electrons
    assert torch.all(torch.isfinite(result))

    # Test with tensor input
    voltages = torch.tensor([100e3, 200e3, 300e3])
    result = calculate_relativistic_gamma(voltages)
    assert result.shape == (3,)
    # Higher voltage should give higher gamma
    assert torch.all(result[1:] > result[:-1])


def test_calculate_relativistic_beta():
    """Test relativistic beta (v/c) calculation."""
    result = calculate_relativistic_beta(300e3)
    assert isinstance(result, torch.Tensor)
    assert 0 < result < 1  # Beta should be between 0 and 1
    assert torch.all(torch.isfinite(result))

    # Test with tensor input
    voltages = torch.tensor([100e3, 200e3, 300e3])
    result = calculate_relativistic_beta(voltages)
    assert result.shape == (3,)
    # Higher voltage should give higher beta
    assert torch.all(result[1:] > result[:-1])


def test_initialize_laser_params():
    """Test laser parameter initialization."""
    NA = 0.1
    laser_wavelength_angstrom = 5000.0

    beam_waist, rayleigh_range = initialize_laser_params(NA, laser_wavelength_angstrom)

    assert isinstance(beam_waist, float)
    assert isinstance(rayleigh_range, float)
    assert beam_waist > 0
    assert rayleigh_range > 0
    assert rayleigh_range > beam_waist  # Rayleigh range should be larger


def test_make_laser_coords():
    """Test laser coordinate transformation."""
    # Create a simple frequency grid
    fft_freq_grid = torch.zeros((10, 10, 2))
    fft_freq_grid[..., 0] = torch.linspace(-0.5, 0.5, 10).unsqueeze(1).expand(10, 10)
    fft_freq_grid[..., 1] = torch.linspace(-0.5, 0.5, 10).unsqueeze(0).expand(10, 10)

    electron_wavelength_angstrom = 0.025
    focal_length_angstrom = 1e6
    laser_xy_angle_deg = 45.0
    laser_long_offset_angstrom = 0.0
    laser_trans_offset_angstrom = 0.0
    beam_waist_angstroms = 1000.0
    rayleigh_range_angstroms = 10000.0

    laser_coords = make_laser_coords(
        fft_freq_grid_angstrom=fft_freq_grid,
        electron_wavelength_angstrom=electron_wavelength_angstrom,
        focal_length_angstrom=focal_length_angstrom,
        laser_xy_angle_deg=laser_xy_angle_deg,
        laser_long_offset_angstrom=laser_long_offset_angstrom,
        laser_trans_offset_angstrom=laser_trans_offset_angstrom,
        beam_waist_angstroms=beam_waist_angstroms,
        rayleigh_range_angstroms=rayleigh_range_angstroms,
    )

    assert laser_coords.shape == (10, 10, 2)
    assert torch.all(torch.isfinite(laser_coords))


def test_get_eta():
    """Test eta (phase modulation) calculation."""
    eta0 = 0.1
    laser_coords = torch.zeros((10, 10, 2))
    laser_coords[..., 0] = torch.linspace(-1, 1, 10).unsqueeze(1).expand(10, 10)
    laser_coords[..., 1] = torch.linspace(-1, 1, 10).unsqueeze(0).expand(10, 10)

    beta = 0.5
    NA = 0.1
    pol_angle_deg = 0.0
    xz_angle_deg = 0.0
    laser_phi_deg = 0.0

    eta = get_eta(
        eta0=eta0,
        laser_coords=laser_coords,
        beta=beta,
        NA=NA,
        pol_angle_deg=pol_angle_deg,
        xz_angle_deg=xz_angle_deg,
        laser_phi_deg=laser_phi_deg,
    )

    assert eta.shape == (10, 10)
    assert torch.all(torch.isfinite(eta))
    assert torch.all(eta >= 0)  # Eta should be non-negative


def test_get_eta0_from_peak_phase_deg():
    """Test eta0 calibration from peak phase."""
    peak_phase_deg = 90.0
    laser_coords = torch.zeros((10, 10, 2))
    laser_coords[..., 0] = torch.linspace(-1, 1, 10).unsqueeze(1).expand(10, 10)
    laser_coords[..., 1] = torch.linspace(-1, 1, 10).unsqueeze(0).expand(10, 10)

    beta = 0.5
    NA = 0.1
    pol_angle_deg = 0.0
    xz_angle_deg = 0.0
    laser_phi_deg = 0.0

    eta0 = get_eta0_from_peak_phase_deg(
        peak_phase_deg=peak_phase_deg,
        laser_coords=laser_coords,
        beta=beta,
        NA=NA,
        pol_angle_deg=pol_angle_deg,
        xz_angle_deg=xz_angle_deg,
        laser_phi_deg=laser_phi_deg,
    )

    assert isinstance(eta0, torch.Tensor)
    assert torch.all(torch.isfinite(eta0))
    assert eta0 > 0

    # Verify that the calibrated eta0 produces the desired peak phase
    eta = get_eta(
        eta0=eta0,
        laser_coords=laser_coords,
        beta=beta,
        NA=NA,
        pol_angle_deg=pol_angle_deg,
        xz_angle_deg=xz_angle_deg,
        laser_phi_deg=laser_phi_deg,
    )
    actual_peak_deg = torch.rad2deg(eta.max())
    assert torch.allclose(actual_peak_deg, torch.tensor(peak_phase_deg), atol=1.0)


def test_calc_LPP_phase():
    """Test LPP phase calculation."""
    # Create frequency grid
    fft_freq_grid = torch.zeros((10, 10, 2))
    fft_freq_grid[..., 0] = torch.linspace(-0.5, 0.5, 10).unsqueeze(1).expand(10, 10)
    fft_freq_grid[..., 1] = torch.linspace(-0.5, 0.5, 10).unsqueeze(0).expand(10, 10)

    NA = 0.1
    laser_wavelength_angstrom = 5000.0
    focal_length_angstrom = 1e6
    laser_xy_angle_deg = 0.0
    laser_xz_angle_deg = 0.0
    laser_long_offset_angstrom = 0.0
    laser_trans_offset_angstrom = 0.0
    laser_polarization_angle_deg = 0.0
    peak_phase_deg = 90.0
    voltage = 300.0

    lpp_phase = calc_LPP_phase(
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

    assert lpp_phase.shape == (10, 10)
    assert torch.all(torch.isfinite(lpp_phase))


def test_calc_LPP_ctf_2D():
    """Test LPP-modified CTF calculation."""
    result = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # LPP CTF should be real (not complex) when no odd Zernikes
    assert not torch.is_complex(result)


def test_calc_LPP_ctf_2D_dual_laser():
    """Test LPP CTF with dual perpendicular laser option."""
    common = {
        "defocus": 1.5,
        "astigmatism": 0,
        "astigmatism_angle": 0,
        "voltage": 300,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.1,
        "pixel_size": 8,
        "image_shape": (10, 10),
        "rfft": False,
        "fftshift": False,
        "NA": 0.1,
        "laser_wavelength_angstrom": 5000.0,
        "focal_length_angstrom": 1e6,
        "laser_xy_angle_deg": 0.0,
        "laser_xz_angle_deg": 0.0,
        "laser_long_offset_angstrom": 0.0,
        "laser_trans_offset_angstrom": 0.0,
        "laser_polarization_angle_deg": 0.0,
        "peak_phase_deg": 90.0,
    }
    result_single = calc_LPP_ctf_2D(**common, dual_laser=False)
    result_dual = calc_LPP_ctf_2D(**common, dual_laser=True)
    assert result_single.shape == (10, 10)
    assert result_dual.shape == (10, 10)
    assert torch.all(torch.isfinite(result_single))
    assert torch.all(torch.isfinite(result_dual))
    assert not torch.is_complex(result_single)
    assert not torch.is_complex(result_dual)
    assert not torch.allclose(result_single, result_dual), (
        "dual_laser=True should differ from dual_laser=False"
    )


def test_calc_LPP_ctf_2D_with_zernikes():
    """Test LPP CTF with Zernike coefficients."""
    with pytest.warns(RuntimeWarning, match="Both beam tilt and Zernike"):
        result = calc_LPP_ctf_2D(
            defocus=1.5,
            astigmatism=0,
            astigmatism_angle=0,
            voltage=300,
            spherical_aberration=2.7,
            amplitude_contrast=0.1,
            pixel_size=8,
            image_shape=(10, 10),
            rfft=False,
            fftshift=False,
            NA=0.1,
            laser_wavelength_angstrom=5000.0,
            focal_length_angstrom=1e6,
            laser_xy_angle_deg=0.0,
            laser_xz_angle_deg=0.0,
            laser_long_offset_angstrom=0.0,
            laser_trans_offset_angstrom=0.0,
            laser_polarization_angle_deg=0.0,
            peak_phase_deg=90.0,
            beam_tilt_mrad=torch.tensor([[1.0, 2.0]]),
            even_zernike_coeffs={"Z44c": torch.tensor(0.1), "Z60": torch.tensor(0.2)},
            odd_zernike_coeffs={"Z31c": torch.tensor(0.1), "Z31s": torch.tensor(0.2)},
        )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # With Zernikes, result should be complex
    assert torch.is_complex(result)


def test_calc_LPP_ctf_2D_with_beam_tilt_only():
    """Test LPP CTF with only beam tilt."""
    result = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
        beam_tilt_mrad=torch.tensor([[1.0, 2.0]]),
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    assert torch.is_complex(result)


def test_calc_LPP_ctf_2D_return_complex_ctf_symmetric_path():
    """LPP return_complex_ctf: unit modulus; imag matches real WPOA when chi_a=0."""
    common = {
        "defocus": 1.5,
        "astigmatism": 0,
        "astigmatism_angle": 0,
        "voltage": 300,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.1,
        "pixel_size": 8,
        "image_shape": (10, 10),
        "rfft": False,
        "fftshift": False,
        "NA": 0.1,
        "laser_wavelength_angstrom": 5000.0,
        "focal_length_angstrom": 1e6,
        "laser_xy_angle_deg": 0.0,
        "laser_xz_angle_deg": 0.0,
        "laser_long_offset_angstrom": 0.0,
        "laser_trans_offset_angstrom": 0.0,
        "laser_polarization_angle_deg": 0.0,
        "peak_phase_deg": 90.0,
    }
    ctf_c = calc_LPP_ctf_2D(**common, return_complex_ctf=True)
    ctf_real = calc_LPP_ctf_2D(**common, return_complex_ctf=False)

    assert torch.is_complex(ctf_c)
    assert ctf_c.shape == ctf_real.shape
    assert torch.all(torch.isfinite(ctf_c))
    assert torch.allclose(torch.abs(ctf_c), torch.ones_like(ctf_c.real), atol=1e-5)
    assert torch.allclose(ctf_c.imag, ctf_real, atol=1e-5)


def test_calc_LPP_ctf_2D_return_complex_ctf_beam_tilt_unit_modulus():
    """LPP return_complex_ctf with beam tilt stays on the unit circle."""
    result = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
        beam_tilt_mrad=torch.tensor([[1.0, 2.0]]),
        return_complex_ctf=True,
    )
    assert result.shape == (10, 10)
    assert torch.is_complex(result)
    assert torch.all(torch.isfinite(result))
    assert torch.allclose(torch.abs(result), torch.ones_like(result.real), atol=1e-5)


def test_calc_LPP_ctf_2D_return_complex_ctf_dual_laser():
    """return_complex_ctf with dual_laser is complex-valued with unit modulus."""
    result = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
        dual_laser=True,
        return_complex_ctf=True,
    )
    assert result.shape == (10, 10)
    assert torch.is_complex(result)
    assert torch.allclose(torch.abs(result), torch.ones_like(result.real), atol=1e-5)


def test_calc_LPP_ctf_2D_with_even_zernikes():
    """Test LPP CTF with only even Zernike coefficients."""
    result = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
        even_zernike_coeffs={"Z44c": torch.tensor(0.1), "Z60": torch.tensor(0.2)},
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # With only even Zernikes, result should be real
    assert not torch.is_complex(result)


def test_calc_LPP_ctf_2D_with_transform_matrix():
    """Test LPP CTF with transform_matrix for anisotropic magnification."""
    # Create a simple scaling matrix (1.02x scaling in x, 1.01x scaling in y)
    # This represents anisotropic magnification
    transform_matrix = torch.tensor([[1.02, 0.0], [0.0, 1.01]])

    # Calculate LPP CTF without transform matrix
    result_no_transform = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
    )

    # Calculate LPP CTF with transform matrix
    result_with_transform = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
        transform_matrix=transform_matrix,
    )

    # Both should have the same shape
    assert result_no_transform.shape == (10, 10)
    assert result_with_transform.shape == (10, 10)

    # Both should be finite
    assert torch.all(torch.isfinite(result_no_transform))
    assert torch.all(torch.isfinite(result_with_transform))

    # The transform matrix should change the output (they should be different)
    assert not torch.allclose(result_no_transform, result_with_transform, atol=1e-6)
