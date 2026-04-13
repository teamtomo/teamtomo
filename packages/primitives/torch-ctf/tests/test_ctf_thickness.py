"""Tests for sample-thickness CTF variants."""

import pytest
import torch

from torch_ctf import calculate_total_phase_shift
from torch_ctf.ctf_thickness import (
    _ctf_from_thickness,
    calculate_ctf_thickness_1d,
    calculate_ctf_thickness_2d,
    calculate_ctf_thickness_lpp,
    calculate_ctf_with_thickness,
)


def test_ctf_thickness_1d_amplitude_small_t_matches_sin_chi():
    """Amplitude formulation with t->0: sinc(pi*lambda*g^2*t/2)->1 => sin(chi)."""
    t_small = 1e-25
    kwargs = {
        "defocus": 1.5,
        "voltage": 300.0,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.1,
        "phase_shift": 0.0,
        "pixel_size": 8.0,
        "n_samples": 32,
        "oversampling_factor": 1,
    }
    ctf = calculate_ctf_thickness_1d(False, t_small, **kwargs)
    ctf_ref = calculate_ctf_thickness_1d(False, t_small, **kwargs)
    assert torch.allclose(ctf, ctf_ref)
    g2 = (torch.linspace(0, 0.5, 32, dtype=torch.float32) / 8.0) ** 2
    chi = calculate_total_phase_shift(
        defocus_um=torch.tensor(1.5),
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
        phase_shift_degrees=torch.tensor(0.0),
        amplitude_contrast_fraction=torch.tensor(0.1),
        fftfreq_grid_angstrom_squared=g2,
    )
    expected = torch.sin(chi)
    assert torch.allclose(ctf, expected, atol=1e-5)


def test_ctf_thickness_1d_power_spectrum_small_t_matches_sin_squared_chi():
    """Power spectrum with t->0: half*(1-cos(2*chi)) = sin^2(chi)."""
    t_small = 1e-25
    kwargs = {
        "defocus": 1.5,
        "voltage": 300.0,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.1,
        "phase_shift": 0.0,
        "pixel_size": 8.0,
        "n_samples": 32,
        "oversampling_factor": 1,
    }
    ctf = calculate_ctf_thickness_1d(True, t_small, **kwargs)
    g2 = (torch.linspace(0, 0.5, 32, dtype=torch.float32) / 8.0) ** 2
    chi = calculate_total_phase_shift(
        defocus_um=torch.tensor(1.5),
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
        phase_shift_degrees=torch.tensor(0.0),
        amplitude_contrast_fraction=torch.tensor(0.1),
        fftfreq_grid_angstrom_squared=g2,
    )
    expected = 0.5 * (1.0 - torch.cos(2.0 * chi))
    assert torch.allclose(ctf, expected, atol=1e-5)


def test_ctf_thickness_router_matches_explicit_1d():
    kwargs = {
        "defocus": 1.5,
        "voltage": 300.0,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.1,
        "phase_shift": 0.0,
        "pixel_size": 8.0,
        "n_samples": 16,
        "oversampling_factor": 1,
    }
    a = calculate_ctf_thickness_1d(False, 100.0, **kwargs)
    b = calculate_ctf_with_thickness("1d", False, 100.0, **kwargs)
    assert torch.allclose(a, b)


def test_ctf_thickness_2d_amplitude_small_t_matches_sin_chi():
    from torch_ctf._ctf_core import _setup_ctf_context_2d

    t_small = 1e-25
    (
        defocus,
        voltage,
        sph,
        amp,
        phase,
        _,  # fft_freq_grid not used here
        g2,
        _rho,
        _theta,
    ) = _setup_ctf_context_2d(
        defocus=1.5,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0.0,
        pixel_size=8.0,
        image_shape=(8, 8),
        rfft=False,
        fftshift=False,
    )
    chi = calculate_total_phase_shift(
        defocus_um=defocus,
        voltage_kv=voltage,
        spherical_aberration_mm=sph,
        phase_shift_degrees=phase,
        amplitude_contrast_fraction=amp,
        fftfreq_grid_angstrom_squared=g2,
    )
    ctf = calculate_ctf_thickness_2d(
        False,
        t_small,
        defocus=1.5,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0.0,
        pixel_size=8.0,
        image_shape=(8, 8),
        rfft=False,
        fftshift=False,
    )
    assert torch.allclose(ctf, torch.sin(chi), atol=1e-4)


def test_ctf_thickness_lpp_shape_and_formulations_differ():
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
    t_angstrom = 200.0
    amp = calculate_ctf_thickness_lpp(False, t_angstrom, **common)
    ps = calculate_ctf_thickness_lpp(True, t_angstrom, **common)
    assert amp.shape == (10, 10)
    assert ps.shape == (10, 10)
    assert torch.all(torch.isfinite(amp))
    assert torch.all(torch.isfinite(ps))
    assert not torch.allclose(amp, ps, atol=1e-3)


def test_ctf_thickness_2d_power_spectrum_ignores_beam_tilt():
    """Power-spectrum CTF is real and unchanged by beam tilt / odd Zernikes."""
    common = {
        "return_power_spectrum": True,
        "sample_thickness_angstrom": 150.0,
        "defocus": 1.5,
        "astigmatism": 0.0,
        "astigmatism_angle": 0.0,
        "voltage": 300.0,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.1,
        "phase_shift": 0.0,
        "pixel_size": 8.0,
        "image_shape": (12, 12),
        "rfft": False,
        "fftshift": False,
    }
    base = calculate_ctf_thickness_2d(**common)
    with_tilt = calculate_ctf_thickness_2d(
        **common,
        beam_tilt_mrad=torch.tensor([[1.5, -0.5]]),
        odd_zernike_coeffs={"Z31c": torch.tensor(0.05)},
    )
    assert torch.allclose(base, with_tilt)
    assert not torch.is_complex(base)


def test_ctf_thickness_invalid_geometry():
    with pytest.raises(ValueError, match="geometry must be"):
        calculate_ctf_with_thickness(
            "3d",
            False,
            100.0,
            defocus=1.5,
            voltage=300.0,
            spherical_aberration=2.7,
            amplitude_contrast=0.1,
            phase_shift=0.0,
            pixel_size=8.0,
            n_samples=8,
            oversampling_factor=1,
        )


def test_ctf_from_thickness_broadcast_unsqueeze():
    """Cover broadcasting in _ctf_from_thickness: 0-dim lambda/t vs 2d grid."""
    lam = torch.tensor(0.02)  # 0-dim
    g2 = torch.rand(4, 4) * 1e-4
    chi = torch.randn(4, 4) * 0.5
    t = torch.tensor(100.0)  # 0-dim
    out = _ctf_from_thickness(False, lam, g2, chi, t)
    assert out.shape == (4, 4)
    out_ps = _ctf_from_thickness(True, lam, g2, chi, t)
    assert out_ps.shape == (4, 4)


def test_ctf_thickness_2d_with_even_zernikes():
    """Cover even_zernike_coeffs branch in calculate_ctf_thickness_2d (line 274)."""
    result = calculate_ctf_thickness_2d(
        False,
        100.0,
        defocus=1.5,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0.0,
        pixel_size=8.0,
        image_shape=(8, 8),
        rfft=False,
        fftshift=False,
        even_zernike_coeffs={"Z44c": torch.tensor(0.05), "Z60": torch.tensor(0.02)},
    )
    assert result.shape == (8, 8)
    assert torch.all(torch.isfinite(result))


def test_ctf_thickness_2d_amplitude_with_beam_tilt_returns_complex():
    """Cover odd Zernike/beam_tilt path in 2d (lines 286-293)."""
    result = calculate_ctf_thickness_2d(
        False,
        100.0,
        defocus=1.5,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0.0,
        pixel_size=8.0,
        image_shape=(8, 8),
        rfft=False,
        fftshift=False,
        beam_tilt_mrad=torch.tensor([[1.0, 0.5]]),
    )
    assert result.shape == (8, 8)
    assert torch.is_complex(result)


def test_ctf_thickness_router_2d_matches_explicit():
    """Cover geometry '2d' dispatch (lines 561-562)."""
    kwargs_2d = {
        "defocus": 1.5,
        "astigmatism": 0.0,
        "astigmatism_angle": 0.0,
        "voltage": 300.0,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.1,
        "phase_shift": 0.0,
        "pixel_size": 8.0,
        "image_shape": (8, 8),
        "rfft": False,
        "fftshift": False,
    }
    a = calculate_ctf_thickness_2d(False, 50.0, **kwargs_2d)
    b = calculate_ctf_with_thickness("2d", False, 50.0, **kwargs_2d)
    assert torch.allclose(a, b)


def test_ctf_thickness_router_lpp_matches_explicit():
    """Cover geometry 'lpp' dispatch (line 567)."""
    common_lpp = {
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
    a = calculate_ctf_thickness_lpp(False, 100.0, **common_lpp)
    b = calculate_ctf_with_thickness("lpp", False, 100.0, **common_lpp)
    assert torch.allclose(a, b)


def test_ctf_thickness_lpp_defocus_float_device_cpu():
    """Cover LPP branch when defocus is float not tensor (line 422)."""
    result = calculate_ctf_thickness_lpp(
        False,
        80.0,
        defocus=1.2,  # float
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8.0,
        image_shape=(8, 8),
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
    assert result.shape == (8, 8)


def test_ctf_thickness_lpp_with_transform_matrix():
    """Cover transform_matrix path in LPP (line 435)."""
    result = calculate_ctf_thickness_lpp(
        False,
        80.0,
        defocus=1.5,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8.0,
        image_shape=(8, 8),
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
        transform_matrix=torch.tensor([[1.02, 0.0], [0.0, 1.01]]),
    )
    assert result.shape == (8, 8)


def test_ctf_thickness_lpp_dual_laser():
    """Cover dual_laser=True path (lines 445, 458, 471)."""
    single = calculate_ctf_thickness_lpp(
        False,
        120.0,
        defocus=1.5,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8.0,
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
        dual_laser=False,
    )
    dual = calculate_ctf_thickness_lpp(
        False,
        120.0,
        defocus=1.5,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8.0,
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
    )
    assert single.shape == dual.shape
    assert not torch.allclose(single, dual)


def test_ctf_thickness_lpp_with_even_zernikes():
    """Cover even_zernike_coeffs in LPP (line 497)."""
    result = calculate_ctf_thickness_lpp(
        False,
        100.0,
        defocus=1.5,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8.0,
        image_shape=(8, 8),
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
        even_zernike_coeffs={"Z44c": torch.tensor(0.05)},
    )
    assert result.shape == (8, 8)


def test_ctf_thickness_lpp_amplitude_with_beam_tilt_returns_complex():
    """Cover odd Zernike/beam_tilt path in LPP (lines 509, 517)."""
    result = calculate_ctf_thickness_lpp(
        False,
        100.0,
        defocus=1.5,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8.0,
        image_shape=(8, 8),
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
        beam_tilt_mrad=torch.tensor([[1.0, 0.5]]),
    )
    assert result.shape == (8, 8)
    assert torch.is_complex(result)
