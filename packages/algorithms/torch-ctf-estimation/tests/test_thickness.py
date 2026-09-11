"""Tests for 1D and 2D sample thickness estimation."""

import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d

from torch_ctf_estimation.estimate_ctf_1d import estimate_thickness_1d
from torch_ctf_estimation.estimate_ctf_2d import estimate_thickness_2d
from torch_ctf_estimation.estimate_ctf_2d.ctf_loss_2d import compute_thickness_ctf_ps_t
from torch_ctf_estimation.estimate_ctf_2d.estimate_ctf_2d_utils import (
    _get_astig_clamped,
    _shared_astigmatism_and_env,
)
from torch_ctf_estimation.models import LaserParams


def _synthetic_rfft_power(h: int) -> torch.Tensor:
    """Positive (h, h//2+1) tensor as a stand-in for an rfft power spectrum."""
    return torch.rand(h, h // 2 + 1, dtype=torch.float32) + 0.01


def test_estimate_thickness_1d_returns_result_in_grid_range():
    """1D thickness grid search returns Thickness1DResults with thickness on a grid."""
    h = 64
    ps = _synthetic_rfft_power(h)
    result = estimate_thickness_1d(
        power_spectrum=ps,
        image_sidelength=h,
        frequency_fit_range_angstroms=(20.0, 5.0),
        defocus_um=2.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast=0.07,
        pixel_spacing_angstroms=1.5,
    )
    assert result.test_thicknesses is not None
    assert result.cross_correlations is not None
    assert result.test_thicknesses.shape[0] == result.cross_correlations.shape[0]
    # Default grid: 300 to 4000 inclusive in 100 Å steps → 38 points
    assert result.test_thicknesses.shape[0] == 38
    assert result.thickness_angstroms in {
        float(x) for x in result.test_thicknesses.cpu().tolist()
    }
    assert result.cross_correlation_final is not None
    assert -1.0 <= result.cross_correlation_final <= 1.0
    assert result.powerspectrum_1d is not None
    assert result.frequencies_1d.shape[0] == h // 2 + 1
    assert result.background_model is not None


def test_estimate_thickness_1d_custom_range_and_step():
    """Custom thickness range and step produce the expected grid length."""
    h = 32
    ps = _synthetic_rfft_power(h)
    result = estimate_thickness_1d(
        power_spectrum=ps,
        image_sidelength=h,
        frequency_fit_range_angstroms=(30.0, 6.0),
        defocus_um=1.5,
        voltage_kev=200.0,
        spherical_aberration_mm=2.0,
        amplitude_contrast=0.1,
        pixel_spacing_angstroms=1.0,
        thickness_range_angstroms=(500.0, 800.0),
        thickness_step_angstroms=100.0,
    )
    # 500, 600, 700, 800 → 4 points
    assert result.test_thicknesses is not None
    assert result.test_thicknesses.shape[0] == 4


def test_estimate_thickness_2d_scalar_defocus():
    """2D thickness optimisation runs with scalar defocus and returns a spline model."""
    t, gh, gw, ph, pw = 1, 2, 2, 64, 33
    patch_ps = torch.rand(t, gh, gw, ph, pw, dtype=torch.float32)
    positions = torch.rand(t, gh, gw, 3, dtype=torch.float32)

    result = estimate_thickness_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=positions,
        thickness_grid_resolution=(1, 2, 2),
        frequency_fit_range_angstroms=(20.0, 5.0),
        initial_thickness=1000.0,
        defocus=2.0,
        pixel_spacing_angstroms=1.5,
        n_iterations=5,
        thickness_lr=50.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast_fraction=0.07,
    )
    assert isinstance(result.thickness_model, CubicCatmullRomGrid3d)
    # Grid library stores an extra leading dim on .data
    assert result.thickness_model.data.squeeze(0).shape == (1, 2, 2)
    assert result.mean_thickness >= 1.0  # clamped minimum
    assert result.envelope_B is not None
    assert result.loss_trace is not None
    assert len(result.loss_trace) > 0


def test_estimate_thickness_2d_spline_defocus():
    """Fixed defocus may be a CubicCatmullRomGrid3d evaluated at patch positions."""
    t, gh, gw, ph, pw = 1, 2, 2, 48, 25
    patch_ps = torch.rand(t, gh, gw, ph, pw, dtype=torch.float32)
    positions = torch.rand(t, gh, gw, 3, dtype=torch.float32)
    defocus_grid = CubicCatmullRomGrid3d.from_grid_data(
        torch.ones(1, 2, 2, dtype=torch.float32) * 1.8
    )

    result = estimate_thickness_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=positions,
        thickness_grid_resolution=(1, 2, 2),
        frequency_fit_range_angstroms=(25.0, 5.0),
        initial_thickness=800.0,
        defocus=defocus_grid,
        pixel_spacing_angstroms=1.2,
        n_iterations=3,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast_fraction=0.1,
    )
    assert result.mean_thickness >= 1.0


def test_estimate_thickness_2d_with_laser_params():
    """LPP thickness CTF path when laser_params is set."""
    t, gh, gw, ph, pw = 1, 1, 1, 32, 17
    patch_ps = torch.rand(t, gh, gw, ph, pw, dtype=torch.float32)
    positions = torch.zeros(t, gh, gw, 3, dtype=torch.float32)
    positions[..., 0] = 0.5
    positions[..., 1] = 0.5
    positions[..., 2] = 0.5

    result = estimate_thickness_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=positions,
        thickness_grid_resolution=(1, 1, 1),
        frequency_fit_range_angstroms=(30.0, 6.0),
        initial_thickness=1500.0,
        defocus=2.5,
        pixel_spacing_angstroms=1.0,
        n_iterations=3,
        laser_params=LaserParams(model_laser=True),
    )
    assert result.thickness_model.data.squeeze(0).shape == (1, 1, 1)


def test_estimate_thickness_2d_debug_returns_traces():
    """debug=True populates model_trace and simulated_ps."""
    t, gh, gw, ph, pw = 1, 1, 1, 32, 17
    patch_ps = torch.rand(t, gh, gw, ph, pw, dtype=torch.float32)
    positions = torch.full((t, gh, gw, 3), 0.5, dtype=torch.float32)

    result = estimate_thickness_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=positions,
        thickness_grid_resolution=(1, 1, 1),
        frequency_fit_range_angstroms=(30.0, 6.0),
        initial_thickness=900.0,
        defocus=2.0,
        n_iterations=4,
        debug=True,
    )
    assert result.model_trace is not None
    assert len(result.model_trace) == 4
    assert result.simulated_ps is not None
    assert result.patch_power_spectra is not None


def test_compute_thickness_ctf_ps_t_shape_matches_patch_grid():
    """compute_thickness_ctf_ps_t return (gh, gw, ph, pw_rfft) for batched thickness."""
    ph, pw_rfft = 16, 9
    image_shape = (ph, (pw_rfft - 1) * 2)
    gh, gw = 2, 3
    device = torch.device("cpu")

    bp_filter, astig, angle_u, angle_v, _, _, _, env_2d = _shared_astigmatism_and_env(
        image_shape=image_shape,
        device=device,
        frequency_fit_range_angstroms=(40.0, 5.0),
        pixel_spacing_angstroms=1.0,
        initial_astigmatism=0.0,
        initial_astigmatism_angle=0.0,
        optimize_astigmatism=False,
        initial_envelope_B=0.0,
    )

    astig_c, ang_c = _get_astig_clamped(astig, angle_u, angle_v, False)

    thickness_t = torch.full((gh, gw), 1200.0, device=device)
    defocus_t = torch.full((gh, gw), 2.0, device=device)

    ps = compute_thickness_ctf_ps_t(
        thickness_t=thickness_t,
        defocus_t=defocus_t,
        astig_clamped=astig_c,
        astig_angle_clamped=ang_c,
        phase_shift_deg=0.0,
        image_shape=image_shape,
        pixel_spacing_angstroms=1.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast_fraction=0.1,
        env_2d=env_2d,
        bp_filter=bp_filter,
        laser_params=None,
    )
    assert ps.shape == (gh, gw, ph, pw_rfft)
