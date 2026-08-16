"""Tests for joint defocus+thickness refine primitives."""

import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d

from torch_ctf_estimation.estimate_ctf_1d import refine_defocus_and_thickness_1d
from torch_ctf_estimation.estimate_ctf_2d import refine_defocus_and_thickness_2d
from torch_ctf_estimation.models import Defocus2DResults, LinearDefocusModel
from torch_ctf_estimation.utils.early_stopping import make_early_stopper


def _synthetic_rfft_power(h: int) -> torch.Tensor:
    return torch.rand(h, h // 2 + 1, dtype=torch.float32) + 0.01


def test_refine_defocus_and_thickness_1d_returns_updated_values():
    h = 32
    ps = _synthetic_rfft_power(h)
    result1d, thickness1d = refine_defocus_and_thickness_1d(
        power_spectrum=ps,
        image_sidelength=h,
        frequency_fit_range_angstroms=(20.0, 5.0),
        initial_defocus_um=2.0,
        initial_thickness_angstroms=1000.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast=0.07,
        pixel_spacing_angstroms=1.5,
        n_iterations=3,
        defocus_range_microns=(0.5, 5.0),
        thickness_range_angstroms=(300.0, 4000.0),
    )
    assert 0.5 <= float(result1d.ctf_model.defocus_um) <= 5.0
    assert 300.0 <= thickness1d.thickness_angstroms <= 4000.0
    assert thickness1d.cross_correlation_final is not None


def test_refine_thickness_only_keeps_defocus():
    h = 32
    ps = _synthetic_rfft_power(h)
    result1d, thickness1d = refine_defocus_and_thickness_1d(
        power_spectrum=ps,
        image_sidelength=h,
        frequency_fit_range_angstroms=(20.0, 5.0),
        initial_defocus_um=2.0,
        initial_thickness_angstroms=1000.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast=0.07,
        pixel_spacing_angstroms=1.5,
        n_iterations=8,
        defocus_range_microns=(0.5, 5.0),
        thickness_range_angstroms=(300.0, 4000.0),
        optimize_defocus=False,
    )
    assert float(result1d.ctf_model.defocus_um) == 2.0
    assert 300.0 <= thickness1d.thickness_angstroms <= 4000.0


def test_refine_defocus_and_thickness_2d_grid():
    t, gh, gw, ph, pw = 1, 2, 2, 32, 17
    patch_ps = torch.rand(t, gh, gw, ph, pw, dtype=torch.float32) + 0.01
    positions = torch.rand(t, gh, gw, 3, dtype=torch.float32)
    grid = CubicCatmullRomGrid3d.from_grid_data(torch.ones(1, 2, 2) * 1.8)
    result2d = Defocus2DResults(
        defocus_model_type="grid",
        defocus_model=grid,
        astigmatism=0.05,
        astigmatism_angle=10.0,
        envelope_B=20.0,
        defocus_u=1.825,
        defocus_v=1.775,
        phase_shift_degrees=0.0,
    )
    out2d, thick2d = refine_defocus_and_thickness_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=positions,
        result2d=result2d,
        initial_thickness_angstroms=900.0,
        frequency_fit_range_angstroms=(20.0, 5.0),
        pixel_spacing_angstroms=1.5,
        n_iterations=3,
        thickness_grid_resolution=(1, 1, 1),
    )
    assert out2d.defocus_model_type == "grid"
    assert thick2d.mean_thickness >= 1.0
    assert out2d.defocus_u is not None


def test_refine_defocus_and_thickness_2d_linear():
    t, gh, gw, ph, pw = 1, 2, 2, 32, 17
    patch_ps = torch.rand(t, gh, gw, ph, pw, dtype=torch.float32) + 0.01
    positions = torch.rand(t, gh, gw, 3, dtype=torch.float32)
    linear = LinearDefocusModel(
        defocus_0=1.6,
        defocus_gradient_magnitude=0.1,
        defocus_gradient_angle=30.0,
    )
    result2d = Defocus2DResults(
        defocus_model_type="linear",
        defocus_model=linear,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        envelope_B=0.0,
        defocus_u=1.6,
        defocus_v=1.6,
        phase_shift_degrees=0.0,
    )
    out2d, thick2d = refine_defocus_and_thickness_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=positions,
        result2d=result2d,
        initial_thickness_angstroms=1200.0,
        frequency_fit_range_angstroms=(25.0, 6.0),
        pixel_spacing_angstroms=1.2,
        n_iterations=2,
    )
    assert out2d.defocus_model_type == "linear"
    assert isinstance(out2d.defocus_model, LinearDefocusModel)
    assert thick2d.mean_thickness >= 1.0


def test_refine_defocus_and_thickness_2d_early_stops_before_n_iterations():
    t, gh, gw, ph, pw = 1, 2, 2, 32, 17
    patch_ps = torch.rand(t, gh, gw, ph, pw, dtype=torch.float32) + 0.01
    positions = torch.rand(t, gh, gw, 3, dtype=torch.float32)
    grid = CubicCatmullRomGrid3d.from_grid_data(torch.ones(1, 2, 2) * 1.8)
    result2d = Defocus2DResults(
        defocus_model_type="grid",
        defocus_model=grid,
        astigmatism=0.05,
        astigmatism_angle=10.0,
        envelope_B=20.0,
        defocus_u=1.825,
        defocus_v=1.775,
        phase_shift_degrees=0.0,
    )
    n_iterations = 40
    out2d, thick2d = refine_defocus_and_thickness_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=positions,
        result2d=result2d,
        initial_thickness_angstroms=900.0,
        frequency_fit_range_angstroms=(20.0, 5.0),
        pixel_spacing_angstroms=1.5,
        n_iterations=n_iterations,
        thickness_grid_resolution=(1, 1, 1),
        early_stopper=make_early_stopper(
            patience=1, window_size=2, tolerance=1e6
        ),
    )
    assert out2d.defocus_model_type == "grid"
    assert thick2d.loss_trace is not None
    assert 0 < len(thick2d.loss_trace) < n_iterations
