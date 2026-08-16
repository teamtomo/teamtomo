"""Tests for equiphase 1D averaging (torch_ctf chi) and 1D spatial + EPA integration."""

import torch
from torch_fourier_filter.dft_utils import rotational_average_dft_2d

from torch_ctf_estimation.estimate_ctf import estimate_ctf
from torch_ctf_estimation.estimate_ctf_1d.equiphase_ctf_1d import (
    equiphase_average_power_to_1d_rfft,
)
from torch_ctf_estimation.models import CTFFittingParams, LaserParams, OpticalParams


def test_equiphase_matches_rotational_when_astigmatism_zero():
    """With zero astigmatism, equiphase 1D equals rotational average (same bins)."""
    h = 32
    # Constant power: equiphase reduces to shell means identical to rotational average.
    ps = torch.ones(h, h // 2 + 1)
    pixel = 1.0
    n_theta = 8
    epa = equiphase_average_power_to_1d_rfft(
        ps,
        h,
        pixel,
        defocus_um=1.5,
        astigmatism_um=0.0,
        astigmatism_angle_deg=0.0,
        phase_shift_deg=0.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast=0.1,
        laser_params=None,
        n_theta=n_theta,
    )
    rot, _ = rotational_average_dft_2d(
        ps.cpu(), image_shape=(h, h), rfft=True, fftshifted=False
    )
    rot = rot.to(ps.device)
    assert epa.shape == rot.shape
    torch.testing.assert_close(epa, rot, rtol=1e-4, atol=1e-4)


def test_equiphase_matches_rotational_for_radial_power():
    """Zero astig: EPA must stay on the local |q| shell, not a Cs-turnover alias."""
    h = 64
    pixel = 1.0
    fy = torch.fft.fftfreq(h, d=pixel)
    fx = torch.fft.rfftfreq(h, d=pixel)
    q = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    ps = q.clone()
    epa = equiphase_average_power_to_1d_rfft(
        ps,
        h,
        pixel,
        defocus_um=1.5,
        astigmatism_um=0.0,
        astigmatism_angle_deg=0.0,
        phase_shift_deg=0.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast=0.1,
        laser_params=None,
        n_theta=16,
    )
    rot, _ = rotational_average_dft_2d(
        ps.cpu(), image_shape=(h, h), rfft=True, fftshifted=False
    )
    # Skip DC / Nyquist and the first few sparse shells
    torch.testing.assert_close(epa[4:-2], rot[4:-2], rtol=0.15, atol=0.02)


def test_equiphase_negative_kx_hermitian_fold():
    """Negative kx must sample (|kx|, -ky), not the rFFT DC column."""
    h = 64
    pixel = 1.0
    fy = torch.fft.fftfreq(h, d=pixel)
    fx = torch.fft.rfftfreq(h, d=pixel)
    q = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    q0 = float(fx[6].item())
    ps = ((q - q0).abs() < (fx[1] * 1.6)).float()
    epa = equiphase_average_power_to_1d_rfft(
        ps,
        h,
        pixel,
        defocus_um=1.5,
        astigmatism_um=0.0,
        astigmatism_angle_deg=0.0,
        phase_shift_deg=0.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast=0.1,
        laser_params=None,
        n_theta=32,
    )
    i = int((fx - q0).abs().argmin().item())
    # Folding keeps samples on the ring (~1). Clamping half the angles to kx=0 → ~0.5.
    assert float(epa[i].item()) > 0.75
    """Nonzero astigmatism should change the 1D reduction vs circular averaging."""
    h = 48
    torch.manual_seed(0)
    ps = torch.abs(torch.randn(h, h // 2 + 1)) ** 2 + 0.01
    pixel = 1.0
    n_theta = 12
    epa = equiphase_average_power_to_1d_rfft(
        ps,
        h,
        pixel,
        defocus_um=1.2,
        astigmatism_um=0.15,
        astigmatism_angle_deg=33.0,
        phase_shift_deg=0.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast=0.1,
        laser_params=None,
        n_theta=n_theta,
    )
    rot, _ = rotational_average_dft_2d(
        ps.cpu(), image_shape=(h, h), rfft=True, fftshifted=False
    )
    rot = rot.to(ps.device)
    assert epa.shape == rot.shape
    assert (epa - rot).abs().max() > 1e-3


def test_equiphase_lpp_average_runs():
    """Smoke: equiphase 1D with LPP params completes."""
    h = 24
    ps = torch.ones(h, h // 2 + 1)
    lp = LaserParams(model_laser=True)
    out = equiphase_average_power_to_1d_rfft(
        ps,
        h,
        1.0,
        defocus_um=1.0,
        astigmatism_um=0.0,
        astigmatism_angle_deg=0.0,
        phase_shift_deg=0.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast=0.1,
        laser_params=lp,
        n_theta=6,
    )
    assert out.shape == (h // 2 + 1,)
    assert torch.all(torch.isfinite(out))


def test_estimate_ctf_whole_image_use_1d_spatial_equiphase_smoke():
    """End-to-end: whole-image + 1D spatial uses EPA by default (single patch)."""
    image = torch.randn(128, 128)
    optical = OpticalParams(
        pixel_spacing_angstroms=1.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast_fraction=0.1,
    )
    fitting = CTFFittingParams(
        defocus_grid_resolution=(1, 1, 1),
        frequency_fit_range_angstroms=(30.0, 5.0),
        defocus_range_microns=(0.5, 3.0),
        patch_sidelength=-1,
        use_1d_defocus_for_spatial=True,
        use_equiphase_for_1d_spatial=True,
        equiphase_n_theta=12,
        refine_steps_1d=5,
        n_iterations_2d=20,
        optimize_envelope_1d=False,
    )
    _mean_ps, _r1d, r2d = estimate_ctf(
        image, optical, fitting, device=torch.device("cpu")
    )
    assert r2d.defocus_model_type == "grid"
    assert r2d.defocus_model.data.numel() == 1
