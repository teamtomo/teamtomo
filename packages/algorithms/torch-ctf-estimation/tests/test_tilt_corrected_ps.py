"""Tests for tilt-corrected mean power spectrum (CTFFIND5-style scaling)."""

import torch
from torch_fourier_filter.dft_utils import rotational_average_dft_2d

from torch_ctf_estimation.models import (
    CTF,
    Defocus1DResults,
    Defocus2DResults,
    LinearDefocusModel,
    OpticalParams,
)
from torch_ctf_estimation.utils.tilt_corrected_ps import (
    effective_pixel_spacing_tilt_magnification,
    tilt_corrected_mean_ps_1d,
    tilt_corrected_mean_ps_2d,
)


def test_effective_pixel_spacing_matches_base_when_defocus_ratio_one():
    """Δf_local = Δf_avg ⇒ sqrt(|ratio|)=1 ⇒ effective spacing equals base."""
    px = effective_pixel_spacing_tilt_magnification(
        1.2,
        defocus_local_um=1.5,
        defocus_average_um=1.5,
    )
    assert abs(px - 1.2) < 1e-9


def _dummy_result1d(*, defocus_um: float, device: torch.device) -> Defocus1DResults:
    return Defocus1DResults(
        frequencies_1d=torch.zeros(1, device=device),
        ctf_model=CTF(
            defocus_um=torch.tensor(defocus_um, device=device),
            voltage_kev=torch.tensor(300.0, device=device),
            spherical_aberration_mm=torch.tensor(2.7, device=device),
            amplitude_contrast_fraction=torch.tensor(0.07, device=device),
            phase_shift_degrees=torch.tensor(0.0, device=device),
        ),
    )


def test_uniform_defocus_m_equals_one_matches_naive_mean_2d():
    """Zero defocus gradient: m=1; tilt-corrected 2D mean equals naive mean."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    h = 32
    w_r = h // 2 + 1
    t = gh = gw = 2
    patches = torch.abs(torch.randn(t, gh, gw, h, w_r, device=device)) ** 2 + 0.01
    naive_mean = patches.mean(dim=(0, 1, 2))

    d0 = 1.5
    result2d = Defocus2DResults(
        defocus_model_type="linear",
        defocus_model=LinearDefocusModel(
            defocus_0=d0,
            defocus_gradient_magnitude=0.0,
            defocus_gradient_angle=0.0,
        ),
        patch_power_spectra=patches,
        astigmatism=0.0,
        astigmatism_angle=0.0,
    )
    pos = torch.zeros(t, gh, gw, 3, device=device)
    pos[..., 0] = torch.linspace(0, 1, t, device=device).view(t, 1, 1)
    pos[..., 1] = (
        torch.linspace(0, 1, gh, device=device).view(1, gh, 1).expand(t, gh, gw)
    )
    pos[..., 2] = (
        torch.linspace(0, 1, gw, device=device).view(1, 1, gw).expand(t, gh, gw)
    )

    optics = OpticalParams(pixel_spacing_angstroms=1.0)
    r1d = _dummy_result1d(defocus_um=d0, device=device)
    mean_tc, aux = tilt_corrected_mean_ps_2d(
        r1d,
        result2d,
        normalised_patch_positions=pos,
        optical_params=optics,
    )
    torch.testing.assert_close(mean_tc, naive_mean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(aux["m"], torch.ones_like(aux["m"]), rtol=0, atol=1e-6)


def test_uniform_defocus_1d_matches_reduction_of_naive_mean():
    """Same setup: 1D path should match equiphase/rotational of the mean 2D spectrum."""
    torch.manual_seed(1)
    device = torch.device("cpu")
    h = 32
    w_r = h // 2 + 1
    t = gh = gw = 2
    patches = torch.abs(torch.randn(t, gh, gw, h, w_r, device=device)) ** 2 + 0.01
    naive_mean = patches.mean(dim=(0, 1, 2))

    d0 = 1.2
    result2d = Defocus2DResults(
        defocus_model_type="linear",
        defocus_model=LinearDefocusModel(
            defocus_0=d0,
            defocus_gradient_magnitude=0.0,
            defocus_gradient_angle=0.0,
        ),
        patch_power_spectra=patches,
        astigmatism=0.0,
        astigmatism_angle=0.0,
    )
    pos = torch.zeros(t, gh, gw, 3, device=device)
    pos[..., 0] = 0.5
    pos[..., 1] = (
        torch.linspace(0, 1, gh, device=device).view(1, gh, 1).expand(t, gh, gw)
    )
    pos[..., 2] = (
        torch.linspace(0, 1, gw, device=device).view(1, 1, gw).expand(t, gh, gw)
    )
    optics = OpticalParams(pixel_spacing_angstroms=1.0)
    r1d = _dummy_result1d(defocus_um=d0, device=device)

    ps_1d_tc, _ = tilt_corrected_mean_ps_1d(
        r1d,
        result2d,
        normalised_patch_positions=pos,
        optical_params=optics,
        use_equiphase=False,
    )
    rot_expected, _ = rotational_average_dft_2d(
        naive_mean.cpu(), image_shape=(h, h), rfft=True, fftshifted=False
    )
    rot_expected = rot_expected.to(device)
    torch.testing.assert_close(ps_1d_tc, rot_expected, rtol=1e-5, atol=1e-5)


def test_tilt_corrected_1d_equiphase_smoke_with_astigmatism():
    """Smoke: equiphase 1D after tilt-corrected mean 2D with nonzero astigmatism."""
    torch.manual_seed(2)
    device = torch.device("cpu")
    h = 24
    w_r = h // 2 + 1
    patches = torch.abs(torch.randn(1, 1, 1, h, w_r, device=device)) ** 2 + 0.05
    result2d = Defocus2DResults(
        defocus_model_type="linear",
        defocus_model=LinearDefocusModel(
            defocus_0=2.0,
            defocus_gradient_magnitude=0.0,
            defocus_gradient_angle=0.0,
        ),
        patch_power_spectra=patches,
        astigmatism=0.08,
        astigmatism_angle=12.0,
    )
    pos = torch.tensor([[[[0.5, 0.5, 0.5]]]], device=device)
    optics = OpticalParams(pixel_spacing_angstroms=1.0)
    r1d = _dummy_result1d(defocus_um=2.0, device=device)
    ps_1d, _ = tilt_corrected_mean_ps_1d(
        r1d,
        result2d,
        normalised_patch_positions=pos,
        optical_params=optics,
        use_equiphase=True,
        equiphase_n_theta=8,
    )
    assert ps_1d.ndim == 1
    assert ps_1d.shape[0] == h // 2 + 1
    assert torch.all(torch.isfinite(ps_1d))
