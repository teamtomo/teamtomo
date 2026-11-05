"""Tests pinning the CTF sign convention.

Positive electrostatic potential multiplied by the CTF at underfocus gives a dark
image: the CTF is -amplitude_contrast at zero frequency (no phase shift) and its
first extremum is -1.
"""

import math

import pytest
import torch

from torch_ctf import (
    apply_odd_zernikes,
    calc_LPP_ctf_2D,
    calculate_ctf_1d,
    calculate_ctf_2d,
    calculate_ctf_thickness_1d,
    calculate_ctf_thickness_2d,
    calculate_ctf_thickness_lpp,
    calculate_ctfp_and_ctfq_2d,
)

# Parameters from teamtomo/teamtomo#153
ISSUE_PARAMS = {
    "defocus": 3.0,
    "voltage": 300.0,
    "spherical_aberration": 2.7,
    "amplitude_contrast": 0.07,
    "phase_shift": 0.0,
    "pixel_size": 1.0,
}

LPP_PARAMS = {
    "defocus": 0.5,
    "astigmatism": 0.0,
    "astigmatism_angle": 0.0,
    "voltage": 300.0,
    "spherical_aberration": 2.7,
    "amplitude_contrast": 0.07,
    "pixel_size": 1.0,
    "image_shape": (32, 32),
    "rfft": True,
    "fftshift": False,
    "NA": 0.05,
    "laser_wavelength_angstrom": 1.064e4,
    "focal_length_angstrom": 2e7,
    "laser_xy_angle_deg": 0.0,
    "laser_xz_angle_deg": 0.0,
    "laser_long_offset_angstrom": 0.0,
    "laser_trans_offset_angstrom": 0.0,
    "laser_polarization_angle_deg": 0.0,
    "peak_phase_deg": 90.0,
}


def _first_extremum(ctf: torch.Tensor) -> float:
    """Value of the first local extremum of a 1D CTF profile."""
    slope = torch.diff(ctf)
    turning = torch.nonzero(slope[1:] * slope[:-1] <= 0)[0].item() + 1
    return ctf[turning].item()


def _negate_frequency(x: torch.Tensor) -> torch.Tensor:
    """Map a full, unshifted 2D FFT layout from k to -k (odd sizes only)."""
    return torch.roll(torch.flip(x, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1))


@pytest.mark.parametrize("amplitude_contrast", [0.0, 0.07, 0.1, 0.3])
def test_ctf_1d_dc_is_minus_amplitude_contrast(amplitude_contrast):
    ctf = calculate_ctf_1d(
        **{**ISSUE_PARAMS, "amplitude_contrast": amplitude_contrast},
        n_samples=64,
        oversampling_factor=1,
    )
    assert ctf[0].item() == pytest.approx(-amplitude_contrast, abs=1e-6)


@pytest.mark.parametrize("phase_shift", [0.0, 30.0, 90.0])
def test_ctf_1d_dc_with_phase_shift(phase_shift):
    """A phase plate makes the low-frequency contrast more negative."""
    amplitude_contrast = 0.07
    ctf = calculate_ctf_1d(
        **{**ISSUE_PARAMS, "phase_shift": phase_shift},
        n_samples=64,
        oversampling_factor=1,
    )
    expected = -math.sin(math.radians(phase_shift) + math.asin(amplitude_contrast))
    assert ctf[0].item() == pytest.approx(expected, abs=1e-5)


def test_ctf_1d_first_extremum_is_minus_one_at_underfocus():
    ctf = calculate_ctf_1d(**ISSUE_PARAMS, n_samples=4096, oversampling_factor=1)
    assert ctf[0].item() == pytest.approx(-0.07, abs=1e-6)
    assert _first_extremum(ctf) == pytest.approx(-1.0, abs=1e-4)


def test_ctf_1d_first_extremum_is_plus_one_at_overfocus():
    ctf = calculate_ctf_1d(
        **{**ISSUE_PARAMS, "defocus": -3.0}, n_samples=4096, oversampling_factor=1
    )
    assert _first_extremum(ctf) == pytest.approx(1.0, abs=1e-4)


def test_ctf_2d_matches_issue_153_expectation():
    """Starts at -0.07 and goes to -1 before +1 along a frequency axis."""
    ctf = calculate_ctf_2d(
        **ISSUE_PARAMS,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        image_shape=(512, 512),
        rfft=True,
        fftshift=False,
    )
    profile = ctf[0, :]  # along kx
    assert profile[0].item() == pytest.approx(-0.07, abs=1e-6)
    # the grid samples the extremum coarsely
    assert _first_extremum(profile) == pytest.approx(-1.0, abs=1e-2)


def test_ctf_2d_positive_potential_is_dark_at_underfocus():
    """A positive Gaussian blob imaged at underfocus has a dark centre."""
    size, sigma = 128, 6.0
    coords = torch.arange(size) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    potential = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    potential = torch.fft.ifftshift(potential)  # centre at the origin

    ctf = calculate_ctf_2d(
        defocus=1.0,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        phase_shift=0.0,
        pixel_size=1.0,
        image_shape=(size, size),
        rfft=True,
        fftshift=False,
    )
    image = torch.fft.irfftn(torch.fft.rfftn(potential) * ctf, s=(size, size))
    assert image[0, 0] < 0
    assert image[0, 0] == image.min()


def test_complex_ctf_dc():
    ctf_c = calculate_ctf_2d(
        **ISSUE_PARAMS,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        image_shape=(16, 16),
        rfft=True,
        fftshift=False,
        return_complex_ctf=True,
    )
    assert -ctf_c[0, 0].imag.item() == pytest.approx(-0.07, abs=1e-6)
    assert ctf_c[0, 0].real.item() == pytest.approx(math.sqrt(1 - 0.07**2), abs=1e-6)


def test_complex_ctf_exit_wave_is_dark_at_underfocus():
    """The complex CTF applied to an exit wave exp(+i sigma V) gives a dark blob."""
    size, sigma = 129, 6.0
    coords = torch.arange(size) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    potential = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    exit_wave = torch.exp(1j * 0.05 * torch.fft.ifftshift(potential))

    ctf_c = calculate_ctf_2d(
        defocus=1.0,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.0,
        phase_shift=0.0,
        pixel_size=1.0,
        image_shape=(size, size),
        rfft=False,
        fftshift=False,
        return_complex_ctf=True,
    )
    image = torch.fft.ifftn(torch.fft.fftn(exit_wave) * ctf_c).abs() ** 2
    assert image[0, 0] < image[size // 2, size // 2]  # centre vs background


@pytest.mark.parametrize(
    "odd_aberrations",
    [
        {},
        {"beam_tilt_mrad": torch.tensor([[1.0, -2.0]])},
        {"odd_zernike_coeffs": {"Z31c": 5.0, "Z33s": -3.0}},
    ],
)
def test_complex_ctf_consistent_with_real_ctf(odd_aberrations):
    """ctf(k) = i (H(k) - conj(H(-k))) / 2 for complex CTF H, odd terms included.

    This is the weak-phase transfer of an exit wave exp(+i sigma V) through H.
    """
    kwargs = {
        **ISSUE_PARAMS,
        "astigmatism": 0.2,
        "astigmatism_angle": 30.0,
        "pixel_size": 2.0,
        "image_shape": (11, 11),
        "rfft": False,
        "fftshift": False,
        **odd_aberrations,
    }
    ctf = calculate_ctf_2d(**kwargs)
    ctf_c = calculate_ctf_2d(**kwargs, return_complex_ctf=True)
    from_complex = 1j * (ctf_c - _negate_frequency(ctf_c).conj()) / 2
    assert torch.allclose(
        torch.as_tensor(ctf, dtype=from_complex.dtype), from_complex, atol=1e-5
    )
    if not odd_aberrations:
        assert torch.allclose(-ctf_c.imag, ctf, atol=1e-6)


def test_odd_aberrations_keep_relion_phase_factor():
    """Odd aberrations multiply the real CTF by exp(+i chi_a), as in RELION."""
    kwargs = {
        **ISSUE_PARAMS,
        "astigmatism": 0.0,
        "astigmatism_angle": 0.0,
        "image_shape": (16, 16),
        "rfft": True,
        "fftshift": False,
    }
    beam_tilt = torch.tensor([[1.0, -2.0]])
    ctf = calculate_ctf_2d(**kwargs)
    ctf_tilted = calculate_ctf_2d(**kwargs, beam_tilt_mrad=beam_tilt)

    from torch_ctf._ctf_core import _setup_ctf_context_2d

    *_, rho, theta = _setup_ctf_context_2d(
        defocus=kwargs["defocus"],
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=kwargs["voltage"],
        spherical_aberration=kwargs["spherical_aberration"],
        amplitude_contrast=kwargs["amplitude_contrast"],
        phase_shift=kwargs["phase_shift"],
        pixel_size=kwargs["pixel_size"],
        image_shape=kwargs["image_shape"],
        rfft=True,
        fftshift=False,
    )
    chi_a = apply_odd_zernikes(
        odd_zernikes=None,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(kwargs["voltage"]),
        spherical_aberration_mm=torch.tensor(kwargs["spherical_aberration"]),
        beam_tilt_mrad=beam_tilt,
    )
    assert torch.allclose(ctf_tilted, ctf * torch.exp(1j * chi_a), atol=1e-5)


def test_ewald_ctfp_ctfq_sign():
    kwargs = {
        **ISSUE_PARAMS,
        "astigmatism": 0.0,
        "astigmatism_angle": 0.0,
        "image_shape": (16, 16),
        "rfft": True,
        "fftshift": False,
    }
    ctfp, ctfq = calculate_ctfp_and_ctfq_2d(**kwargs)
    ctf = calculate_ctf_2d(**kwargs)

    assert torch.allclose((ctfp + ctfq) / 2, ctf.to(ctfp.dtype), atol=1e-6)
    assert torch.allclose(ctfp, ctfq.conj(), atol=1e-6)
    # ctfp = sin(chi) - i cos(chi), the negative of RELION's CTFP
    assert ctfp[0, 0].real.item() == pytest.approx(-0.07, abs=1e-6)
    assert ctfp[0, 0].imag.item() == pytest.approx(-math.sqrt(1 - 0.07**2), abs=1e-6)


def test_ewald_matches_ctf_2d_with_beam_tilt():
    kwargs = {
        **ISSUE_PARAMS,
        "astigmatism": 0.1,
        "astigmatism_angle": 20.0,
        "image_shape": (16, 16),
        "rfft": True,
        "fftshift": False,
        "beam_tilt_mrad": torch.tensor([[1.0, -2.0]]),
    }
    ctfp, ctfq = calculate_ctfp_and_ctfq_2d(**kwargs)
    ctf = calculate_ctf_2d(**kwargs)
    # _enforce_symmetry edits the kx = 0 column, so compare elsewhere
    assert torch.allclose(((ctfp + ctfq) / 2)[:, 1:], ctf[:, 1:], atol=1e-5)


def test_ewald_ctfp_ctfq_from_complex_ctf():
    """ctfq = i H(k) and ctfp = -i conj(H(-k)), odd aberrations included."""
    kwargs = {
        **ISSUE_PARAMS,
        "astigmatism": 0.2,
        "astigmatism_angle": 30.0,
        "pixel_size": 2.0,
        "image_shape": (11, 11),
        "fftshift": False,
        "beam_tilt_mrad": torch.tensor([[1.0, -2.0]]),
    }
    ctfp, ctfq = calculate_ctfp_and_ctfq_2d(**kwargs, rfft=True)
    ctf_c = calculate_ctf_2d(**kwargs, rfft=False, return_complex_ctf=True)

    n_rfft = ctfp.shape[-1]
    expected_q = (1j * ctf_c)[:, :n_rfft]
    expected_p = (-1j * _negate_frequency(ctf_c).conj())[:, :n_rfft]
    # _enforce_symmetry edits the kx = 0 column, so compare elsewhere
    assert torch.allclose(ctfq[:, 1:], expected_q[:, 1:], atol=1e-5)
    assert torch.allclose(ctfp[:, 1:], expected_p[:, 1:], atol=1e-5)


def test_lpp_ctf_is_dark_at_low_frequency():
    ctf = calc_LPP_ctf_2D(**LPP_PARAMS)
    assert ctf[0, 0] < -0.5


def test_thickness_amplitude_ctf_matches_ctf_at_zero_thickness():
    one_d = {**ISSUE_PARAMS, "n_samples": 64, "oversampling_factor": 3}
    assert torch.allclose(
        calculate_ctf_thickness_1d(False, 0.0, **one_d),
        calculate_ctf_1d(**one_d),
        atol=1e-6,
    )

    two_d = {
        **ISSUE_PARAMS,
        "astigmatism": 0.2,
        "astigmatism_angle": 30.0,
        "image_shape": (16, 16),
        "rfft": True,
        "fftshift": False,
        "beam_tilt_mrad": torch.tensor([[1.0, -2.0]]),
    }
    assert torch.allclose(
        calculate_ctf_thickness_2d(False, 0.0, **two_d),
        calculate_ctf_2d(**two_d),
        atol=1e-6,
    )

    assert torch.allclose(
        calculate_ctf_thickness_lpp(False, 0.0, **LPP_PARAMS),
        calc_LPP_ctf_2D(**LPP_PARAMS),
        atol=1e-6,
    )
