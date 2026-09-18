import torch
from torch_fourier_filter.envelopes import Cc_envelope, Cs_envelope

from torch_simulate_image import (
    CtfConfig,
    EnvelopeConfig,
    FluenceConfig,
    apply_envelopes,
)


def test_apply_envelopes_disabled_returns_input():
    intensity = torch.rand(32, 32)
    result = apply_envelopes(
        intensity,
        EnvelopeConfig(apply=False, cs_envelope=True),
        pixel_size=1.0,
        fluence=FluenceConfig(),
        ctf=CtfConfig(),
    )
    assert torch.allclose(result, intensity)


def test_apply_envelopes_noop_when_no_subflags():
    intensity = torch.rand(16, 16)
    result = apply_envelopes(
        intensity,
        EnvelopeConfig(apply=True),
        pixel_size=1.0,
        fluence=FluenceConfig(),
        ctf=CtfConfig(),
    )
    assert torch.allclose(result, intensity)


def test_apply_cs_envelope_matches_primitive():
    intensity = torch.rand(32, 32)
    ctf = CtfConfig(defocus_um=1.5, spherical_aberration_mm=2.7, voltage_kv=300.0)
    config = EnvelopeConfig(
        apply=True,
        cs_envelope=True,
        illumination_semiangle_mrad=0.1,
    )
    result = apply_envelopes(
        intensity,
        config,
        pixel_size=1.2,
        fluence=FluenceConfig(),
        ctf=ctf,
    )
    expected_env = Cs_envelope(
        spherical_aberration=ctf.spherical_aberration_mm,
        defocus=ctf.defocus_um,
        image_shape=(32, 32),
        pixel_size=1.2,
        rfft=True,
        fftshift=False,
        voltage=ctf.voltage_kv,
        alpha=0.1,
    )
    expected = torch.fft.irfft2(
        torch.fft.rfft2(intensity) * expected_env,
        s=(32, 32),
    )
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_apply_cc_envelope_matches_primitive():
    intensity = torch.rand(32, 32)
    ctf = CtfConfig(voltage_kv=300.0)
    config = EnvelopeConfig(
        apply=True,
        cc_envelope=True,
        chromatic_aberration_mm=2.7,
        energy_spread_ev=0.7,
    )
    result = apply_envelopes(
        intensity,
        config,
        pixel_size=1.0,
        fluence=FluenceConfig(),
        ctf=ctf,
    )
    expected_env = Cc_envelope(
        chromatic_aberration=2.7,
        image_shape=(32, 32),
        pixel_size=1.0,
        rfft=True,
        fftshift=False,
        voltage=300.0,
        energy_spread=0.7,
    )
    expected = torch.fft.irfft2(
        torch.fft.rfft2(intensity) * expected_env,
        s=(32, 32),
    )
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)


def test_cs_and_cc_together_change_image():
    intensity = torch.rand(32, 32) + 0.5
    result = apply_envelopes(
        intensity,
        EnvelopeConfig(apply=True, cs_envelope=True, cc_envelope=True),
        pixel_size=1.0,
        fluence=FluenceConfig(),
        ctf=CtfConfig(defocus_um=2.0),
    )
    assert result.shape == intensity.shape
    assert not torch.allclose(result, intensity)
