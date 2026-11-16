import torch

from torch_simulate_image import CtfConfig, LppConfig, apply_ctf_to_exit_wave


def _structured_exit_wave(size: int = 64) -> torch.Tensor:
    """Non-uniform exit wave so CTF differences are visible beyond DC."""
    y = torch.linspace(-1, 1, size)
    x = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase = 0.3 * (xx**2 + yy**2)
    return torch.exp(1j * phase).to(torch.complex64)


def test_apply_ctf_disabled_returns_input():
    exit_wave = torch.ones(16, 16, dtype=torch.complex64)
    config = CtfConfig(apply=False)
    result = apply_ctf_to_exit_wave(exit_wave, config, pixel_size=1.0)
    assert torch.allclose(result, exit_wave)


def test_apply_ctf_changes_wave():
    exit_wave = _structured_exit_wave()
    config = CtfConfig(apply=True, defocus_um=1.5, voltage_kv=300.0)
    result = apply_ctf_to_exit_wave(exit_wave, config, pixel_size=1.0)
    assert result.shape == exit_wave.shape
    assert not torch.allclose(result, exit_wave)


def test_apply_ctf_voltage_affects_result():
    exit_wave = _structured_exit_wave()
    result_300 = apply_ctf_to_exit_wave(
        exit_wave, CtfConfig(defocus_um=1.5, voltage_kv=300.0), pixel_size=1.0
    )
    result_200 = apply_ctf_to_exit_wave(
        exit_wave, CtfConfig(defocus_um=1.5, voltage_kv=200.0), pixel_size=1.0
    )
    assert not torch.allclose(result_300, result_200)


def test_apply_ctf_with_astigmatism_beam_tilt_and_zernike():
    exit_wave = _structured_exit_wave()
    baseline = apply_ctf_to_exit_wave(
        exit_wave, CtfConfig(defocus_um=1.5), pixel_size=1.0
    )
    with_extras = apply_ctf_to_exit_wave(
        exit_wave,
        CtfConfig(
            defocus_um=1.5,
            astigmatism_um=0.05,
            astigmatism_angle_deg=30.0,
            beam_tilt_mrad=(0.1, -0.05),
            even_zernike_coeffs={"Z60": 0.1},
        ),
        pixel_size=1.0,
    )
    assert with_extras.shape == exit_wave.shape
    assert not torch.allclose(baseline, with_extras)


def test_apply_ctf_with_odd_zernikes_runs():
    exit_wave = _structured_exit_wave()
    result = apply_ctf_to_exit_wave(
        exit_wave,
        CtfConfig(defocus_um=1.5, odd_zernike_coeffs={"Z31c": 0.5, "Z33s": 0.2}),
        pixel_size=1.0,
    )
    assert result.shape == exit_wave.shape
    assert torch.isfinite(result.real).all()


def test_apply_ctf_lpp_changes_result_vs_standard():
    exit_wave = _structured_exit_wave()
    standard = apply_ctf_to_exit_wave(
        exit_wave, CtfConfig(defocus_um=0.0, phase_shift_deg=0.0), pixel_size=1.0
    )
    with_lpp = apply_ctf_to_exit_wave(
        exit_wave,
        CtfConfig(
            defocus_um=0.0,
            lpp=LppConfig(apply=True, peak_phase_deg=90.0),
        ),
        pixel_size=1.0,
    )
    assert with_lpp.shape == exit_wave.shape
    assert torch.isfinite(with_lpp.real).all()
    assert not torch.allclose(standard, with_lpp)


def test_apply_ctf_lpp_dual_laser_differs():
    exit_wave = _structured_exit_wave()
    single = apply_ctf_to_exit_wave(
        exit_wave,
        CtfConfig(defocus_um=0.5, lpp=LppConfig(apply=True, dual_laser=False)),
        pixel_size=1.0,
    )
    dual = apply_ctf_to_exit_wave(
        exit_wave,
        CtfConfig(defocus_um=0.5, lpp=LppConfig(apply=True, dual_laser=True)),
        pixel_size=1.0,
    )
    assert not torch.allclose(single, dual)
