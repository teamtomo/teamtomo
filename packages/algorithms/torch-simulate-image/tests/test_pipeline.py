import torch

from torch_simulate_image import (
    CtfConfig,
    DqeConfig,
    FluenceConfig,
    MicrographSimulationConfig,
    PoissonConfig,
    simulate_micrograph,
    simulate_micrograph_from_intensity,
)


def _test_config(*, apply_ctf: bool = True) -> MicrographSimulationConfig:
    return MicrographSimulationConfig(
        pixel_size=1.0,
        ctf=CtfConfig(apply=apply_ctf, voltage_kv=300.0),
        fluence=FluenceConfig(dose_e_per_A2=20.0),
        poisson=PoissonConfig(apply=False),
        dqe=DqeConfig(
            apply=True,
            mtf_frequencies=torch.tensor([0.0, 0.5]),
            mtf_amplitudes=torch.tensor([1.0, 1.0]),
        ),
        return_expected_counts=True,
    )


def test_simulate_micrograph_from_uniform_exit_wave():
    exit_wave = torch.ones(32, 32, dtype=torch.complex64)
    micrograph = simulate_micrograph(exit_wave, _test_config())
    assert micrograph.shape == (32, 32)
    assert micrograph.ndim == 2
    assert torch.isfinite(micrograph).all()


def test_simulate_micrograph_from_intensity_path():
    intensity = torch.ones(16, 16)
    config = _test_config(apply_ctf=False)
    micrograph = simulate_micrograph_from_intensity(intensity, config)
    assert micrograph.shape == (16, 16)
    assert torch.allclose(micrograph.mean(), torch.tensor(20.0), rtol=1e-4)


def test_return_expected_counts_overrides_poisson():
    exit_wave = torch.ones(16, 16, dtype=torch.complex64)
    config = MicrographSimulationConfig(
        pixel_size=1.0,
        ctf=CtfConfig(apply=False, voltage_kv=300.0),
        fluence=FluenceConfig(dose_e_per_A2=25.0),
        poisson=PoissonConfig(apply=True, seed=0),
        return_expected_counts=True,
    )
    micrograph = simulate_micrograph(exit_wave, config)
    assert torch.allclose(micrograph, torch.full_like(micrograph, 25.0), rtol=1e-4)


def test_voltage_is_sourced_from_ctf_config():
    """Dose weighting / aperture / CTF all read ``ctf.voltage_kv``."""
    config = MicrographSimulationConfig(
        pixel_size=1.0,
        ctf=CtfConfig(voltage_kv=200.0),
    )
    assert config.ctf.voltage_kv == 200.0
    assert "voltage_kv" not in MicrographSimulationConfig.model_fields
