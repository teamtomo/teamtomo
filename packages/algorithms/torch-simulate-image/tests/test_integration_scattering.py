import pytest
import torch
from torch_scattering import multislice

from torch_simulate_image import (
    CtfConfig,
    DqeConfig,
    FluenceConfig,
    MicrographSimulationConfig,
    PoissonConfig,
    simulate_micrograph,
)


@pytest.mark.slow
def test_potential_to_multislice_to_micrograph():
    pixel_size = 1.0
    shape = (4, 32, 32)
    potential = torch.zeros(shape)
    exit_wave = multislice(potential, pixel_size=pixel_size, voltage=300.0)
    config = MicrographSimulationConfig(
        pixel_size=pixel_size,
        ctf=CtfConfig(apply=True, defocus_um=1.5, voltage_kv=300.0),
        fluence=FluenceConfig(dose_e_per_A2=30.0),
        poisson=PoissonConfig(apply=False),
        dqe=DqeConfig(
            apply=True,
            mtf_frequencies=torch.tensor([0.0, 0.5]),
            mtf_amplitudes=torch.tensor([1.0, 1.0]),
        ),
        return_expected_counts=True,
    )
    micrograph = simulate_micrograph(exit_wave, config)
    assert micrograph.shape == exit_wave.shape
    assert torch.isfinite(micrograph).all()
