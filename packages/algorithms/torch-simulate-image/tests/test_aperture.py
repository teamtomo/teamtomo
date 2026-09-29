import pytest
import torch
from torch_ctf import calculate_relativistic_electron_wavelength

from torch_simulate_image import (
    ObjectiveApertureConfig,
    apply_objective_aperture,
)
from torch_simulate_image.optics.aperture import (
    make_objective_aperture_mask,
    resolve_aperture_cutoff_frequency,
)


def _structured_exit_wave(size: int = 64) -> torch.Tensor:
    y = torch.linspace(-1, 1, size)
    x = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase = 0.5 * (xx**2 + yy**2)
    return torch.exp(1j * phase).to(torch.complex64)


def test_aperture_disabled_returns_input():
    exit_wave = torch.ones(16, 16, dtype=torch.complex64)
    result = apply_objective_aperture(
        exit_wave,
        ObjectiveApertureConfig(apply=False),
        pixel_size=1.0,
        voltage_kv=300.0,
    )
    assert torch.allclose(result, exit_wave)


def test_aperture_config_requires_exactly_one_cutoff():
    with pytest.raises(ValueError, match="exactly one"):
        ObjectiveApertureConfig(apply=True)
    with pytest.raises(ValueError, match="exactly one"):
        ObjectiveApertureConfig(
            apply=True,
            outer_semiangle_mrad=5.0,
            cutoff_frequency_inv_A=0.2,
        )


def test_semiangle_to_frequency_conversion():
    voltage_kv = 300.0
    alpha_mrad = 10.0
    config = ObjectiveApertureConfig(apply=True, outer_semiangle_mrad=alpha_mrad)
    q_max = resolve_aperture_cutoff_frequency(config, voltage_kv=voltage_kv)
    wavelength_m = calculate_relativistic_electron_wavelength(voltage_kv * 1e3)
    wavelength_A = float(wavelength_m) * 1e10
    expected = (alpha_mrad * 1e-3) / wavelength_A
    assert abs(q_max - expected) < 1e-8


def test_hard_mask_zeros_high_frequencies():
    mask = make_objective_aperture_mask(
        (64, 64),
        pixel_size=1.0,
        q_max=0.15,
        soft_edge_half_width_inv_A=0.0,
    )
    assert mask.shape == (64, 64)
    assert torch.isclose(mask[0, 0], torch.tensor(1.0))
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    assert (mask == 0).any()
    assert (mask == 1).any()


def test_aperture_changes_structured_wave_and_can_reduce_power():
    exit_wave = _structured_exit_wave(64)
    result = apply_objective_aperture(
        exit_wave,
        ObjectiveApertureConfig(apply=True, cutoff_frequency_inv_A=0.1),
        pixel_size=1.0,
        voltage_kv=300.0,
    )
    assert result.shape == exit_wave.shape
    assert not torch.allclose(result, exit_wave)
    power_in = (exit_wave.real.square() + exit_wave.imag.square()).sum()
    power_out = (result.real.square() + result.imag.square()).sum()
    assert power_out < power_in


def test_soft_edge_mask_has_intermediate_values():
    mask = make_objective_aperture_mask(
        (64, 64),
        pixel_size=1.0,
        q_max=0.2,
        soft_edge_half_width_inv_A=0.05,
    )
    intermediate = (mask > 0.05) & (mask < 0.95)
    assert intermediate.any()
    assert torch.isclose(mask[0, 0], torch.tensor(1.0))
    assert (mask == 0).any()
