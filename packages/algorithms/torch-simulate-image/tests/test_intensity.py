import pytest
import torch

from torch_simulate_image import exit_wave_to_intensity
from torch_simulate_image._validate import validate_exit_wave


def test_exit_wave_to_intensity_uniform():
    exit_wave = torch.ones(32, 32, dtype=torch.complex64)
    intensity = exit_wave_to_intensity(exit_wave)
    assert intensity.shape == (32, 32)
    assert torch.allclose(intensity, torch.ones(32, 32))


def test_exit_wave_to_intensity_batched():
    exit_wave = torch.randn(2, 3, 16, 16, dtype=torch.complex64)
    intensity = exit_wave_to_intensity(exit_wave)
    assert intensity.shape == (2, 3, 16, 16)


def test_validate_exit_wave_rejects_real():
    with pytest.raises(ValueError, match="complex"):
        validate_exit_wave(torch.ones(8, 8))
