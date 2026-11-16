import torch

from torch_simulate_image import FluenceConfig, scale_to_expected_counts


def test_scale_to_expected_counts_uniform_intensity():
    intensity = torch.ones(64, 64)
    config = FluenceConfig(dose_e_per_A2=30.0)
    expected = scale_to_expected_counts(intensity, config, pixel_size=2.0)
    assert expected.shape == intensity.shape
    assert torch.allclose(expected.mean(), torch.tensor(30.0 * 4.0), rtol=1e-5)


def test_scale_to_expected_counts_with_coincidence_loss():
    intensity = torch.ones(16, 16)
    config = FluenceConfig(dose_e_per_A2=30.0, coincidence_loss=0.8)
    expected = scale_to_expected_counts(intensity, config, pixel_size=1.0)
    assert torch.allclose(expected.mean(), torch.tensor(30.0 * 0.8), rtol=1e-5)
