import pytest
import torch

from torch_simulate_image import PoissonConfig, poisson_sample


def test_poisson_disabled_returns_expected():
    expected = torch.full((16, 16), 10.0)
    config = PoissonConfig(apply=False)
    result = poisson_sample(expected, config)
    assert torch.allclose(result, expected)


def test_poisson_sample_mean_approximates_lambda():
    expected = torch.full((32, 32), 50.0)
    config = PoissonConfig(apply=True)
    samples = torch.stack(
        [poisson_sample(expected, config) for _ in range(200)],
        dim=0,
    )
    assert torch.allclose(samples.mean(), expected.mean(), rtol=0.05)


def test_poisson_sample_default_is_random_each_call():
    expected = torch.full((32, 32), 50.0)
    config = PoissonConfig(apply=True)
    first = poisson_sample(expected, config)
    second = poisson_sample(expected, config)
    assert not torch.equal(first, second)


def test_poisson_sample_deterministic_seed_is_reproducible():
    expected = torch.full((32, 32), 50.0)
    config = PoissonConfig(apply=True, deterministic=True, seed=0)
    first = poisson_sample(expected, config)
    second = poisson_sample(expected, config)
    assert torch.equal(first, second)


def test_poisson_config_requires_seed_when_deterministic():
    with pytest.raises(ValueError, match="seed"):
        PoissonConfig(apply=True, deterministic=True)
