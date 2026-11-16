import pytest
import torch

from torch_simulate_image import PoissonConfig, poisson_sample


def _noise_correlation(a: torch.Tensor, b: torch.Tensor, expected: torch.Tensor):
    a, b = (a - expected).flatten(), (b - expected).flatten()
    return float((a @ b) / (a.norm() * b.norm()))


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


def test_poisson_sample_default_follows_torch_manual_seed():
    expected = torch.full((32, 32), 50.0)
    config = PoissonConfig(apply=True)
    torch.manual_seed(0)
    first = poisson_sample(expected, config)
    torch.manual_seed(0)
    second = poisson_sample(expected, config)
    assert torch.equal(first, second)


def test_poisson_sample_seed_and_index_is_reproducible():
    expected = torch.full((32, 32), 50.0)
    config = PoissonConfig(apply=True, seed=0)
    first = poisson_sample(expected, config, draw_index=3)
    second = poisson_sample(expected, config, draw_index=3)
    assert torch.equal(first, second)


def test_poisson_sample_draw_indices_are_independent():
    expected = torch.full((256, 256), 48.0)
    config = PoissonConfig(apply=True, seed=0)
    first = poisson_sample(expected, config, draw_index=0)
    second = poisson_sample(expected, config, draw_index=1)
    assert abs(_noise_correlation(first, second, expected)) < 0.02


def test_poisson_sample_seeds_do_not_collide_across_indices():
    """Seed 0's draw 1 must not be seed 1's draw 0."""
    expected = torch.full((32, 32), 50.0)
    first = poisson_sample(expected, PoissonConfig(seed=0), draw_index=1)
    second = poisson_sample(expected, PoissonConfig(seed=1), draw_index=0)
    assert not torch.equal(first, second)


def test_poisson_sample_seed_requires_draw_index():
    expected = torch.full((16, 16), 10.0)
    with pytest.raises(ValueError, match="draw_index"):
        poisson_sample(expected, PoissonConfig(apply=True, seed=0))


def test_poisson_sample_generator_takes_precedence():
    expected = torch.full((32, 32), 50.0)
    config = PoissonConfig(apply=True, seed=0)
    first = poisson_sample(expected, config, generator=torch.Generator().manual_seed(7))
    second = poisson_sample(
        expected, config, draw_index=5, generator=torch.Generator().manual_seed(7)
    )
    assert torch.equal(first, second)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seed", [None, 0])
def test_poisson_sample_consecutive_cuda_draws_are_independent(seed):
    """One shared CUDA generator correlates consecutive draws at ~48 counts/px."""
    expected = torch.full((512, 512), 48.0, device="cuda")
    config = PoissonConfig(apply=True, seed=seed)
    first = poisson_sample(expected, config, draw_index=0)
    second = poisson_sample(expected, config, draw_index=1)
    assert abs(_noise_correlation(first, second, expected)) < 0.02
