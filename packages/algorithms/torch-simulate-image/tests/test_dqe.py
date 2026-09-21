import pytest
import torch

from torch_simulate_image import DqeConfig, apply_dqe


def _identity_mtf_config() -> DqeConfig:
    return DqeConfig(
        apply=True,
        mtf_frequencies=torch.tensor([0.0, 0.5]),
        mtf_amplitudes=torch.tensor([1.0, 1.0]),
    )


def test_apply_dqe_preserves_mean():
    image = torch.rand(32, 32) + 1.0
    config = _identity_mtf_config()
    result = apply_dqe(image, config)
    assert result.shape == image.shape
    assert torch.allclose(result.mean(), image.mean(), rtol=1e-4)


def test_apply_dqe_preserves_dtype():
    image = (torch.rand(16, 16) + 1.0).to(torch.float64)
    config = _identity_mtf_config()
    result = apply_dqe(image, config)
    assert result.dtype == torch.float64


def test_apply_dqe_disabled():
    image = torch.ones(8, 8)
    config = DqeConfig(apply=False)
    result = apply_dqe(image, config)
    assert torch.allclose(result, image)


def test_dqe_defaults_to_disabled():
    assert DqeConfig().apply is False


def test_dqe_config_raises_when_neither_source_given():
    with pytest.raises(ValueError, match="starfile_path"):
        DqeConfig(apply=True)


def test_dqe_config_raises_when_both_sources_given():
    with pytest.raises(ValueError, match="starfile_path"):
        DqeConfig(
            apply=True,
            mtf_frequencies=torch.tensor([0.0, 0.5]),
            mtf_amplitudes=torch.tensor([1.0, 1.0]),
            starfile_path="mtf.star",
        )


def test_dqe_config_raises_when_only_one_tensor_set():
    with pytest.raises(ValueError, match="mtf_frequencies and mtf_amplitudes"):
        DqeConfig(apply=True, mtf_frequencies=torch.tensor([0.0, 0.5]))


def test_dqe_config_raises_when_partial_tensor_set_alongside_starfile():
    with pytest.raises(ValueError, match="mtf_frequencies and mtf_amplitudes"):
        DqeConfig(
            apply=True,
            mtf_frequencies=torch.tensor([0.0, 0.5]),
            starfile_path="mtf.star",
        )
