import pytest
import torch
from torch_fourier_filter.mtf import get_mtf_reference_path

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
    with pytest.raises(ValueError, match="exactly one"):
        DqeConfig(apply=True)


def test_dqe_config_raises_when_both_sources_given():
    with pytest.raises(ValueError, match="exactly one"):
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


def test_dqe_config_raises_when_reference_and_starfile_given():
    with pytest.raises(ValueError, match="exactly one"):
        DqeConfig(
            apply=True,
            mtf_reference="k2_300kv",
            starfile_path="mtf.star",
        )


def test_dqe_config_rejects_unknown_mtf_reference():
    with pytest.raises(ValueError, match="Unknown mtf_reference"):
        DqeConfig(apply=True, mtf_reference="not_a_detector")


def test_apply_dqe_mtf_reference_matches_its_starfile():
    """A bundled MTF by name gives the same image as its STAR file by path."""
    name = "K3_standard_300kV_FL2"  # names are case-insensitive
    image = torch.rand(64, 48) + 1.0
    by_name = apply_dqe(image, DqeConfig(apply=True, mtf_reference=name))
    by_path = apply_dqe(
        image,
        DqeConfig(apply=True, starfile_path=str(get_mtf_reference_path(name))),
    )
    torch.testing.assert_close(by_name, by_path)
    assert not torch.allclose(by_name, image)
