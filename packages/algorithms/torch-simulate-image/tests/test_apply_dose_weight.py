import pytest
import torch
from torch_fourier_filter.dose_weight import cumulative_dose_filter_3d

from torch_simulate_image import DoseWeightConfig, apply_dose_weight


def _filtered(image, start, end, pixel_size):
    dose_filter = cumulative_dose_filter_3d(
        volume_shape=tuple(image.shape[-2:]),
        pixel_size=pixel_size,
        start_exposure=start,
        end_exposure=end,
    )
    return torch.fft.irfft2(torch.fft.rfft2(image) * dose_filter, s=image.shape[-2:])


@pytest.mark.parametrize(("start", "end"), [(0.0, 30.0), (12.5, 15.0), (25.0, 25.0)])
def test_apply_dose_weight_matches_cumulative_dose_filter(start, end):
    """The mean exposure filter over [dose_start, dose_end], odd width included."""
    image = torch.rand(32, 33)
    config = DoseWeightConfig(apply=True, dose_start=start, dose_end=end)
    result = apply_dose_weight(image, config, pixel_size=1.2, voltage_kv=300.0)
    torch.testing.assert_close(result, _filtered(image, start, end, 1.2))


def test_apply_dose_weight_preserves_mean():
    """Damage does not attenuate zero frequency, so fluence scaling is unaffected."""
    image = torch.rand(2, 32, 32) + 1.0
    config = DoseWeightConfig(apply=True, dose_start=10.0, dose_end=12.0)
    result = apply_dose_weight(image, config, pixel_size=1.0, voltage_kv=300.0)
    torch.testing.assert_close(result.mean(dim=(-2, -1)), image.mean(dim=(-2, -1)))


def test_apply_dose_weight_scales_exposure_at_200kv():
    """200 kV damages as 300 kV does at 1/0.8 the exposure (RELION's convention)."""
    image = torch.rand(32, 32)
    config = DoseWeightConfig(apply=True, dose_start=4.0, dose_end=8.0)
    result = apply_dose_weight(image, config, pixel_size=1.0, voltage_kv=200.0)
    torch.testing.assert_close(result, _filtered(image, 4.0 / 0.8, 8.0 / 0.8, 1.0))


def test_dose_weight_config_requires_dose_end_when_applied():
    with pytest.raises(ValueError, match="dose_end"):
        DoseWeightConfig(apply=True, dose_start=0.0)


def test_dose_weight_config_without_dose_end_when_not_applied():
    config = DoseWeightConfig()
    image = torch.rand(8, 8)
    assert config.dose_end is None
    assert apply_dose_weight(image, config, pixel_size=1.0, voltage_kv=300.0) is image
