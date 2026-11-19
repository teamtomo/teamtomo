"""Tests for projection-based alignment."""

import pytest
import torch

from torch_fit_in_map import ProjectionAlignmentConfig, projection_align
from torch_fit_in_map._projection import _cross_correlate_2d, _find_2d_peak_shift


def test_projection_align_identity():
    """Aligning a cubic volume with itself should give score > 0."""
    vol = torch.rand(24, 24, 24)
    cfg = ProjectionAlignmentConfig(angular_step_degrees=30.0)
    result = projection_align(vol, vol, config=cfg)
    assert result.score > 0.0
    assert result.rotation_matrix.shape == (3, 3)
    assert result.translation_pixels.shape == (3,)
    # Depth shift should be zero for projection-based method
    assert result.translation_pixels[0].item() == 0.0


@pytest.mark.parametrize("d", [32, 33])
@pytest.mark.parametrize("shift", [(3, -8), (-14, 14), (15, -15)])
def test_cross_correlate_2d_recovers_shift_for_odd_and_even_sizes(d, shift):
    """CC map keeps the input size and its peak gives the exact shift."""
    reference = torch.rand(d, d)
    mobile = torch.roll(reference, shifts=shift, dims=(-2, -1))
    cc = _cross_correlate_2d(reference, mobile)
    assert cc.shape == (d, d)
    recovered, _ = _find_2d_peak_shift(cc)
    # mobile is reference moved by +shift, the CC peak sits at -shift
    assert recovered.tolist() == [-float(shift[0]), -float(shift[1])]
