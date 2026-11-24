"""Tests for the patch mask and phase-shift helpers."""

import pytest
import torch
from torch_fourier_shift import fourier_shift_dft_2d
from torch_grid_utils import circle

from torch_motion_correction.estimate_motion_optimizer import _phase_shift_rfft_2d
from torch_motion_correction.utils import soft_circle


@pytest.mark.parametrize(
    ("shape", "smoothing_fraction"), [((256, 256), 1.0), ((300, 211), 0.5)]
)
def test_soft_circle_matches_circle(shape, smoothing_fraction):
    radius = min(shape) / 4
    expected = circle(
        radius=radius, image_shape=shape, smoothing_radius=radius * smoothing_fraction
    )
    actual = soft_circle(
        radius=radius, image_shape=shape, smoothing_radius=radius * smoothing_fraction
    )
    # Only the distance transform's pixel rounding of the edge distance differs
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0)
    assert actual[shape[0] // 2, shape[1] // 2] == 1
    assert actual[0, 0] == 0


def test_soft_circle_without_smoothing_is_binary():
    mask = soft_circle(radius=10, image_shape=(32, 32), smoothing_radius=0)
    assert set(mask.unique().tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("shape", [(64, 64), (63, 50)])
def test_phase_shift_matches_fourier_shift(shape):
    torch.manual_seed(0)
    images = torch.randn(3, 4, *shape)
    dft = torch.fft.rfftn(images, dim=(-2, -1))
    shifts = 5 * torch.randn(3, 4, 2)
    expected = fourier_shift_dft_2d(
        dft=dft, image_shape=shape, shifts=shifts, rfft=True, fftshifted=False
    )
    actual = _phase_shift_rfft_2d(dft, shifts, shape)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
