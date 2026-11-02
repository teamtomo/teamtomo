"""Shift estimation must not depend on whether image dimensions are odd or even."""

import pytest
import torch
from torch_fourier_shift import fourier_shift_image_2d

from torch_tiltxcorr import tiltxcorr
from torch_tiltxcorr.utils import (
    calculate_cross_correlation,
    get_shift_from_correlation_image,
)

SHAPES = [(64, 64), (63, 65), (64, 65), (63, 64)]


def _blobs(shape: tuple[int, int], seed: int = 0) -> torch.Tensor:
    """Random image, mildly low-pass filtered so the correlation peak is sharp."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(shape, generator=generator)
    ky = torch.fft.fftfreq(shape[0])[:, None]
    kx = torch.fft.rfftfreq(shape[1])[None, :]
    lowpass = torch.exp(-(ky**2 + kx**2) / (2 * 0.2**2))
    return torch.fft.irfft2(torch.fft.rfft2(noise) * lowpass, s=shape)


@pytest.mark.parametrize("shape", SHAPES)
def test_identical_images_give_zero_shift(shape):
    image = _blobs(shape)
    correlation = calculate_cross_correlation(image, image)
    shift = get_shift_from_correlation_image(correlation)
    # sub-pixel peaks are found on a 1/64 px grid
    torch.testing.assert_close(shift, torch.zeros(2), atol=0.05, rtol=0)


@pytest.mark.parametrize("shape", SHAPES)
def test_known_shift_is_recovered(shape):
    """The returned shift, applied to b, aligns it with a."""
    a = _blobs(shape)
    b = torch.roll(a, shifts=(3, -2), dims=(-2, -1))
    shift = get_shift_from_correlation_image(calculate_cross_correlation(a, b))
    torch.testing.assert_close(shift, torch.tensor([-3.0, 2.0]), atol=0.05, rtol=0)


@pytest.mark.parametrize("shape", [(128, 128), (127, 129)])
def test_tiltxcorr_does_not_accumulate_drift_for_odd_sizes(shape):
    """Per-pair errors add up along the tilt chain, so even 1 px per pair matters."""
    n_tilts = 9
    angles = torch.linspace(-4.0, 4.0, n_tilts)  # near-identical stretch
    generator = torch.Generator().manual_seed(1)
    applied = torch.rand((n_tilts, 2), generator=generator) * 6 - 3
    applied[n_tilts // 2] = 0  # the reference (lowest tilt) image is not shifted
    image = _blobs(shape)
    tilt_series = fourier_shift_image_2d(image.expand(n_tilts, *shape), applied)
    shifts = tiltxcorr(tilt_series, angles, tilt_axis_angle=0.0)
    torch.testing.assert_close(shifts, -applied, atol=0.2, rtol=0)
