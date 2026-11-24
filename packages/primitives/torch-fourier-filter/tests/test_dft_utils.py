"""Tests for dft_utils (rotational averaging)."""

import pytest
import torch
from torch_grid_utils import fftfreq_grid

from torch_fourier_filter.dft_utils import (
    rotational_average_dft_2d,
    rotational_average_dft_3d,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rotational_average_dft_2d_cuda():
    """rotational_average_dft_2d runs on cuda:0 with power spectrum input."""
    device = torch.device("cuda:0")
    image_shape = (64, 64)
    dft = torch.fft.fft2(torch.rand(image_shape, device=device))
    power_spectrum = dft.abs().pow(2)

    rotational_average, frequency_bins = rotational_average_dft_2d(
        power_spectrum,
        image_shape,
        rfft=False,
        fftshifted=False,
        return_1d_average=True,
    )

    assert rotational_average.device == device
    assert frequency_bins.device == device
    n_bins = min((d // 2) + 1 for d in image_shape)
    assert rotational_average.shape == (n_bins,)
    assert frequency_bins.shape == (n_bins,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rotational_average_dft_2d_cuda_rfft():
    """rotational_average_dft_2d runs on cuda:0 with rfft power spectrum input."""
    device = torch.device("cuda:0")
    image_shape = (64, 64)
    dft = torch.fft.rfft2(torch.rand(image_shape, device=device))
    power_spectrum = dft.abs().pow(2)

    rotational_average, frequency_bins = rotational_average_dft_2d(
        power_spectrum, image_shape, rfft=True, fftshifted=False, return_1d_average=True
    )

    assert rotational_average.device == device
    assert frequency_bins.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rotational_average_dft_3d_cuda():
    """rotational_average_dft_3d runs on cuda:0 with power spectrum input."""
    device = torch.device("cuda:0")
    image_shape = (32, 64, 64)
    dft = torch.fft.fftn(torch.rand(image_shape, device=device))
    power_spectrum = dft.abs().pow(2)

    rotational_average, frequency_bins = rotational_average_dft_3d(
        power_spectrum,
        image_shape,
        rfft=False,
        fftshifted=False,
        return_1d_average=True,
    )

    assert rotational_average.device == device
    assert frequency_bins.device == device
    _, h, w = image_shape
    n_bins = min((i // 2) + 1 for i in (h, w))
    assert rotational_average.shape == (n_bins,)
    assert frequency_bins.shape == (n_bins,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rotational_average_dft_3d_cuda_return_2d_average():
    """rotational_average_dft_3d on cuda:0, return_1d_average=False (power spectrum)."""
    device = torch.device("cuda:0")
    image_shape = (32, 64, 64)
    dft = torch.fft.fftn(torch.rand(image_shape, device=device))
    power_spectrum = dft.abs().pow(2)

    rotational_average, frequency_bins = rotational_average_dft_3d(
        power_spectrum,
        image_shape,
        rfft=False,
        fftshifted=False,
        return_1d_average=False,
    )

    assert rotational_average.device == device
    assert frequency_bins.device == device
    assert rotational_average.shape == image_shape
    assert frequency_bins.shape == image_shape


@pytest.mark.parametrize("rfft", [True, False])
@pytest.mark.parametrize(
    ("image_shape", "sidelength"),
    [((20, 20), 20), ((21, 21), 21), ((21, 32), 21), ((32, 21), 21)],
)
def test_rotational_average_dft_2d_bins_are_dft_sample_frequencies(
    image_shape, sidelength, rfft
):
    """Shells sit at k / n for the smallest dimension n, ending at (n // 2) / n."""
    dft = fftfreq_grid(image_shape, rfft=rfft, norm=True)
    average, bins = rotational_average_dft_2d(dft, image_shape=image_shape, rfft=rfft)
    assert torch.allclose(bins, torch.fft.rfftfreq(sidelength), atol=1e-6)
    # the mean |k| in each shell is close to the shell center
    assert torch.allclose(average[1:], bins[1:], atol=0.3 / sidelength)


@pytest.mark.parametrize("rfft", [True, False])
@pytest.mark.parametrize("n", [20, 21])
def test_rotational_average_dft_3d_bins_are_dft_sample_frequencies(n, rfft):
    image_shape = (n, n, n)
    dft = fftfreq_grid(image_shape, rfft=rfft, norm=True)
    average, bins = rotational_average_dft_3d(dft, image_shape=image_shape, rfft=rfft)
    assert torch.allclose(bins, torch.fft.rfftfreq(n), atol=1e-6)
    assert torch.allclose(average[1:], bins[1:], atol=0.3 / n)


@pytest.mark.parametrize("rfft", [True, False])
@pytest.mark.parametrize("n", [20, 21])
def test_rotational_average_dft_2d_on_axis_samples_keep_their_shell(n, rfft):
    """Samples on the kx axis at k / n are averaged into shell k, at either parity."""
    image_shape = (n, n)
    dft = fftfreq_grid(image_shape, rfft=rfft, norm=True)
    average_2d, bins_2d = rotational_average_dft_2d(
        dft, image_shape=image_shape, rfft=rfft, return_1d_average=False
    )
    on_axis = torch.arange(n // 2 + 1)
    assert torch.allclose(bins_2d[0, on_axis], torch.fft.rfftfreq(n), atol=1e-6)
    assert torch.allclose(average_2d[0, on_axis], dft[0, on_axis], atol=0.3 / n)
