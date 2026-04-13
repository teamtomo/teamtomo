"""Tests for dft_utils (rotational averaging) with CUDA device."""

import pytest
import torch

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
