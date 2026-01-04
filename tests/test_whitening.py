import torch

from torch_fourier_filter.whitening import (
    power_spectral_density,
    real_space_shape_from_dft_shape,
    whitening_filter,
)


def test_real_space_shape_from_dft_shape():
    """Basic tests for real_space_shape_from_dft_shape."""
    # Test 2D shape
    dft_shape = (64, 64)
    real_space_shape = real_space_shape_from_dft_shape(dft_shape, rfft=False)
    assert real_space_shape == (64, 64), "Real space shape incorrect for 2D input"

    # Test 2D shape with rfft
    dft_shape = (64, 33)
    real_space_shape = real_space_shape_from_dft_shape(dft_shape, rfft=True)
    assert real_space_shape == (
        64,
        64,
    ), "Real space shape incorrect for 2D input with rfft"

    # Test 3D shape
    dft_shape = (32, 64, 64)
    real_space_shape = real_space_shape_from_dft_shape(dft_shape, rfft=False)
    assert real_space_shape == (32, 64, 64), "Real space shape incorrect for 3D input"

    # Test 3D shape with rfft
    dft_shape = (32, 64, 33)
    real_space_shape = real_space_shape_from_dft_shape(dft_shape, rfft=True)
    assert real_space_shape == (
        32,
        64,
        64,
    ), "Real space shape incorrect for 3D input with rfft"


def test_power_spectral_density():
    """Basic tests for outputs of power_spectral_density."""
    img = torch.rand((64, 64))
    img_dft = torch.fft.rfftn(img)

    psd_1d = power_spectral_density(img_dft, rfft=True, fftshift=False)

    assert isinstance(psd_1d, torch.Tensor), "Output should be a tensor"
    assert psd_1d.ndim == 1, "Output should be 1D"


def test_whitening_filter():
    """Basic tests for outputs of whitening_filter."""
    img = torch.rand((64, 64))
    img_dft = torch.fft.rfftn(img)

    # Test for same output shape, etc.
    wf = whitening_filter(img_dft, rfft=False, fftshift=False)
    assert isinstance(wf, torch.Tensor), "Output should be a tensor"
    assert wf.shape == img_dft.shape, "Output shape should be the same as input"

    # Test for different output shape
    wf = whitening_filter(img_dft, rfft=False, fftshift=False, output_shape=(32, 32))
    assert isinstance(wf, torch.Tensor), "Output should be a tensor"
    assert wf.shape == (32, 32), "Output shape should be the same as input"

    # Test for different output shape
    wf = whitening_filter(img_dft, rfft=True, fftshift=False, output_shape=(32, 32))
    assert isinstance(wf, torch.Tensor), "Output should be a tensor"
    assert wf.shape == (32, 17), "Output shape should be the same as input"
