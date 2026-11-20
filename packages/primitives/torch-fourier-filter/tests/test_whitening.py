import warnings

import pytest
import torch

from torch_fourier_filter.whitening import (
    power_spectral_density,
    whitening_filter,
)


def test_power_spectral_density():
    """Basic tests for outputs of power_spectral_density."""
    img = torch.rand((64, 64))
    img_dft = torch.fft.rfftn(img)

    psd_1d = power_spectral_density(
        img_dft, rfft=True, fftshift=False, image_shape=(64, 64)
    )

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
    wf = whitening_filter(
        img_dft, rfft=True, fftshift=False, output_shape=(32, 32), image_shape=(64, 64)
    )
    assert isinstance(wf, torch.Tensor), "Output should be a tensor"
    assert wf.shape == (32, 17), "Output shape should be the same as input"


@pytest.mark.parametrize("shape", [(12, 16), (12, 17), (6, 12, 17)])
def test_whitening_uses_input_width(shape):
    """Full and half output layouts must sample the same physical frequencies."""
    image = torch.randn(shape)
    half = torch.fft.rfftn(image)
    dim = tuple(range(image.ndim))
    width = shape[-1]
    full_filter = whitening_filter(half, dim=dim, image_shape=shape, output_rfft=False)
    half_filter = whitening_filter(half, dim=dim, image_shape=shape)
    assert full_filter.shape == image.shape
    torch.testing.assert_close(half_filter, full_filter[..., : half.shape[-1]])
    with pytest.raises(ValueError, match="does not match"):
        power_spectral_density(half, dim=dim, image_shape=(*shape[:-1], width + 2))


@pytest.mark.parametrize("shape", [(12, 16), (12, 17)])
def test_whitening_rfft_without_image_shape_raises(shape):
    """An rfft width is ambiguous (16 and 17 both give 9 columns): raise."""
    half = torch.fft.rfftn(torch.randn(shape))
    with pytest.raises(ValueError, match="ambiguous"):
        power_spectral_density(half)
    with pytest.raises(ValueError, match="ambiguous"):
        whitening_filter(half)


def test_whitening_full_dft_needs_no_image_shape():
    """A full DFT has no width ambiguity, so no warning is raised."""
    dft = torch.fft.fftn(torch.randn(12, 17))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wf = whitening_filter(dft, rfft=False)
    assert wf.shape == dft.shape
