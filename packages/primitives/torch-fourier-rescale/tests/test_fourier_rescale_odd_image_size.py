"""The image centre [n // 2] stays at the centre for odd and even sizes."""

import pytest
import torch

from torch_fourier_rescale import fourier_rescale_2d, fourier_rescale_3d

SIZES = [(64, 32), (65, 33), (64, 33), (65, 32)]  # (input, output) per dimension


def _peak(image: torch.Tensor) -> list[int]:
    return [int(i) for i in torch.unravel_index(torch.argmax(image), image.shape)]


@pytest.mark.parametrize(("n_in", "n_out"), SIZES)
def test_centre_stays_at_centre_2d(n_in, n_out):
    image = torch.zeros(n_in, n_in + 2)
    image[n_in // 2, (n_in + 2) // 2] = 1.0
    rescaled, _ = fourier_rescale_2d(image, target_shape=(n_out, n_out + 1))
    assert _peak(rescaled) == [n_out // 2, (n_out + 1) // 2]


@pytest.mark.parametrize(("n_in", "n_out"), SIZES)
def test_centre_stays_at_centre_3d(n_in, n_out):
    volume = torch.zeros(n_in, n_in, n_in)
    volume[n_in // 2, n_in // 2, n_in // 2] = 1.0
    rescaled, _ = fourier_rescale_3d(volume, target_shape=(n_out, n_out, n_out))
    assert _peak(rescaled) == [n_out // 2] * 3


def test_centre_stays_at_centre_by_spacing():
    """A K2-sized image binned to 10 A goes from even to odd sizes."""
    image = torch.zeros(3710, 3838)
    image[3710 // 2, 3838 // 2] = 1.0
    rescaled, _ = fourier_rescale_2d(image, source_spacing=1.7005, target_spacing=10.0)
    h, w = rescaled.shape
    assert (h % 2, w % 2) == (1, 1)
    assert _peak(rescaled) == [h // 2, w // 2]
