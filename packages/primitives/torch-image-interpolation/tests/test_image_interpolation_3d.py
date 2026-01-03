import numpy as np
import torch
import einops
import pytest

from torch_image_interpolation import sample_image_3d, insert_into_image_3d


def test_sample_image_3d():
    # basic sanity check only
    image = torch.rand((28, 28, 28))

    # make an arbitrary stack (..., 3) of 3d coords
    arbitrary_shape = (6, 7, 8)
    coords = torch.tensor(np.random.randint(low=0, high=27, size=(*arbitrary_shape, 3)))

    # sample
    samples = sample_image_3d(image=image, coordinates=coords)
    assert samples.shape == arbitrary_shape


def test_sample_image_3d_complex_input():
    # basic sanity check only
    image = torch.complex(real=torch.rand((28, 28, 28)), imag=torch.rand(28, 28, 28))

    # make an arbitrary stack (..., 3) of 3d coords
    arbitrary_shape = (6, 7, 8)
    coords = torch.tensor(np.random.randint(low=0, high=27, size=(*arbitrary_shape, 3)))

    # sample
    samples = sample_image_3d(image=image, coordinates=coords)
    assert samples.shape == arbitrary_shape


def test_sample_image_3d_multichannel_input():
    n_channels = 3

    # basic sanity check only
    image = torch.rand((n_channels, 28, 28, 28))

    # make an arbitrary stack (..., 3) of 3d coords
    arbitrary_shape = (6, 7, 8)
    coords = torch.tensor(np.random.randint(low=0, high=27, size=(*arbitrary_shape, 3)))

    # sample
    samples = sample_image_3d(image=image, coordinates=coords)
    assert samples.shape == (*arbitrary_shape, n_channels)


def test_sample_image_3d_multichannel_complex_input():
    n_channels = 3

    # basic sanity check only
    image = torch.complex(real=torch.rand((28, 28, 28)), imag=torch.rand(28, 28, 28))
    image = einops.repeat(image, 'd h w -> c d h w', c=n_channels)

    # make an arbitrary stack (..., 3) of 3d coords
    arbitrary_shape = (6, 7, 8)
    coords = torch.tensor(np.random.randint(low=0, high=27, size=(*arbitrary_shape, 3)))

    # sample
    samples = sample_image_3d(image=image, coordinates=coords)
    assert samples.shape == (*arbitrary_shape, n_channels)


def test_insert_into_image_3d():
    image = torch.zeros((28, 28, 28)).float()

    # single value
    value = torch.tensor([5]).float()
    coordinate = torch.tensor([10.5, 14.5, 18.5]).view((1, 3))

    # sample
    image, weights = insert_into_image_3d(value, coordinates=coordinate, image=image, interpolation="trilinear")

    # check value (5) is evenly split over 4 nearest pixels
    expected = einops.repeat(torch.tensor([5 / 8]), '1 -> 2 2 2')
    assert torch.allclose(image[10:12, 14:16, 18:20], expected)

    # check for zeros elsewhere
    assert torch.allclose(image[:10, :14, :18], torch.zeros_like(image[:10, :14, :18]))


def test_insert_into_image_3d_multiple():
    image = torch.zeros((28, 28, 28)).float()

    # multiple values
    values = torch.ones(size=(6, 7, 8)).float()
    coordinates = torch.tensor(np.random.randint(low=0, high=27, size=(6, 7, 8, 3)))

    # sample
    image, weights = insert_into_image_3d(values, coordinates=coordinates, image=image, interpolation="trilinear")

    # check for nonzero value at one point
    sample_point = coordinates[0, 0, 0]
    z, y, x = sample_point
    assert image[z, y, x] > 0


def test_insert_into_image_nearest_interp_3d():
    image = torch.zeros((28, 28, 28)).float()

    # single value
    value = torch.tensor([5]).float()
    coordinate = torch.tensor([10.7, 14.3, 18.9]).view((1, 3))

    # sample
    image, weights = insert_into_image_3d(value, coordinates=coordinate, image=image, interpolation='nearest')

    # check value (5) is added at nearest pixel
    expected = torch.zeros((28, 28, 28)).float()
    expected[11, 14, 19] = 5
    assert torch.allclose(image, expected)


def test_insert_multiple_values_into_multichannel_image_3d_trilinear():
    n_channels = 3
    image = torch.zeros((n_channels, 28, 28, 28)).float()

    # multiple values
    arbitrary_shape = (6, 7, 8)
    values = torch.ones(size=(*arbitrary_shape, n_channels)).float()
    coordinates = torch.tensor(np.random.randint(low=0, high=27, size=(*arbitrary_shape, 3)))

    # sample
    image, weights = insert_into_image_3d(values, coordinates=coordinates, image=image, interpolation="trilinear")

    # check for nonzero value at one point
    sample_point = coordinates[0, 0, 0]
    z, y, x = sample_point
    assert image[0, z, y, x] > 0

    # check output shapes
    assert image.shape == (n_channels, 28, 28, 28)
    assert weights.shape == (28, 28, 28)


def test_insert_multiple_values_into_multichannel_image_2d_nearest():
    n_channels = 3
    image = torch.zeros((n_channels, 28, 28, 28)).float()

    # multiple values
    arbitrary_shape = (6, 7, 8)
    values = torch.ones(size=(*arbitrary_shape, n_channels)).float()
    coordinates = torch.tensor(np.random.randint(low=0, high=27, size=(*arbitrary_shape, 3)))

    # sample
    image, weights = insert_into_image_3d(values, coordinates=coordinates, image=image, interpolation="nearest")

    # check for nonzero value at one point
    sample_point = coordinates[0, 0, 0]
    z, y, x = sample_point
    assert image[0, z, y, x] > 0

    # check output shapes
    assert image.shape == (n_channels, 28, 28, 28)
    assert weights.shape == (28, 28, 28)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.complex64, torch.complex128]
)
def test_insert_into_image_2d_type_consistency(dtype):
    image = torch.rand((4, 4, 4), dtype=dtype)
    coords = torch.tensor(np.random.uniform(low=0, high=3, size=(3, 4, 5, 3)))
    values = torch.rand(size=(3, 4, 5), dtype=dtype)
    weights = torch.zeros_like(image, dtype=torch.float64)

    for mode in ['bilinear', 'nearest']:
        image, weights = insert_into_image_3d(
            values,
            image=image,
            weights=weights,
            coordinates=coords,
            interpolation=mode,
        )
        assert image.dtype == dtype
        assert weights.dtype == torch.float64


def test_insert_into_image_3d_type_error():
    image = torch.rand((4, 4, 4), dtype=torch.complex64)
    coords = torch.tensor(np.random.uniform(low=0, high=3, size=(3, 4, 5, 3)))
    values = torch.rand(size=(3, 4, 5), dtype=torch.complex128)
    with pytest.raises(ValueError):
        insert_into_image_3d(
            values,
            image=image,
            coordinates=coords,
        )
