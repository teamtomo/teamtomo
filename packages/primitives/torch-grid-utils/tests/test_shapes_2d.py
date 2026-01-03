import torch
import pytest

from torch_grid_utils.shapes_2d import circle, rectangle, square, wedge


dims = [5, (5, ) * 2]

@pytest.mark.parametrize("image_shape", dims)
def test_circle(image_shape):
    # Test basic circle creation
    result = circle(radius=2.0, image_shape=image_shape)
    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("image_shape", dims)
def test_rectangle(image_shape):
    # Test basic rectangle creation
    result = rectangle(dimensions=(3.0, 2.0), image_shape=image_shape)
    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("image_shape", dims)
def test_square(image_shape):
    # Test basic square creation
    result = square(sidelength=3.0, image_shape=image_shape)
    assert isinstance(result, torch.Tensor)

@pytest.mark.parametrize("image_shape", dims)
def test_wedge(image_shape):
    # Test basic wedge creation
    result = wedge(aperture=90.0, image_shape=image_shape)
    assert isinstance(result, torch.Tensor)