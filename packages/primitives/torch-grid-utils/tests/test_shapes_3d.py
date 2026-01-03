import torch
import pytest

from torch_grid_utils.shapes_3d import sphere, cuboid, cube, cone


dims = [5, (5, ) * 3]

@pytest.mark.parametrize("image_shape", dims)
def test_sphere(image_shape):
    # Test basic sphere creation
    result = sphere(radius=2.0, image_shape=image_shape)
    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("image_shape", dims)
def test_cuboid(image_shape):
    # Test basic cuboid creation
    result = cuboid(dimensions=(3.0, 2.0, 2.0), image_shape=image_shape)
    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("image_shape", dims)
def test_cube(image_shape):
    # Test basic cube creation
    result = cube(sidelength=3.0, image_shape=image_shape)
    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("image_shape", dims)
def test_cone(image_shape):
    # Test basic cone creation
    result = cone(aperture=90., image_shape=image_shape)
    assert isinstance(result, torch.Tensor)