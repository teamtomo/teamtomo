import torch

from pytest_utils import device_test
from torch_grid_utils.soft_edge import (
    _add_soft_edge_single_binary_image,
    add_soft_edge_2d,
)


@device_test
def test_add_soft_edge_single_binary_image():
    dim_length = 5
    smoothing_radius = 4
    image = torch.zeros(size=(dim_length, 1))
    image[0, 0] = 1
    smoothed = _add_soft_edge_single_binary_image(
        image, smoothing_radius=smoothing_radius
    )
    # cosine falloff, 1 to zero over smooothing radius
    expected = torch.cos((torch.pi / 2) * torch.arange(5) / smoothing_radius)
    expected = expected.view((dim_length, 1))
    assert torch.allclose(smoothed, expected)


@device_test
def test_add_soft_edge_2d():
    # single image
    image = torch.zeros(size=(5, 5))
    image[0, 0] = 1
    smoothing_radius = 5
    result = add_soft_edge_2d(image, smoothing_radius=smoothing_radius)
    assert result.shape == image.shape

    # nD batch
    images = torch.zeros(size=(2, 2, 10, 5, 5))
    smoothing_radius = 5
    images[..., 0, 0] = 1
    results = add_soft_edge_2d(images, smoothing_radius=smoothing_radius)
    assert results.shape == images.shape
    assert torch.allclose(results[0], results[1])
