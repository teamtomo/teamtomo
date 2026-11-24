import pytest
import torch

from torch_grid_utils.shapes_2d import circle, rectangle, soft_circle, square, wedge

dims = [5, (5,) * 2]


@pytest.mark.parametrize("image_shape", dims)
def test_circle(image_shape):
    # Test basic circle creation
    result = circle(radius=2.0, image_shape=image_shape)
    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("image_shape", dims)
def test_soft_circle(image_shape):
    result = soft_circle(radius=2.0, image_shape=image_shape, smoothing_radius=1.0)
    assert result.shape == (5, 5)
    assert result.dtype == torch.float32


@pytest.mark.parametrize(
    ("shape", "smoothing_fraction"), [((256, 256), 1.0), ((300, 211), 0.5)]
)
def test_soft_circle_matches_circle(shape, smoothing_fraction):
    radius = min(shape) / 4
    expected = circle(
        radius=radius, image_shape=shape, smoothing_radius=radius * smoothing_fraction
    )
    actual = soft_circle(
        radius=radius, image_shape=shape, smoothing_radius=radius * smoothing_fraction
    )
    # circle measures the edge distance to the nearest pixel inside the circle, so it
    # differs only by that pixel rounding
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0)
    assert actual[shape[0] // 2, shape[1] // 2] == 1
    assert actual[0, 0] == 0


def test_soft_circle_edge_values():
    radius, smoothing_radius = 10.0, 4.0
    mask = soft_circle(
        radius=radius,
        image_shape=(64, 64),
        center=(20.0, 30.0),
        smoothing_radius=smoothing_radius,
    )
    # Distances 10, 12 and 14 from the centre along +x
    assert mask[20, 40] == pytest.approx(1.0)
    assert mask[20, 42] == pytest.approx(torch.cos(torch.tensor(torch.pi / 4)).item())
    assert mask[20, 44] == pytest.approx(0.0, abs=1e-6)
    assert mask[20, 45] == 0
    assert mask[20, 30] == 1


def test_soft_circle_without_smoothing_is_binary():
    mask = soft_circle(radius=10, image_shape=(32, 32), smoothing_radius=0)
    assert set(mask.unique().tolist()) == {0.0, 1.0}
    torch.testing.assert_close(mask, circle(radius=10, image_shape=(32, 32)))


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
