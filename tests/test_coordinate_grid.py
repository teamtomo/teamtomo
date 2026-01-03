import torch
import pytest

from torch_grid_utils import coordinate_grid, image_center


def test_coordinate_grid_2d_basic():
    grid = coordinate_grid(image_shape=(4, 4))
    expected = torch.tensor([[[0., 0.],
                              [0., 1.],
                              [0., 2.],
                              [0., 3.]],

                             [[1., 0.],
                              [1., 1.],
                              [1., 2.],
                              [1., 3.]],

                             [[2., 0.],
                              [2., 1.],
                              [2., 2.],
                              [2., 3.]],

                             [[3., 0.],
                              [3., 1.],
                              [3., 2.],
                              [3., 3.]]])
    assert torch.allclose(grid, expected)


def test_coordinate_grid_2d_with_center():
    grid = coordinate_grid(image_shape=(4, 4), center=(2, 2))
    assert torch.allclose(grid[2, 2], torch.tensor([0, 0]).float())
    expected = torch.tensor([[[-2., -2.],
                              [-2., -1.],
                              [-2., 0.],
                              [-2., 1.]],

                             [[-1., -2.],
                              [-1., -1.],
                              [-1., 0.],
                              [-1., 1.]],

                             [[0., -2.],
                              [0., -1.],
                              [0., 0.],
                              [0., 1.]],

                             [[1., -2.],
                              [1., -1.],
                              [1., 0.],
                              [1., 1.]]])
    assert torch.allclose(grid, expected)


def test_coordinate_grid_2d_with_norm():
    grid = coordinate_grid(image_shape=(4, 4), center=(2, 2), norm=True)
    expected = torch.tensor([[2.8284, 2.2361, 2.0000, 2.2361],
                             [2.2361, 1.4142, 1.0000, 1.4142],
                             [2.0000, 1.0000, 0.0000, 1.0000],
                             [2.2361, 1.4142, 1.0000, 1.4142]])
    assert torch.allclose(grid, expected, atol=1e-4)


def test_coordinate_grid_3d_basic():
    grid = coordinate_grid(image_shape=(4, 4, 4))
    expected = torch.tensor([[[[0., 0., 0.],
                               [0., 0., 1.],
                               [0., 0., 2.],
                               [0., 0., 3.]],

                              [[0., 1., 0.],
                               [0., 1., 1.],
                               [0., 1., 2.],
                               [0., 1., 3.]],

                              [[0., 2., 0.],
                               [0., 2., 1.],
                               [0., 2., 2.],
                               [0., 2., 3.]],

                              [[0., 3., 0.],
                               [0., 3., 1.],
                               [0., 3., 2.],
                               [0., 3., 3.]]],

                             [[[1., 0., 0.],
                               [1., 0., 1.],
                               [1., 0., 2.],
                               [1., 0., 3.]],

                              [[1., 1., 0.],
                               [1., 1., 1.],
                               [1., 1., 2.],
                               [1., 1., 3.]],

                              [[1., 2., 0.],
                               [1., 2., 1.],
                               [1., 2., 2.],
                               [1., 2., 3.]],

                              [[1., 3., 0.],
                               [1., 3., 1.],
                               [1., 3., 2.],
                               [1., 3., 3.]]],

                             [[[2., 0., 0.],
                               [2., 0., 1.],
                               [2., 0., 2.],
                               [2., 0., 3.]],

                              [[2., 1., 0.],
                               [2., 1., 1.],
                               [2., 1., 2.],
                               [2., 1., 3.]],

                              [[2., 2., 0.],
                               [2., 2., 1.],
                               [2., 2., 2.],
                               [2., 2., 3.]],

                              [[2., 3., 0.],
                               [2., 3., 1.],
                               [2., 3., 2.],
                               [2., 3., 3.]]],

                             [[[3., 0., 0.],
                               [3., 0., 1.],
                               [3., 0., 2.],
                               [3., 0., 3.]],

                              [[3., 1., 0.],
                               [3., 1., 1.],
                               [3., 1., 2.],
                               [3., 1., 3.]],

                              [[3., 2., 0.],
                               [3., 2., 1.],
                               [3., 2., 2.],
                               [3., 2., 3.]],

                              [[3., 3., 0.],
                               [3., 3., 1.],
                               [3., 3., 2.],
                               [3., 3., 3.]]]])
    assert torch.allclose(grid, expected)


def test_coordinate_grid_3d_with_center():
    grid = coordinate_grid(image_shape=(4, 4, 4), center=(2, 2, 2))
    assert torch.allclose(grid[2, 2, 2], torch.tensor([0, 0, 0]).float())
    expected = torch.tensor([[[[-2., -2., -2.],
                               [-2., -2., -1.],
                               [-2., -2., 0.],
                               [-2., -2., 1.]],

                              [[-2., -1., -2.],
                               [-2., -1., -1.],
                               [-2., -1., 0.],
                               [-2., -1., 1.]],

                              [[-2., 0., -2.],
                               [-2., 0., -1.],
                               [-2., 0., 0.],
                               [-2., 0., 1.]],

                              [[-2., 1., -2.],
                               [-2., 1., -1.],
                               [-2., 1., 0.],
                               [-2., 1., 1.]]],

                             [[[-1., -2., -2.],
                               [-1., -2., -1.],
                               [-1., -2., 0.],
                               [-1., -2., 1.]],

                              [[-1., -1., -2.],
                               [-1., -1., -1.],
                               [-1., -1., 0.],
                               [-1., -1., 1.]],

                              [[-1., 0., -2.],
                               [-1., 0., -1.],
                               [-1., 0., 0.],
                               [-1., 0., 1.]],

                              [[-1., 1., -2.],
                               [-1., 1., -1.],
                               [-1., 1., 0.],
                               [-1., 1., 1.]]],

                             [[[0., -2., -2.],
                               [0., -2., -1.],
                               [0., -2., 0.],
                               [0., -2., 1.]],

                              [[0., -1., -2.],
                               [0., -1., -1.],
                               [0., -1., 0.],
                               [0., -1., 1.]],

                              [[0., 0., -2.],
                               [0., 0., -1.],
                               [0., 0., 0.],
                               [0., 0., 1.]],

                              [[0., 1., -2.],
                               [0., 1., -1.],
                               [0., 1., 0.],
                               [0., 1., 1.]]],

                             [[[1., -2., -2.],
                               [1., -2., -1.],
                               [1., -2., 0.],
                               [1., -2., 1.]],

                              [[1., -1., -2.],
                               [1., -1., -1.],
                               [1., -1., 0.],
                               [1., -1., 1.]],

                              [[1., 0., -2.],
                               [1., 0., -1.],
                               [1., 0., 0.],
                               [1., 0., 1.]],

                              [[1., 1., -2.],
                               [1., 1., -1.],
                               [1., 1., 0.],
                               [1., 1., 1.]]]])
    assert torch.allclose(grid, expected)


def test_coordinate_grid_3d_with_norm():
    grid = coordinate_grid(image_shape=(4, 4, 4), center=(2, 2, 2), norm=True)
    expected = torch.tensor([[[3.4641, 3.0000, 2.8284, 3.0000],
                              [3.0000, 2.4495, 2.2361, 2.4495],
                              [2.8284, 2.2361, 2.0000, 2.2361],
                              [3.0000, 2.4495, 2.2361, 2.4495]],

                             [[3.0000, 2.4495, 2.2361, 2.4495],
                              [2.4495, 1.7321, 1.4142, 1.7321],
                              [2.2361, 1.4142, 1.0000, 1.4142],
                              [2.4495, 1.7321, 1.4142, 1.7321]],

                             [[2.8284, 2.2361, 2.0000, 2.2361],
                              [2.2361, 1.4142, 1.0000, 1.4142],
                              [2.0000, 1.0000, 0.0000, 1.0000],
                              [2.2361, 1.4142, 1.0000, 1.4142]],

                             [[3.0000, 2.4495, 2.2361, 2.4495],
                              [2.4495, 1.7321, 1.4142, 1.7321],
                              [2.2361, 1.4142, 1.0000, 1.4142],
                              [2.4495, 1.7321, 1.4142, 1.7321]]])
    assert torch.allclose(grid, expected, atol=1e-4)


@pytest.mark.parametrize(
    "image_shape,expected_center",
    [
        # 2D cases
        ((28, 14), torch.tensor([14, 7])),  # 2D even dimensions
        ((29, 15), torch.tensor([14, 7])),  # 2D odd dimensions
        ((30, 16), torch.tensor([15, 8])),  # 2D even dimensions
        ((31, 17), torch.tensor([15, 8])),  # 2D odd dimensions
        # 3D cases
        ((28, 14, 20), torch.tensor([14, 7, 10])),  # 3D even dimensions
        ((29, 15, 21), torch.tensor([14, 7, 10])),  # 3D odd dimensions
        ((30, 16, 22), torch.tensor([15, 8, 11])),  # 3D even dimensions
        ((31, 17, 23), torch.tensor([15, 8, 11])),  # 3D odd dimensions
    ]
)
def test_image_center(image_shape, expected_center):
    """image center should be at position of DC component in fourier transform"""
    center = image_center(image_shape=image_shape)
    assert torch.allclose(center, expected_center)
    
    # Verify that the center corresponds to the DC component position
    for i, dim_size in enumerate(image_shape):
        freq = torch.fft.fftshift(torch.fft.fftfreq(dim_size))
        idx_dc = torch.where(freq == 0)[0]
        assert center[i] == idx_dc[0]
