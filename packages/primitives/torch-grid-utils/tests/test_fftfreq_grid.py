import pytest
import torch

from torch_grid_utils import (
    fftfreq_grid,
    fftfreq_shell_centers,
    transform_fftfreq_grid,
)


def test_fftfreq_grid_2d_basic():
    grid = fftfreq_grid(
        image_shape=(4, 4),
        rfft=False,
    )
    assert tuple(grid.shape) == (4, 4, 2)  # (image_h, image_w, freq_hw)
    expected = torch.tensor(
        [
            [[0.0000, 0.0000], [0.0000, 0.2500], [0.0000, -0.5000], [0.0000, -0.2500]],
            [[0.2500, 0.0000], [0.2500, 0.2500], [0.2500, -0.5000], [0.2500, -0.2500]],
            [
                [-0.5000, 0.0000],
                [-0.5000, 0.2500],
                [-0.5000, -0.5000],
                [-0.5000, -0.2500],
            ],
            [
                [-0.2500, 0.0000],
                [-0.2500, 0.2500],
                [-0.2500, -0.5000],
                [-0.2500, -0.2500],
            ],
        ]
    )
    assert torch.allclose(grid, expected)


def test_fftfreq_grid_2d_rfft():
    grid = fftfreq_grid(
        image_shape=(4, 4),
        rfft=True,
    )
    assert tuple(grid.shape) == (4, 3, 2)  # (image_h, image_w_rfft, freq_hw)
    expected = torch.tensor(
        [
            [[0.0000, 0.0000], [0.0000, 0.2500], [0.0000, 0.5000]],
            [[0.2500, 0.0000], [0.2500, 0.2500], [0.2500, 0.5000]],
            [[-0.5000, 0.0000], [-0.5000, 0.2500], [-0.5000, 0.5000]],
            [[-0.2500, 0.0000], [-0.2500, 0.2500], [-0.2500, 0.5000]],
        ]
    )
    assert torch.allclose(grid, expected)


def test_fftfreq_grid_2d_with_spacing():
    grid = fftfreq_grid(
        image_shape=(4, 4),
        rfft=False,
        spacing=2,
    )
    assert tuple(grid.shape) == (4, 4, 2)  # (image_h, image_w, freq_hw)
    expected = torch.tensor(
        [
            [[0.0000, 0.0000], [0.0000, 0.1250], [0.0000, -0.2500], [0.0000, -0.1250]],
            [[0.1250, 0.0000], [0.1250, 0.1250], [0.1250, -0.2500], [0.1250, -0.1250]],
            [
                [-0.2500, 0.0000],
                [-0.2500, 0.1250],
                [-0.2500, -0.2500],
                [-0.2500, -0.1250],
            ],
            [
                [-0.1250, 0.0000],
                [-0.1250, 0.1250],
                [-0.1250, -0.2500],
                [-0.1250, -0.1250],
            ],
        ]
    )
    assert torch.allclose(grid, expected)


def test_fftfreq_grid_2d_with_norm():
    grid = fftfreq_grid(
        image_shape=(4, 4),
        rfft=False,
        norm=True,
    )
    assert tuple(grid.shape) == (4, 4)  # (image_h, image_w, freq_hw)
    expected = torch.tensor(
        [
            [0.0000, 0.2500, 0.5000, 0.2500],
            [0.2500, 0.3536, 0.5590, 0.3536],
            [0.5000, 0.5590, 0.7071, 0.5590],
            [0.2500, 0.3536, 0.5590, 0.3536],
        ]
    )
    assert torch.allclose(grid, expected, atol=1e-4)


def test_fftfreq_grid_2d_with_fftshift():
    grid = fftfreq_grid(
        image_shape=(4, 4),
        rfft=False,
        fftshift=True,
    )
    assert tuple(grid.shape) == (4, 4, 2)
    expected = torch.tensor(
        [
            [
                [-0.5000, -0.5000],
                [-0.5000, -0.2500],
                [-0.5000, 0.0000],
                [-0.5000, 0.2500],
            ],
            [
                [-0.2500, -0.5000],
                [-0.2500, -0.2500],
                [-0.2500, 0.0000],
                [-0.2500, 0.2500],
            ],
            [[0.0000, -0.5000], [0.0000, -0.2500], [0.0000, 0.0000], [0.0000, 0.2500]],
            [[0.2500, -0.5000], [0.2500, -0.2500], [0.2500, 0.0000], [0.2500, 0.2500]],
        ]
    )
    assert torch.allclose(grid, expected, atol=1e-4)


def test_fftfreq_grid_3d_basic():
    grid = fftfreq_grid(
        image_shape=(4, 4, 4),
        rfft=False,
    )
    assert tuple(grid.shape) == (4, 4, 4, 3)  # (image_d, image_h, image_w, freq_dhw)
    expected = torch.tensor(
        [
            [
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.2500],
                    [0.0000, 0.0000, -0.5000],
                    [0.0000, 0.0000, -0.2500],
                ],
                [
                    [0.0000, 0.2500, 0.0000],
                    [0.0000, 0.2500, 0.2500],
                    [0.0000, 0.2500, -0.5000],
                    [0.0000, 0.2500, -0.2500],
                ],
                [
                    [0.0000, -0.5000, 0.0000],
                    [0.0000, -0.5000, 0.2500],
                    [0.0000, -0.5000, -0.5000],
                    [0.0000, -0.5000, -0.2500],
                ],
                [
                    [0.0000, -0.2500, 0.0000],
                    [0.0000, -0.2500, 0.2500],
                    [0.0000, -0.2500, -0.5000],
                    [0.0000, -0.2500, -0.2500],
                ],
            ],
            [
                [
                    [0.2500, 0.0000, 0.0000],
                    [0.2500, 0.0000, 0.2500],
                    [0.2500, 0.0000, -0.5000],
                    [0.2500, 0.0000, -0.2500],
                ],
                [
                    [0.2500, 0.2500, 0.0000],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, -0.5000],
                    [0.2500, 0.2500, -0.2500],
                ],
                [
                    [0.2500, -0.5000, 0.0000],
                    [0.2500, -0.5000, 0.2500],
                    [0.2500, -0.5000, -0.5000],
                    [0.2500, -0.5000, -0.2500],
                ],
                [
                    [0.2500, -0.2500, 0.0000],
                    [0.2500, -0.2500, 0.2500],
                    [0.2500, -0.2500, -0.5000],
                    [0.2500, -0.2500, -0.2500],
                ],
            ],
            [
                [
                    [-0.5000, 0.0000, 0.0000],
                    [-0.5000, 0.0000, 0.2500],
                    [-0.5000, 0.0000, -0.5000],
                    [-0.5000, 0.0000, -0.2500],
                ],
                [
                    [-0.5000, 0.2500, 0.0000],
                    [-0.5000, 0.2500, 0.2500],
                    [-0.5000, 0.2500, -0.5000],
                    [-0.5000, 0.2500, -0.2500],
                ],
                [
                    [-0.5000, -0.5000, 0.0000],
                    [-0.5000, -0.5000, 0.2500],
                    [-0.5000, -0.5000, -0.5000],
                    [-0.5000, -0.5000, -0.2500],
                ],
                [
                    [-0.5000, -0.2500, 0.0000],
                    [-0.5000, -0.2500, 0.2500],
                    [-0.5000, -0.2500, -0.5000],
                    [-0.5000, -0.2500, -0.2500],
                ],
            ],
            [
                [
                    [-0.2500, 0.0000, 0.0000],
                    [-0.2500, 0.0000, 0.2500],
                    [-0.2500, 0.0000, -0.5000],
                    [-0.2500, 0.0000, -0.2500],
                ],
                [
                    [-0.2500, 0.2500, 0.0000],
                    [-0.2500, 0.2500, 0.2500],
                    [-0.2500, 0.2500, -0.5000],
                    [-0.2500, 0.2500, -0.2500],
                ],
                [
                    [-0.2500, -0.5000, 0.0000],
                    [-0.2500, -0.5000, 0.2500],
                    [-0.2500, -0.5000, -0.5000],
                    [-0.2500, -0.5000, -0.2500],
                ],
                [
                    [-0.2500, -0.2500, 0.0000],
                    [-0.2500, -0.2500, 0.2500],
                    [-0.2500, -0.2500, -0.5000],
                    [-0.2500, -0.2500, -0.2500],
                ],
            ],
        ]
    )
    assert torch.allclose(grid, expected)


def test_fftfreq_grid_3d_rfft():
    grid = fftfreq_grid(
        image_shape=(4, 4, 4),
        rfft=True,
    )
    assert tuple(grid.shape) == (
        4,
        4,
        3,
        3,
    )  # (image_d, image_h, image_w_rfft, freq_dhw)
    expected = torch.tensor(
        [
            [
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.2500],
                    [0.0000, 0.0000, 0.5000],
                ],
                [
                    [0.0000, 0.2500, 0.0000],
                    [0.0000, 0.2500, 0.2500],
                    [0.0000, 0.2500, 0.5000],
                ],
                [
                    [0.0000, -0.5000, 0.0000],
                    [0.0000, -0.5000, 0.2500],
                    [0.0000, -0.5000, 0.5000],
                ],
                [
                    [0.0000, -0.2500, 0.0000],
                    [0.0000, -0.2500, 0.2500],
                    [0.0000, -0.2500, 0.5000],
                ],
            ],
            [
                [
                    [0.2500, 0.0000, 0.0000],
                    [0.2500, 0.0000, 0.2500],
                    [0.2500, 0.0000, 0.5000],
                ],
                [
                    [0.2500, 0.2500, 0.0000],
                    [0.2500, 0.2500, 0.2500],
                    [0.2500, 0.2500, 0.5000],
                ],
                [
                    [0.2500, -0.5000, 0.0000],
                    [0.2500, -0.5000, 0.2500],
                    [0.2500, -0.5000, 0.5000],
                ],
                [
                    [0.2500, -0.2500, 0.0000],
                    [0.2500, -0.2500, 0.2500],
                    [0.2500, -0.2500, 0.5000],
                ],
            ],
            [
                [
                    [-0.5000, 0.0000, 0.0000],
                    [-0.5000, 0.0000, 0.2500],
                    [-0.5000, 0.0000, 0.5000],
                ],
                [
                    [-0.5000, 0.2500, 0.0000],
                    [-0.5000, 0.2500, 0.2500],
                    [-0.5000, 0.2500, 0.5000],
                ],
                [
                    [-0.5000, -0.5000, 0.0000],
                    [-0.5000, -0.5000, 0.2500],
                    [-0.5000, -0.5000, 0.5000],
                ],
                [
                    [-0.5000, -0.2500, 0.0000],
                    [-0.5000, -0.2500, 0.2500],
                    [-0.5000, -0.2500, 0.5000],
                ],
            ],
            [
                [
                    [-0.2500, 0.0000, 0.0000],
                    [-0.2500, 0.0000, 0.2500],
                    [-0.2500, 0.0000, 0.5000],
                ],
                [
                    [-0.2500, 0.2500, 0.0000],
                    [-0.2500, 0.2500, 0.2500],
                    [-0.2500, 0.2500, 0.5000],
                ],
                [
                    [-0.2500, -0.5000, 0.0000],
                    [-0.2500, -0.5000, 0.2500],
                    [-0.2500, -0.5000, 0.5000],
                ],
                [
                    [-0.2500, -0.2500, 0.0000],
                    [-0.2500, -0.2500, 0.2500],
                    [-0.2500, -0.2500, 0.5000],
                ],
            ],
        ]
    )
    assert torch.allclose(grid, expected)


def test_fftfreq_grid_3d_with_spacing():
    grid = fftfreq_grid(
        image_shape=(4, 4, 4),
        rfft=False,
        spacing=2,
    )
    assert tuple(grid.shape) == (4, 4, 4, 3)  # (image_h, image_w, freq_hw)
    expected = torch.tensor(
        [
            [
                [
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.1250],
                    [0.0000, 0.0000, -0.2500],
                    [0.0000, 0.0000, -0.1250],
                ],
                [
                    [0.0000, 0.1250, 0.0000],
                    [0.0000, 0.1250, 0.1250],
                    [0.0000, 0.1250, -0.2500],
                    [0.0000, 0.1250, -0.1250],
                ],
                [
                    [0.0000, -0.2500, 0.0000],
                    [0.0000, -0.2500, 0.1250],
                    [0.0000, -0.2500, -0.2500],
                    [0.0000, -0.2500, -0.1250],
                ],
                [
                    [0.0000, -0.1250, 0.0000],
                    [0.0000, -0.1250, 0.1250],
                    [0.0000, -0.1250, -0.2500],
                    [0.0000, -0.1250, -0.1250],
                ],
            ],
            [
                [
                    [0.1250, 0.0000, 0.0000],
                    [0.1250, 0.0000, 0.1250],
                    [0.1250, 0.0000, -0.2500],
                    [0.1250, 0.0000, -0.1250],
                ],
                [
                    [0.1250, 0.1250, 0.0000],
                    [0.1250, 0.1250, 0.1250],
                    [0.1250, 0.1250, -0.2500],
                    [0.1250, 0.1250, -0.1250],
                ],
                [
                    [0.1250, -0.2500, 0.0000],
                    [0.1250, -0.2500, 0.1250],
                    [0.1250, -0.2500, -0.2500],
                    [0.1250, -0.2500, -0.1250],
                ],
                [
                    [0.1250, -0.1250, 0.0000],
                    [0.1250, -0.1250, 0.1250],
                    [0.1250, -0.1250, -0.2500],
                    [0.1250, -0.1250, -0.1250],
                ],
            ],
            [
                [
                    [-0.2500, 0.0000, 0.0000],
                    [-0.2500, 0.0000, 0.1250],
                    [-0.2500, 0.0000, -0.2500],
                    [-0.2500, 0.0000, -0.1250],
                ],
                [
                    [-0.2500, 0.1250, 0.0000],
                    [-0.2500, 0.1250, 0.1250],
                    [-0.2500, 0.1250, -0.2500],
                    [-0.2500, 0.1250, -0.1250],
                ],
                [
                    [-0.2500, -0.2500, 0.0000],
                    [-0.2500, -0.2500, 0.1250],
                    [-0.2500, -0.2500, -0.2500],
                    [-0.2500, -0.2500, -0.1250],
                ],
                [
                    [-0.2500, -0.1250, 0.0000],
                    [-0.2500, -0.1250, 0.1250],
                    [-0.2500, -0.1250, -0.2500],
                    [-0.2500, -0.1250, -0.1250],
                ],
            ],
            [
                [
                    [-0.1250, 0.0000, 0.0000],
                    [-0.1250, 0.0000, 0.1250],
                    [-0.1250, 0.0000, -0.2500],
                    [-0.1250, 0.0000, -0.1250],
                ],
                [
                    [-0.1250, 0.1250, 0.0000],
                    [-0.1250, 0.1250, 0.1250],
                    [-0.1250, 0.1250, -0.2500],
                    [-0.1250, 0.1250, -0.1250],
                ],
                [
                    [-0.1250, -0.2500, 0.0000],
                    [-0.1250, -0.2500, 0.1250],
                    [-0.1250, -0.2500, -0.2500],
                    [-0.1250, -0.2500, -0.1250],
                ],
                [
                    [-0.1250, -0.1250, 0.0000],
                    [-0.1250, -0.1250, 0.1250],
                    [-0.1250, -0.1250, -0.2500],
                    [-0.1250, -0.1250, -0.1250],
                ],
            ],
        ]
    )
    assert torch.allclose(grid, expected)


def test_fftfreq_grid_3d_with_norm():
    grid = fftfreq_grid(
        image_shape=(4, 4, 4),
        rfft=False,
        norm=True,
    )
    assert tuple(grid.shape) == (4, 4, 4)  # (image_h, image_w, freq_hw)
    expected = torch.tensor(
        [
            [
                [0.0000, 0.2500, 0.5000, 0.2500],
                [0.2500, 0.3536, 0.5590, 0.3536],
                [0.5000, 0.5590, 0.7071, 0.5590],
                [0.2500, 0.3536, 0.5590, 0.3536],
            ],
            [
                [0.2500, 0.3536, 0.5590, 0.3536],
                [0.3536, 0.4330, 0.6124, 0.4330],
                [0.5590, 0.6124, 0.7500, 0.6124],
                [0.3536, 0.4330, 0.6124, 0.4330],
            ],
            [
                [0.5000, 0.5590, 0.7071, 0.5590],
                [0.5590, 0.6124, 0.7500, 0.6124],
                [0.7071, 0.7500, 0.8660, 0.7500],
                [0.5590, 0.6124, 0.7500, 0.6124],
            ],
            [
                [0.2500, 0.3536, 0.5590, 0.3536],
                [0.3536, 0.4330, 0.6124, 0.4330],
                [0.5590, 0.6124, 0.7500, 0.6124],
                [0.3536, 0.4330, 0.6124, 0.4330],
            ],
        ]
    )
    assert torch.allclose(grid, expected, atol=1e-4)


def test_fftfreq_grid_3d_with_fftshift():
    grid = fftfreq_grid(
        image_shape=(4, 4, 4),
        rfft=False,
        fftshift=True,
    )
    assert tuple(grid.shape) == (4, 4, 4, 3)
    expected = torch.tensor(
        [
            [
                [
                    [-0.5000, -0.5000, -0.5000],
                    [-0.5000, -0.5000, -0.2500],
                    [-0.5000, -0.5000, 0.0000],
                    [-0.5000, -0.5000, 0.2500],
                ],
                [
                    [-0.5000, -0.2500, -0.5000],
                    [-0.5000, -0.2500, -0.2500],
                    [-0.5000, -0.2500, 0.0000],
                    [-0.5000, -0.2500, 0.2500],
                ],
                [
                    [-0.5000, 0.0000, -0.5000],
                    [-0.5000, 0.0000, -0.2500],
                    [-0.5000, 0.0000, 0.0000],
                    [-0.5000, 0.0000, 0.2500],
                ],
                [
                    [-0.5000, 0.2500, -0.5000],
                    [-0.5000, 0.2500, -0.2500],
                    [-0.5000, 0.2500, 0.0000],
                    [-0.5000, 0.2500, 0.2500],
                ],
            ],
            [
                [
                    [-0.2500, -0.5000, -0.5000],
                    [-0.2500, -0.5000, -0.2500],
                    [-0.2500, -0.5000, 0.0000],
                    [-0.2500, -0.5000, 0.2500],
                ],
                [
                    [-0.2500, -0.2500, -0.5000],
                    [-0.2500, -0.2500, -0.2500],
                    [-0.2500, -0.2500, 0.0000],
                    [-0.2500, -0.2500, 0.2500],
                ],
                [
                    [-0.2500, 0.0000, -0.5000],
                    [-0.2500, 0.0000, -0.2500],
                    [-0.2500, 0.0000, 0.0000],
                    [-0.2500, 0.0000, 0.2500],
                ],
                [
                    [-0.2500, 0.2500, -0.5000],
                    [-0.2500, 0.2500, -0.2500],
                    [-0.2500, 0.2500, 0.0000],
                    [-0.2500, 0.2500, 0.2500],
                ],
            ],
            [
                [
                    [0.0000, -0.5000, -0.5000],
                    [0.0000, -0.5000, -0.2500],
                    [0.0000, -0.5000, 0.0000],
                    [0.0000, -0.5000, 0.2500],
                ],
                [
                    [0.0000, -0.2500, -0.5000],
                    [0.0000, -0.2500, -0.2500],
                    [0.0000, -0.2500, 0.0000],
                    [0.0000, -0.2500, 0.2500],
                ],
                [
                    [0.0000, 0.0000, -0.5000],
                    [0.0000, 0.0000, -0.2500],
                    [0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.2500],
                ],
                [
                    [0.0000, 0.2500, -0.5000],
                    [0.0000, 0.2500, -0.2500],
                    [0.0000, 0.2500, 0.0000],
                    [0.0000, 0.2500, 0.2500],
                ],
            ],
            [
                [
                    [0.2500, -0.5000, -0.5000],
                    [0.2500, -0.5000, -0.2500],
                    [0.2500, -0.5000, 0.0000],
                    [0.2500, -0.5000, 0.2500],
                ],
                [
                    [0.2500, -0.2500, -0.5000],
                    [0.2500, -0.2500, -0.2500],
                    [0.2500, -0.2500, 0.0000],
                    [0.2500, -0.2500, 0.2500],
                ],
                [
                    [0.2500, 0.0000, -0.5000],
                    [0.2500, 0.0000, -0.2500],
                    [0.2500, 0.0000, 0.0000],
                    [0.2500, 0.0000, 0.2500],
                ],
                [
                    [0.2500, 0.2500, -0.5000],
                    [0.2500, 0.2500, -0.2500],
                    [0.2500, 0.2500, 0.0000],
                    [0.2500, 0.2500, 0.2500],
                ],
            ],
        ]
    )
    assert torch.allclose(grid, expected, atol=1e-4)


def test_fftfreq_grid_2d_with_transform_matrix():
    """Test transformation matrix for anisotropic magnification.

    Matrix corresponds to 1.35% anisotropy along two perpendicular axes
    rotated by 66°.
    """
    transform_matrix = torch.tensor([[1.006, 0.005], [0.006, 0.998]])

    # Test without fftshift
    grid_base = fftfreq_grid(
        image_shape=(4, 4),
        rfft=False,
        fftshift=False,
    )
    grid_no_shift = transform_fftfreq_grid(
        frequency_grid=grid_base,
        real_space_matrix=transform_matrix,
    )
    assert tuple(grid_no_shift.shape) == (4, 4, 2)

    # Test with fftshift
    grid_base_shifted = fftfreq_grid(
        image_shape=(4, 4),
        rfft=False,
        fftshift=True,
    )
    grid_shifted = transform_fftfreq_grid(
        frequency_grid=grid_base_shifted,
        real_space_matrix=transform_matrix,
    )
    assert tuple(grid_shifted.shape) == (4, 4, 2)

    # Verify transformation was applied by checking that frequencies are different
    # from the untransformed grid
    grid_untransformed = fftfreq_grid(
        image_shape=(4, 4),
        rfft=False,
        fftshift=False,
    )

    # The transformed grid should be different from untransformed
    assert not torch.allclose(grid_no_shift, grid_untransformed, atol=1e-6)

    # Verify that DC component (0, 0) remains at (0, 0) after transformation
    # In unshifted grid, DC is at position [0, 0]
    dc_freq = grid_no_shift[0, 0]
    # DC should remain at (0, 0) after transformation
    assert torch.allclose(dc_freq, torch.tensor([0.0, 0.0]), atol=1e-5)


def test_fftfreq_grid_2d_with_transform_matrix_fftshifted():
    """Test transformation matrix with fftshift=True."""
    transform_matrix = torch.tensor([[1.006, 0.005], [0.006, 0.998]])

    grid_base = fftfreq_grid(
        image_shape=(4, 4),
        rfft=False,
        fftshift=True,
    )
    grid = transform_fftfreq_grid(
        frequency_grid=grid_base,
        real_space_matrix=transform_matrix,
    )
    assert tuple(grid.shape) == (4, 4, 2)

    # DC component should be at center (2, 2) in fftshifted grid
    dc_freq = grid[2, 2]
    assert torch.allclose(dc_freq, torch.tensor([0.0, 0.0]), atol=1e-5)

    # Verify transformation was applied
    grid_untransformed = fftfreq_grid(
        image_shape=(4, 4),
        rfft=False,
        fftshift=True,
    )
    assert not torch.allclose(grid, grid_untransformed, atol=1e-6)


@pytest.mark.parametrize("n", [1, 2, 7, 20, 21, 96, 97])
def test_fftfreq_shell_centers_match_rfftfreq(n):
    """Shells are 1 / n apart and end at the highest sampled frequency (n // 2) / n."""
    centers = fftfreq_shell_centers(n)
    assert centers.shape == (n // 2 + 1,)
    assert torch.allclose(centers, torch.fft.rfftfreq(n), atol=1e-7)
    assert torch.allclose(centers[-1], torch.fft.fftfreq(n).abs().max())


@pytest.mark.parametrize("n", [20, 21])
def test_fftfreq_shell_centers_end_at_nyquist_only_for_even_sizes(n):
    expected_last = 0.5 if n % 2 == 0 else 0.5 - 1 / (2 * n)
    assert fftfreq_shell_centers(n)[-1].item() == pytest.approx(expected_last)


@pytest.mark.parametrize("w", [20, 21])
@pytest.mark.parametrize("rfft", [True, False])
@pytest.mark.parametrize("fftshift", [True, False])
def test_fftfreq_grid_1d(w, rfft, fftshift):
    """1D grids are (w, 1), matching the (*image_shape, ndim) layout of 2D/3D."""
    grid = fftfreq_grid((w,), rfft=rfft, fftshift=fftshift, spacing=2.0)
    expected = torch.fft.rfftfreq(w, d=2.0) if rfft else torch.fft.fftfreq(w, d=2.0)
    if fftshift and not rfft:
        expected = torch.fft.fftshift(expected)
    assert grid.shape == (expected.shape[0], 1)
    torch.testing.assert_close(grid[..., 0], expected)
    torch.testing.assert_close(
        fftfreq_grid((w,), rfft=rfft, fftshift=fftshift, spacing=2.0, norm=True),
        expected.abs(),
    )
