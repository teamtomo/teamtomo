import torch

from torch_fourier_filter.mtf import (
    make_mtf_grid,
)


def test_make_mtf_grid():
    # Test parameters
    image_shape_2d = (64, 64)
    image_shape_3d = (32, 64, 64)
    mtf_frequencies = torch.linspace(0, 0.5, 10)
    mtf_amplitudes = torch.linspace(1, 0, 10)

    mtf_grid_2d = make_mtf_grid(
        image_shape_2d, mtf_frequencies, mtf_amplitudes, rfft=True, fftshift=False
    )
    mtf_grid_3d = make_mtf_grid(
        image_shape_3d, mtf_frequencies, mtf_amplitudes, rfft=True, fftshift=False
    )

    # Test if the MTF grid values are within the expected range
    assert torch.all(mtf_grid_2d >= 0) and torch.all(
        mtf_grid_2d <= 1
    ), "MTF grid values out of range for 2D input"
    assert torch.all(mtf_grid_3d >= 0) and torch.all(
        mtf_grid_3d <= 1
    ), "MTF grid values out of range for 3D input"
