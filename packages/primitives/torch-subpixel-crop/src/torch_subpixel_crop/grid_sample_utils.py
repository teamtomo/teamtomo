from typing import Sequence

import torch


def array_to_grid_sample(
    array_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate grids for `torch.nn.functional.grid_sample` from array coordinates.

    These coordinates should be used with `align_corners=True` in
    `torch.nn.functional.grid_sample`.


    Parameters
    ----------
    array_coordinates: torch.Tensor
        `(..., d)` array of d-dimensional coordinates.
        Coordinates are in the range `[0, N-1]` for the `N` elements in each dimension.
    array_shape: Sequence[int]
        shape of the array being sampled at `array_coordinates`.
    """
    dtype, device = array_coordinates.dtype, array_coordinates.device
    array_shape = torch.as_tensor(array_shape, dtype=dtype, device=device)
    grid_sample_coordinates = (array_coordinates / (0.5 * array_shape - 0.5)) - 1
    grid_sample_coordinates = torch.flip(grid_sample_coordinates, dims=(-1,))
    return grid_sample_coordinates


def grid_sample_to_array(
    grid_sample_coordinates: torch.Tensor, array_shape: Sequence[int]
) -> torch.Tensor:
    """Generate array coordinates from `torch.nn.functional.grid_sample` grids.

    Parameters
    ----------
    grid_sample_coordinates: torch.Tensor
        `(..., d)` array of coordinates to be used with `torch.nn.functional.grid_sample`.
    array_shape: Sequence[int]
        shape of the array `grid_sample_coordinates` are used to sample.
    """
    dtype, device = grid_sample_coordinates.dtype, grid_sample_coordinates.device
    array_shape = torch.as_tensor(array_shape, dtype=dtype, device=device)
    array_shape = torch.flip(array_shape, dims=(-1,))
    array_coordinates = (grid_sample_coordinates + 1) * (0.5 * array_shape - 0.5)
    array_coordinates = torch.flip(array_coordinates, dims=(-1,))
    return array_coordinates
