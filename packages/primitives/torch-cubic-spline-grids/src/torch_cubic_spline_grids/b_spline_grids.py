from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from torch_cubic_spline_grids._base_cubic_grid import CubicSplineGrid
from torch_cubic_spline_grids._constants import CUBIC_B_SPLINE_MATRIX
from torch_cubic_spline_grids.interpolate_grids import (
    interpolate_grid_1d as _interpolate_grid_1d,
    interpolate_grid_2d as _interpolate_grid_2d,
    interpolate_grid_3d as _interpolate_grid_3d,
    interpolate_grid_4d as _interpolate_grid_4d,
)
from torch_cubic_spline_grids.utils import MonotonicityType

CoordinateLike = Union[float, Sequence[float], torch.Tensor]


class _CubicBSplineGrid(CubicSplineGrid):
    _interpolation_matrix = CUBIC_B_SPLINE_MATRIX


class CubicBSplineGrid1d(_CubicBSplineGrid):
    """Continuous parametrisation of a 1D space with a specific resolution.

    Values are interpolated with a cubic B-spline through a grid of learnable
    control points of shape `(n_channels, w)`. The grid is queried by calling it on
    `(..., 1)` or `(...)` coordinates, where `[0, 1]` covers the full grid.

    Parameters
    ----------
    resolution : int | tuple[int] | None, optional
        Number of control points along the grid dimension, as an int or a
        length-1 tuple. Defaults to 2.
    n_channels : int, optional
        Number of channels (independent values) at each grid point.
    minibatch_size : int, optional
        Maximum number of query points interpolated at once.
    monotonicity : Literal['increasing', 'decreasing'] | None, optional
        If set, control points are made monotonic along every grid dimension
        (cumulative max for 'increasing', cumulative min for 'decreasing')
        before interpolation. `None` applies no constraint.
    """

    ndim: int = 1
    _interpolation_function: Callable = staticmethod(_interpolate_grid_1d)

    def __init__(
        self,
        resolution: Optional[Union[int, Tuple[int]]] = None,
        n_channels: int = 1,
        minibatch_size: int = 1_000_000,
        monotonicity: Optional[MonotonicityType] = None,
    ):
        if isinstance(resolution, int):
            resolution = (resolution,)
        super().__init__(
            resolution=resolution,
            n_channels=n_channels,
            minibatch_size=minibatch_size,
            monotonicity=monotonicity,
        )


class CubicBSplineGrid2d(_CubicBSplineGrid):
    """Continuous parametrisation of a 2D space with a specific resolution.

    Values are interpolated with a cubic B-spline through a grid of learnable
    control points of shape `(n_channels, h, w)`. The grid is queried by calling it
    on `(..., 2)` coordinates in which `[0, 1]` along each axis covers the
    corresponding `(h, w)` grid dimension.

    Parameters
    ----------
    resolution : tuple[int, ...] | None, optional
        `(h, w)` number of control points along each grid dimension.
        Defaults to 2 in every dimension.
    n_channels : int, optional
        Number of channels (independent values) at each grid point.
    minibatch_size : int, optional
        Maximum number of query points interpolated at once.
    monotonicity : Literal['increasing', 'decreasing'] | None, optional
        If set, control points are made monotonic along every grid dimension
        (cumulative max for 'increasing', cumulative min for 'decreasing')
        before interpolation. `None` applies no constraint.
    """

    ndim: int = 2
    _interpolation_function: Callable = staticmethod(_interpolate_grid_2d)


class CubicBSplineGrid3d(_CubicBSplineGrid):
    """Continuous parametrisation of a 3D space with a specific resolution.

    Values are interpolated with a cubic B-spline through a grid of learnable
    control points of shape `(n_channels, d, h, w)`. The grid is queried by calling
    it on `(..., 3)` coordinates in which `[0, 1]` along each axis covers the
    corresponding `(d, h, w)` grid dimension.

    Parameters
    ----------
    resolution : tuple[int, ...] | None, optional
        `(d, h, w)` number of control points along each grid dimension.
        Defaults to 2 in every dimension.
    n_channels : int, optional
        Number of channels (independent values) at each grid point.
    minibatch_size : int, optional
        Maximum number of query points interpolated at once.
    monotonicity : Literal['increasing', 'decreasing'] | None, optional
        If set, control points are made monotonic along every grid dimension
        (cumulative max for 'increasing', cumulative min for 'decreasing')
        before interpolation. `None` applies no constraint.
    """

    ndim: int = 3
    _interpolation_function: Callable = staticmethod(_interpolate_grid_3d)


class CubicBSplineGrid4d(_CubicBSplineGrid):
    """Continuous parametrisation of a 4D space with a specific resolution.

    Values are interpolated with a cubic B-spline through a grid of learnable
    control points of shape `(n_channels, t, d, h, w)`. The grid is queried by
    calling it on `(..., 4)` coordinates in which `[0, 1]` along each axis covers
    the corresponding `(t, d, h, w)` grid dimension.

    Parameters
    ----------
    resolution : tuple[int, ...] | None, optional
        `(t, d, h, w)` number of control points along each grid dimension.
        Defaults to 2 in every dimension.
    n_channels : int, optional
        Number of channels (independent values) at each grid point.
    minibatch_size : int, optional
        Maximum number of query points interpolated at once.
    monotonicity : Literal['increasing', 'decreasing'] | None, optional
        If set, control points are made monotonic along every grid dimension
        (cumulative max for 'increasing', cumulative min for 'decreasing')
        before interpolation. `None` applies no constraint.
    """

    ndim: int = 4
    _interpolation_function: Callable = staticmethod(_interpolate_grid_4d)
