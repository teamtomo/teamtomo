import warnings
from typing import Callable, Optional, Tuple

import einops
import torch
from typing_extensions import Self

from torch_cubic_spline_grids.utils import (
    MonotonicityType,
    batch,
    coerce_to_multichannel_grid,
)


class CubicSplineGrid(torch.nn.Module):
    """Base class for continuous parametrisations of multidimensional spaces."""

    ndim: int
    _data: torch.nn.Parameter
    _interpolation_function: Callable
    _interpolation_matrix: torch.Tensor
    _minibatch_size: int

    def __init__(
        self,
        resolution: Optional[Tuple[int, ...]] = None,
        n_channels: int = 1,
        minibatch_size: int = 1_000_000,
        monotonicity: Optional[MonotonicityType] = None,
    ):
        super().__init__()
        if resolution is None:
            resolution = (2,) * self.ndim
        grid_shape = (n_channels, *resolution)
        self.data = torch.zeros(size=grid_shape)
        self._minibatch_size = minibatch_size
        self._monotonicity = monotonicity
        self.register_buffer(
            name='interpolation_matrix',
            tensor=self._interpolation_matrix,
            persistent=False,
        )

    def _interpolate(self, u: torch.Tensor) -> torch.Tensor:
        return self._interpolation_function(
            self._data,
            u,
            matrix=self.interpolation_matrix,
            monotonicity=self._monotonicity,
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u = self._coerce_to_batched_coordinates(u)  # (b, d)

        interpolated = [
            self._interpolate(minibatch_u)
            for minibatch_u in batch(u, n=self._minibatch_size)
        ]  # List[Tensor[(b, d)]]
        interpolated = torch.cat(interpolated, dim=0)  # (b, d)
        return self._unpack_interpolated_output(interpolated)

    @classmethod
    def from_grid_data(cls, data: torch.Tensor) -> Self:
        """Instantiate a grid from existing grid data.

        Parameters
        ----------
        data: torch.Tensor
            (c, *grid_dimensions) or (*grid_dimensions) array of multichannel values at
            each grid point.
        """
        grid = cls()
        grid.data = data
        return grid

    @classmethod
    def from_samples(
        cls, points: torch.Tensor, values: torch.Tensor, resolution: Tuple[int, ...]
    ) -> Self:
        """Instantiate a grid that best fits `values` sampled at `points`.

        The control points are found by least squares. The spline is linear in
        its control points, so evaluating each one-hot control point at `points`
        gives a matrix A with A[k, j] = B_j(points[k]), and the fit solves
        A @ c ≈ values. With one sample at each grid point the fit is exact
        (see `from_interpolants_at_control_points`).

        Parameters
        ----------
        points: torch.Tensor
            (m, ndim) sample positions in [0, 1] along each axis.
        values: torch.Tensor
            (m, ) or (m, c) values at `points`.
        resolution: Tuple[int, ...]
            Number of control points along each axis.
        """
        points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, cls.ndim)
        values = torch.as_tensor(values)
        values_2d = values.reshape(points.shape[0], -1)  # (m, c)
        n = int(torch.tensor(resolution).prod())
        with torch.no_grad():
            # column j of A: the spline at every sample when control point j is 1
            one_hot = torch.eye(n).reshape(n, *resolution)
            A = cls.from_grid_data(one_hot)(points.cpu()).reshape(-1, n).double()
            rank = int(torch.linalg.matrix_rank(A))
            if rank < n:
                warnings.warn(
                    f'samples determine only {rank} of {n} control points; '
                    'the fit is not unique where there are no samples',
                    stacklevel=2,
                )
            c = torch.linalg.lstsq(A, values_2d.cpu().double()).solution  # (n, c)
        data = c.T.reshape(-1, *resolution).to(values.dtype).to(values.device)
        return cls.from_grid_data(data)

    @classmethod
    def from_interpolants_at_control_points(cls, data: torch.Tensor) -> Self:
        """Instantiate a grid whose spline passes through `data` at the grid points.

        Same input as `from_grid_data`, but the values are *interpolated*: the
        control points are solved for so that evaluating the grid at the grid
        points (i / (n - 1) along each axis) returns `data`. This is
        `from_samples` with one sample at each grid point; that system is square
        and invertible, so the fit is exact.

        Parameters
        ----------
        data: torch.Tensor
            (c, *grid_dimensions) or (*grid_dimensions) array of values at each
            grid point.
        """
        data = coerce_to_multichannel_grid(data, grid_ndim=cls.ndim)  # (c, *res)
        resolution = tuple(data.shape[1:])
        values = data.reshape(data.shape[0], -1).T  # (n, c), one per grid point
        return cls.from_samples(_grid_points(resolution), values, resolution)

    def to_interpolants_at_control_points(self) -> torch.Tensor:
        """Evaluate the grid at its grid points.

        The inverse of `from_interpolants_at_control_points`.

        Returns
        -------
        data: torch.Tensor
            `(c, *grid_dimensions)` values of the spline at each grid point
            (i / (n - 1) along each axis). For a Catmull-Rom grid this equals
            `data`; for a B-spline grid it differs from the control points.
        """
        grid_points = _grid_points(self.resolution).to(self._data.device)
        with torch.no_grad():
            values = self(grid_points)  # (n, c) or (n,) for a single channel
        values = values.reshape(-1, self.n_channels).T  # (c, n)
        return values.reshape(self.n_channels, *self.resolution)

    @property
    def data(self) -> torch.Tensor:
        return self._data.detach()

    @data.setter
    def data(self, grid_data: torch.Tensor) -> None:
        grid_data = coerce_to_multichannel_grid(grid_data, grid_ndim=self.ndim)
        self._data = torch.nn.Parameter(grid_data)

    @property
    def n_channels(self) -> int:
        return int(self._data.size(0))

    @property
    def resolution(self) -> Tuple[int, ...]:
        return tuple(self._data.shape[1:])

    def _coerce_to_batched_coordinates(self, u: torch.Tensor) -> torch.Tensor:
        u = torch.atleast_1d(torch.as_tensor(u, dtype=torch.float32))
        self._input_is_coordinate_like = u.shape[-1] == self.ndim
        if self._input_is_coordinate_like is False and self.ndim == 1:
            u = einops.rearrange(u, '... -> ... 1')  # add singleton coord dimension
        else:
            u = torch.atleast_2d(u)  # add batch dimension if missing
        u, self._packed_shapes = einops.pack([u], pattern='* coords')
        if u.shape[-1] != self.ndim:
            ndim = u.shape[-1]
            raise ValueError(
                f'Cannot interpolate on a {self.ndim}D grid with {ndim}D coordinates'
            )
        return u

    def _unpack_interpolated_output(self, interpolated: torch.Tensor) -> torch.Tensor:
        [interpolated] = einops.unpack(
            interpolated, packed_shapes=self._packed_shapes, pattern='* coords'
        )
        if self._input_is_coordinate_like is False and self.ndim == 1:
            interpolated = einops.rearrange(interpolated, '... 1 -> ...')
        return interpolated


def _grid_points(resolution: Tuple[int, ...]) -> torch.Tensor:
    """`(n, ndim)` coordinates of every grid point, i / (r - 1) along each axis."""
    axes = [torch.linspace(0, 1, r) for r in resolution]
    grid_points = torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1)
    return grid_points.reshape(-1, len(resolution))
