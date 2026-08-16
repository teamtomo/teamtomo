import einops
import torch

from torch_cubic_spline_grids import (
    CubicBSplineGrid1d,
    CubicBSplineGrid2d,
    CubicBSplineGrid3d,
    CubicBSplineGrid4d,
)


def _fit_lbfgs(grid, x, observations, loss_fn, max_iter):
    optimiser = torch.optim.LBFGS(
        grid.parameters(), lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe'
    )

    def closure():
        optimiser.zero_grad()
        prediction = grid(x).squeeze()
        loss = loss_fn(prediction, observations)
        loss.backward()
        return loss

    optimiser.step(closure)


def test_1d_grid_optimisation():
    grid_resolution = 6
    n_observations = 200
    grid = CubicBSplineGrid1d(resolution=grid_resolution, n_channels=1)

    def f(x: torch.Tensor):
        return torch.sin(x * 2 * torch.pi)

    x = torch.rand(size=(n_observations,))
    observations = f(x)
    _fit_lbfgs(
        grid, x, observations, lambda p, o: torch.mean(torch.abs(p - o)), max_iter=20
    )

    x = torch.linspace(0, 1, steps=100)
    ground_truth = f(x)
    prediction = grid(x).squeeze()
    mean_absolute_error = torch.mean(torch.abs(prediction - ground_truth))
    assert mean_absolute_error.item() < 0.02


def test_1d_grid_optimization_decreasing():
    grid_resolution = 8
    n_observations = 200
    grid = CubicBSplineGrid1d(
        resolution=grid_resolution, n_channels=1, monotonicity='decreasing'
    )

    def f(x: torch.Tensor):
        return torch.exp(-5 * x)

    x = torch.rand(size=(n_observations,))
    observations = f(x)
    _fit_lbfgs(
        grid, x, observations, lambda p, o: torch.mean(torch.abs(p - o)), max_iter=15
    )

    x = torch.linspace(0, 1, steps=100)
    prediction = grid(x).squeeze()

    eps = torch.tensor(1e-5, dtype=prediction.dtype)
    non_increasing = torch.diff(prediction, dim=-1) <= eps
    assert non_increasing.all().item()


def test_2d_grid_optimisation():
    grid_resolution = (3, 3)
    n_observations = 200
    grid = CubicBSplineGrid2d(resolution=grid_resolution, n_channels=1)

    def f(x: torch.Tensor):
        centered = x - 0.5
        return torch.sqrt(torch.sum(centered**2, dim=-1))  # (x**2 + y**2) ** 0.5

    x = torch.rand(size=(n_observations, 2))
    observations = f(x)
    _fit_lbfgs(
        grid, x, observations, lambda p, o: torch.mean((p - o) ** 2), max_iter=15
    )

    _x = torch.linspace(0, 1, steps=100)
    x = torch.meshgrid(_x, _x, indexing='xy')
    x = einops.rearrange([*x], 'xy h w -> (h w) xy')
    ground_truth = f(x)
    prediction = grid(x).squeeze()
    mean_absolute_error = torch.mean(torch.abs(prediction - ground_truth))
    assert mean_absolute_error.item() < 0.02


def test_3d_grid_optimisation():
    grid_resolution = (3, 3, 3)
    n_observations = 1000
    grid = CubicBSplineGrid3d(resolution=grid_resolution, n_channels=1)

    def f(x: torch.Tensor):
        centered = x - 0.5
        return torch.sqrt(torch.sum(centered**2, dim=-1))  # (x**2 + y**2 + z**2) ** 0.5

    x = torch.rand(size=(n_observations, 3))
    observations = f(x)
    _fit_lbfgs(
        grid, x, observations, lambda p, o: torch.mean((p - o) ** 2), max_iter=10
    )

    _x = torch.linspace(0, 1, steps=100)
    x = torch.meshgrid(_x, _x, _x, indexing='xy')
    x = einops.rearrange([*x], 'xyz d h w -> (d h w) xyz')
    ground_truth = f(x)
    prediction = grid(x).squeeze()
    mean_absolute_error = torch.mean(torch.abs(prediction - ground_truth))
    assert mean_absolute_error.item() < 0.02


def test_4d_grid_optimisation():
    grid_resolution = (3, 3, 3, 3)
    n_observations = 1000
    grid = CubicBSplineGrid4d(resolution=grid_resolution, n_channels=1)

    def f(x: torch.Tensor):
        centered = x - 0.5
        return torch.sqrt(torch.sum(centered**2, dim=-1))

    x = torch.rand(size=(n_observations, 4))
    observations = f(x)
    _fit_lbfgs(
        grid, x, observations, lambda p, o: torch.mean((p - o) ** 2), max_iter=15
    )

    _x = torch.linspace(0, 1, steps=10)
    x = torch.meshgrid(_x, _x, _x, _x, indexing='xy')
    x = einops.rearrange([*x], 'xyz u d h w -> (u d h w) xyz')
    ground_truth = f(x)
    prediction = grid(x).squeeze()
    mean_absolute_error = torch.mean(torch.abs(prediction - ground_truth))
    assert mean_absolute_error.item() < 0.02
