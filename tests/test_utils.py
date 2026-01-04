"""Test functions in utils.py."""

import numpy as np
import pytest
import torch

from torch_fourier_filter.utils import (
    bin_1dim_with_lerp,
    curve_1dim_to_ndim,
    torch_interp,
)


def test_torch_interp():
    """Compare torch_interp to np.interp."""
    # Construct some random data
    x = torch.rand(1000) * 12 - 2
    xp = torch.linspace(0, 8, 200)
    fp = torch.sin(xp)

    # Test 1. Interpolation with other default values
    y_torch = torch_interp(x, xp, fp)
    y_numpy = np.interp(x.numpy(), xp.numpy(), fp.numpy())

    assert np.allclose(y_torch.numpy(), y_numpy, atol=1e-6)

    # Test 2. Interpolation with left and right values
    y_torch = torch_interp(x, xp, fp, left=10, right=10)
    y_numpy = np.interp(x.numpy(), xp.numpy(), fp.numpy(), left=10, right=10)

    assert np.allclose(y_torch.numpy(), y_numpy, atol=1e-6)

    # Test 3. NotImplementedError for period
    with pytest.raises(NotImplementedError):
        torch_interp(x, xp, fp, period=1.0)

    # Test 4. ValueError for different shapes
    with pytest.raises(ValueError):
        torch_interp(x, xp, fp[:-1])

    # Test 5. ValueError for multiple dimensions
    with pytest.raises(ValueError):
        torch_interp(x, xp.unsqueeze(0), fp)
    with pytest.raises(ValueError):
        torch_interp(x, xp, fp.unsqueeze(0))


def test_torch_interp_batched():
    """Basic test for interpolating across batched inputs."""
    # Construct some random data
    x = torch.rand(1000) * 12 - 2  # Grid is always constant
    xp0 = torch.linspace(0, 8, 200)
    xp1 = torch.linspace(0, 12, 200)
    xp2 = torch.linspace(4, 12, 200)
    fp0 = torch.sin(xp0)
    fp1 = torch.cos(xp1) + 1
    fp2 = torch.exp(xp2)

    # stack into batched inputs
    xp = torch.stack([xp0, xp1, xp2], dim=0)
    fp = torch.stack([fp0, fp1, fp2], dim=0)

    # Test 1. Interpolation with other default values
    y_torch = torch_interp(x, xp, fp)
    y0_numpy = np.interp(x.numpy(), xp0.numpy(), fp0.numpy())
    y1_numpy = np.interp(x.numpy(), xp1.numpy(), fp1.numpy())
    y2_numpy = np.interp(x.numpy(), xp2.numpy(), fp2.numpy())

    assert np.allclose(y_torch[0].numpy(), y0_numpy, atol=1e-6)
    assert np.allclose(y_torch[1].numpy(), y1_numpy, atol=1e-6)
    assert np.allclose(y_torch[2].numpy(), y2_numpy, atol=1e-6)


def test_curve_1dim_to_ndim():
    """Test converting a 1D curve at frequencies into a multi-dimensional grid."""

    def f(x):
        """Example function to construct data."""
        return torch.sin(2 * np.pi * x) + 2 * x

    def g(x):
        """Another example function to construct data."""
        return torch.exp(-(x**2)) * torch.cos(2 * np.pi * x)

    # Construct some example data
    frequencies = torch.linspace(0, 1, 1000)  # Needs to be highly sampled
    values = f(frequencies)

    # Construct a 2D grid
    x = torch.linspace(0, 0.5, 16)
    y = torch.linspace(0, 0.5, 16)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    frequency_grid = torch.sqrt(xx**2 + yy**2)

    # Test 1. Convert 1D curve to 2D grid
    filter_2d = curve_1dim_to_ndim(values, frequencies, frequency_grid)
    expected_2d = f(frequency_grid)

    assert torch.allclose(filter_2d, expected_2d, atol=1e-6)

    # Test 2. Batched inputs
    frequencies = torch.linspace(0, 1, 1000)  # Needs to be highly sampled
    values1 = f(frequencies)
    values2 = g(frequencies)
    values_batched = torch.stack([values1, values2], dim=0)

    filter_batched = curve_1dim_to_ndim(values_batched, frequencies, frequency_grid)
    expected1 = f(frequency_grid)
    expected2 = g(frequency_grid)
    expected_batched = torch.stack([expected1, expected2], dim=0)

    assert torch.allclose(filter_batched, expected_batched, atol=1e-6)

    # Test 3. ValueError for too many dimensions
    with pytest.raises(ValueError):
        curve_1dim_to_ndim(values_batched.unsqueeze(0), frequencies, frequency_grid)

    # Test 4. ValueError for different shapes
    with pytest.raises(ValueError):
        curve_1dim_to_ndim(values_batched, frequencies[:-1], frequency_grid)


def test_bin_1dim_with_lerp():
    # Test case 1: Basic functionality
    x = torch.tensor([0.5, 1.5, 2.5])
    y = torch.tensor([1.0, 2.0, 3.0])
    xp = torch.tensor([0.0, 1.0, 2.0, 3.0])
    expected_output = torch.tensor([0.5, 1.5, 2.5, 1.5])
    output = bin_1dim_with_lerp(x, y, xp, normalize_by_count=False)
    assert torch.allclose(
        output, expected_output
    ), f"Expected {expected_output}, but got {output}"

    # Test case 1b: Basic functionality with normalization
    x = torch.tensor([0.5, 1.5, 2.5])
    y = torch.tensor([1.0, 2.0, 3.0])
    xp = torch.tensor([0.0, 1.0, 2.0, 3.0])
    expected_output = torch.tensor([1.0, 1.5, 2.5, 3.0])
    output = bin_1dim_with_lerp(x, y, xp, normalize_by_count=True)
    assert torch.allclose(
        output, expected_output
    ), f"Expected {expected_output}, but got {output}"

    # Test case 2: x and y have different shapes
    x = torch.tensor([0.5, 1.5])
    y = torch.tensor([1.0, 2.0, 3.0])
    xp = torch.tensor([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="x and y must have the same shape."):
        bin_1dim_with_lerp(x, y, xp)

    # Test case 3: xp is not 1D
    x = torch.tensor([0.5, 1.5, 2.5])
    y = torch.tensor([1.0, 2.0, 3.0])
    xp = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="xp must be 1D."):
        bin_1dim_with_lerp(x, y, xp)

    # Test case 4: Out of bounds values
    x = torch.tensor([-10, 0.0, 1.0, 10])
    y = torch.tensor([-1.0, 3.0, 9.0, -1.0])
    xp = torch.tensor([0.0, 1.0, 2.0, 3.0])
    expected_output = torch.tensor([3.0, 9.0, 0.0, 0.0])
    output = bin_1dim_with_lerp(x, y, xp)
    assert torch.allclose(
        output, expected_output
    ), f"Expected {expected_output}, but got {output}"
