"""Tests for the polar grid functions."""

import pytest
import torch

from torch_grid_utils import (
    cartesian_to_polar,
    fftfreq_grid,
    fftfreq_grid_polar,
    normalize_polar_grid,
    polar_grid,
    polar_to_cartesian,
)


def test_cartesian_to_polar_basic():
    """Test basic Cartesian to polar conversion."""
    # Simple 2D grid with [y, x] order
    # Grid: [[(0,0), (0,1)], [(1,0), (1,1)]] in (y, x) coordinates
    cartesian = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]])
    rho, theta = cartesian_to_polar(cartesian)

    assert rho.shape == (2, 2)
    assert theta.shape == (2, 2)

    # Origin should have rho ≈ 0 (with eps)
    assert torch.allclose(rho[0, 0], torch.tensor(0.0), atol=1e-6)

    # Point at (x=1, y=0) should have rho=1, theta=0
    assert torch.allclose(rho[0, 1], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(theta[0, 1], torch.tensor(0.0), atol=1e-6)

    # Point at (x=0, y=1) should have rho=1, theta=π/2
    assert torch.allclose(rho[1, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(theta[1, 0], torch.tensor(torch.pi / 2), atol=1e-6)

    # Point at (x=1, y=1) should have rho=√2, theta=π/4
    assert torch.allclose(rho[1, 1], torch.tensor(2.0**0.5), atol=1e-6)
    assert torch.allclose(theta[1, 1], torch.tensor(torch.pi / 4), atol=1e-6)


def test_cartesian_to_polar_negative_angles():
    """Test that negative angles are handled correctly."""
    # Point at (x=-1, y=0) should have theta=π
    cartesian = torch.tensor([[[0.0, -1.0]]])  # [y, x] order
    rho, theta = cartesian_to_polar(cartesian)
    assert torch.allclose(rho[0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(theta[0, 0], torch.tensor(torch.pi), atol=1e-6)

    # Point at (x=-1, y=-1) should have theta=-3π/4
    cartesian = torch.tensor([[[-1.0, -1.0]]])  # [y, x] order
    rho, theta = cartesian_to_polar(cartesian)
    assert torch.allclose(rho[0, 0], torch.tensor(2.0**0.5), atol=1e-6)
    assert torch.allclose(theta[0, 0], torch.tensor(-3 * torch.pi / 4), atol=1e-6)


def test_cartesian_to_polar_eps():
    """Test that eps prevents division by zero at origin."""
    cartesian = torch.tensor([[[0.0, 0.0]]])
    rho, _ = cartesian_to_polar(cartesian, eps=1e-12)
    # rho should be small but not zero
    # Note: rho = sqrt(x^2 + y^2 + eps) = sqrt(eps) = sqrt(1e-12) ≈ 1e-6
    assert rho[0, 0] > 0
    assert rho[0, 0] < 1e-5


def test_cartesian_to_polar_error_handling():
    """Test error handling for invalid input shapes."""
    # Wrong last dimension size
    cartesian = torch.tensor([[[1.0, 2.0, 3.0]]])
    with pytest.raises(ValueError, match="cartesian_grid must have shape"):
        cartesian_to_polar(cartesian)


def test_polar_to_cartesian_basic():
    """Test basic polar to Cartesian conversion."""
    rho = torch.tensor([[0.0, 1.0], [1.0, 2.0**0.5]])
    theta = torch.tensor([[0.0, 0.0], [torch.pi / 2, torch.pi / 4]])

    cartesian = polar_to_cartesian(rho, theta)

    assert cartesian.shape == (2, 2, 2)

    # Point with rho=1, theta=0 should be at (x=1, y=0) -> [y=0, x=1]
    assert torch.allclose(cartesian[0, 1, 0], torch.tensor(0.0), atol=1e-6)  # y
    assert torch.allclose(cartesian[0, 1, 1], torch.tensor(1.0), atol=1e-6)  # x

    # Point with rho=1, theta=π/2 should be at (x=0, y=1) -> [y=1, x=0]
    assert torch.allclose(cartesian[1, 0, 0], torch.tensor(1.0), atol=1e-6)  # y
    assert torch.allclose(cartesian[1, 0, 1], torch.tensor(0.0), atol=1e-6)  # x

    # Point with rho=√2, theta=π/4 should be at (x=1, y=1) -> [y=1, x=1]
    assert torch.allclose(cartesian[1, 1, 0], torch.tensor(1.0), atol=1e-6)  # y
    assert torch.allclose(cartesian[1, 1, 1], torch.tensor(1.0), atol=1e-6)  # x


def test_polar_to_cartesian_error_handling():
    """Test error handling for mismatched shapes."""
    rho = torch.tensor([[1.0, 2.0]])
    theta = torch.tensor([[0.0]])  # Different shape

    with pytest.raises(ValueError, match="rho and theta must have the same shape"):
        polar_to_cartesian(rho, theta)


def test_cartesian_polar_roundtrip():
    """Test that converting Cartesian -> polar -> Cartesian recovers original."""
    # Create a simple Cartesian grid with [y, x] order
    cartesian_original = torch.tensor(
        [[[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]], [[1.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]]
    )

    # Convert to polar and back
    rho, theta = cartesian_to_polar(cartesian_original)
    cartesian_recovered = polar_to_cartesian(rho, theta)

    # Should recover original (within numerical precision)
    assert torch.allclose(cartesian_original, cartesian_recovered, atol=1e-6)


def test_polar_grid_basic():
    """Test basic polar grid creation."""
    rho, theta = polar_grid(image_shape=(4, 4))

    assert rho.shape == (4, 4)
    assert theta.shape == (4, 4)

    # Center should have rho ≈ 0 (with eps)
    center_h, center_w = 2, 2
    assert torch.allclose(rho[center_h, center_w], torch.tensor(0.0), atol=1e-6)

    # Theta should range from -π to π
    assert theta.min() >= -torch.pi
    assert theta.max() <= torch.pi


def test_polar_grid_with_center():
    """Test polar grid with custom center."""
    rho, theta = polar_grid(image_shape=(4, 4), center=(1.0, 1.0))

    assert rho.shape == (4, 4)
    assert theta.shape == (4, 4)

    # Center at (1, 1) should have rho ≈ 0
    assert torch.allclose(rho[1, 1], torch.tensor(0.0), atol=1e-6)


def test_polar_grid_with_normalize_rho():
    """Test polar grid with normalized rho."""
    rho, theta = polar_grid(image_shape=(4, 4))
    rho, theta = normalize_polar_grid(rho, theta)

    assert rho.shape == (4, 4)
    assert theta.shape == (4, 4)

    # Normalized rho should be in [0, 1]
    assert rho.min() >= 0.0
    assert rho.max() <= 1.0
    assert torch.allclose(rho.max(), torch.tensor(1.0), atol=1e-6)


def test_polar_grid_without_normalize_rho():
    """Test polar grid without normalized rho."""
    rho, _ = polar_grid(image_shape=(4, 4))

    assert rho.shape == (4, 4)
    # Without normalization, rho can be > 1
    assert rho.max() > 1.0


def test_polar_grid_error_handling():
    """Test error handling for non-2D shapes."""
    with pytest.raises(ValueError, match="polar_grid currently only supports 2D"):
        polar_grid(image_shape=(4, 4, 4))


def test_fftfreq_grid_polar_basic():
    """Test conversion of frequency grid to polar coordinates."""
    # Create a simple frequency grid
    freq_grid = fftfreq_grid(image_shape=(4, 4), rfft=False, fftshift=False)

    rho, theta = fftfreq_grid_polar(freq_grid, normalize_rho=False)

    assert rho.shape == (4, 4)
    assert theta.shape == (4, 4)

    # DC component (0, 0) should have rho ≈ 0
    assert torch.allclose(rho[0, 0], torch.tensor(0.0), atol=1e-6)


def test_fftfreq_grid_polar_with_normalize():
    """Test frequency grid to polar with normalization."""
    freq_grid = fftfreq_grid(image_shape=(4, 4), rfft=False, fftshift=False)

    rho, theta = fftfreq_grid_polar(freq_grid, normalize_rho=True)

    assert rho.shape == (4, 4)
    assert theta.shape == (4, 4)

    # Normalized rho should be in [0, 1]
    assert rho.min() >= 0.0
    assert rho.max() <= 1.0
    assert torch.allclose(rho.max(), torch.tensor(1.0), atol=1e-6)


def test_fftfreq_grid_polar_without_normalize():
    """Test frequency grid to polar without normalization."""
    freq_grid = fftfreq_grid(image_shape=(4, 4), rfft=False, fftshift=False)

    rho, _ = fftfreq_grid_polar(freq_grid, normalize_rho=False)

    assert rho.shape == (4, 4)
    # Without normalization, rho can be > 1
    assert rho.max() > 0.0


def test_fftfreq_grid_polar_with_fftshift():
    """Test frequency grid to polar with fftshift."""
    freq_grid = fftfreq_grid(image_shape=(4, 4), rfft=False, fftshift=True)

    rho, theta = fftfreq_grid_polar(freq_grid, normalize_rho=True)

    assert rho.shape == (4, 4)
    assert theta.shape == (4, 4)

    # DC component should be at center (2, 2) in fftshifted grid
    # Note: rho will be small but not exactly zero due to eps
    assert rho[2, 2] < 1e-5


def test_fftfreq_grid_polar_rfft():
    """Test frequency grid to polar with rfft."""
    freq_grid = fftfreq_grid(image_shape=(4, 4), rfft=True, fftshift=False)

    rho, theta = fftfreq_grid_polar(freq_grid, normalize_rho=True)

    # rfft changes the width dimension
    assert rho.shape == (4, 3)
    assert theta.shape == (4, 3)


def test_polar_grid_device():
    """Test that polar grid respects device parameter."""
    if torch.cuda.is_available():
        rho, theta = polar_grid(image_shape=(4, 4), device=torch.device("cuda"))

        assert rho.device.type == "cuda"
        assert theta.device.type == "cuda"


def test_cartesian_to_polar_3d_batch():
    """Test cartesian_to_polar with 3D batch dimensions."""
    # Shape: (batch, height, width, 2) with [y, x] order
    cartesian = torch.tensor(
        [
            [[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]],
            [[[0.0, 2.0], [2.0, 0.0]], [[0.0, -1.0], [-1.0, 0.0]]],
        ]
    )

    rho, theta = cartesian_to_polar(cartesian)

    assert rho.shape == (2, 2, 2)
    assert theta.shape == (2, 2, 2)

    # First batch, origin
    assert torch.allclose(rho[0, 0, 0], torch.tensor(0.0), atol=1e-6)

    # First batch, (x=1, y=0)
    assert torch.allclose(rho[0, 0, 1], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(theta[0, 0, 1], torch.tensor(0.0), atol=1e-6)


def test_polar_to_cartesian_3d_batch():
    """Test polar_to_cartesian with 3D batch dimensions."""
    rho = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    theta = torch.tensor(
        [
            [[0.0, torch.pi / 2], [torch.pi, -torch.pi / 2]],
            [[0.0, torch.pi / 4], [torch.pi / 3, -torch.pi / 6]],
        ]
    )

    cartesian = polar_to_cartesian(rho, theta)

    assert cartesian.shape == (2, 2, 2, 2)

    # Round-trip test
    rho_recovered, theta_recovered = cartesian_to_polar(cartesian)
    assert torch.allclose(rho, rho_recovered, atol=1e-6)
    # Note: atan2 returns -π for negative x, but π and -π represent the same angle
    # So we need to handle the case where theta might differ by 2π
    theta_diff = torch.abs(theta - theta_recovered)
    theta_diff = torch.minimum(theta_diff, 2 * torch.pi - theta_diff)
    assert torch.allclose(theta_diff, torch.zeros_like(theta_diff), atol=1e-6)
