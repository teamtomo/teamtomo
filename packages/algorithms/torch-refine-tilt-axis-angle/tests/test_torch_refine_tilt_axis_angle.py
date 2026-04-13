from unittest.mock import MagicMock, patch

import pytest
import torch

from torch_refine_tilt_axis_angle import refine_tilt_axis_angle


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a small tilt series (5 images of 20x20 pixels)
    tilt_series = torch.rand(5, 20, 20, device=device)

    return {"tilt_series": tilt_series, "device": device}


def test_refine_tilt_axis_angle_default_parameters(sample_data):
    """Test refine_tilt_axis_angle with default parameters."""
    mock_result = MagicMock()
    mock_result.x = 0.0

    with patch(
        "torch_refine_tilt_axis_angle.refine_tilt_axis_angle.minimize_scalar",
        return_value=mock_result,
    ):
        # Setup
        tilt_series = sample_data["tilt_series"]

        # Call the function with default parameters
        result = refine_tilt_axis_angle(tilt_series=tilt_series)

        # Assertions
        assert isinstance(result, float)


def test_refine_tilt_axis_angle_with_custom_parameters(sample_data):
    """Test refine_tilt_axis_angle with custom parameters."""
    # Create a mock result object
    mock_result = MagicMock()
    mock_result.x = 42.5

    with patch(
        "torch_refine_tilt_axis_angle.refine_tilt_axis_angle.minimize_scalar",
        return_value=mock_result,
    ):
        # Setup
        tilt_series = sample_data["tilt_series"]
        initial_angle = 45.0

        # Call the function
        result = refine_tilt_axis_angle(
            tilt_series=tilt_series,
            tilt_axis_angle=initial_angle,
            search_range=15.0,
            max_iter=20,
        )

        # Assertions
        assert isinstance(result, float)
        assert result == 42.5


def test_refine_tilt_axis_angle_bounds(sample_data):
    """Test that minimize_scalar is called with correct bounds."""
    mock_result = MagicMock()
    mock_result.x = 5.0

    with patch(
        "torch_refine_tilt_axis_angle.refine_tilt_axis_angle.minimize_scalar",
        return_value=mock_result,
    ) as mock_minimize:
        # Setup
        tilt_series = sample_data["tilt_series"]
        initial_angle = 30.0
        search_range = 10.0

        # Call the function
        refine_tilt_axis_angle(
            tilt_series=tilt_series,
            tilt_axis_angle=initial_angle,
            search_range=search_range,
        )

        # Assert minimize_scalar was called with correct bounds
        mock_minimize.assert_called_once()
        call_kwargs = mock_minimize.call_args[1]
        assert call_kwargs["bounds"] == (20.0, 40.0)  # initial ± search_range
        assert call_kwargs["method"] == "bounded"


def test_refine_tilt_axis_angle_optimization_max_iter(sample_data):
    """Test that max_iter parameter is passed to minimize_scalar."""
    mock_result = MagicMock()
    mock_result.x = 0.0

    with patch(
        "torch_refine_tilt_axis_angle.refine_tilt_axis_angle.minimize_scalar",
        return_value=mock_result,
    ) as mock_minimize:
        # Setup
        tilt_series = sample_data["tilt_series"]
        max_iter = 15

        # Call the function
        refine_tilt_axis_angle(
            tilt_series=tilt_series,
            max_iter=max_iter,
        )

        # Assert minimize_scalar was called with correct max_iter
        mock_minimize.assert_called_once()
        call_kwargs = mock_minimize.call_args[1]
        assert call_kwargs["options"]["maxiter"] == max_iter
