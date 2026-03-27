"""Tests for motion estimation functions."""

import pytest
import torch

from torch_motion_correction.estimate_motion_optimizer import estimate_local_motion
from torch_motion_correction.estimate_motion_xc import (
    estimate_global_motion,
    estimate_motion_cross_correlation_patches,
)


@pytest.fixture
def sample_image():
    """Create a sample image tensor for testing."""
    # Create a simple test image with some structure
    t, h, w = 5, 64, 64
    image = torch.zeros((t, h, w))
    # Add a simple pattern that shifts across frames
    for frame_idx in range(t):
        y_center = h // 2 + frame_idx * 2  # Shift down by 2 pixels per frame
        x_center = w // 2 + frame_idx * 1  # Shift right by 1 pixel per frame
        y_center = y_center % h
        x_center = x_center % w
        # Create a simple Gaussian-like blob
        y, x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        dist_sq = (y - y_center) ** 2 + (x - x_center) ** 2
        image[frame_idx] = torch.exp(-dist_sq / (2 * 10**2))
    return image


@pytest.fixture
def pixel_spacing():
    """Pixel spacing in Angstroms."""
    return 1.0


class TestEstimateGlobalMotion:
    """Tests for estimate_global_motion function."""

    def test_basic_functionality(self, sample_image, pixel_spacing):
        """Test basic motion estimation."""
        deformation_field = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        # Check output shape: (2, t, 1, 1) for global motion
        assert deformation_field.shape == (2, sample_image.shape[0], 1, 1)
        # Check that it's a tensor
        assert isinstance(deformation_field, torch.Tensor)

    def test_reference_frame(self, sample_image, pixel_spacing):
        """Test that reference frame parameter works."""
        deformation_field = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            reference_frame=0,
            device=torch.device("cpu"),
        )
        assert deformation_field.shape == (2, sample_image.shape[0], 1, 1)

    def test_different_devices(self, sample_image, pixel_spacing):
        """Test that device parameter works."""
        if torch.cuda.is_available():
            deformation_field = estimate_global_motion(
                image=sample_image,
                pixel_spacing=pixel_spacing,
                device=torch.device("cuda"),
            )
            assert deformation_field.device.type == "cuda"
        else:
            pytest.skip("CUDA not available")

    def test_b_factor_parameter(self, sample_image, pixel_spacing):
        """Test that b_factor parameter works."""
        deformation_field = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            b_factor=1000,
            device=torch.device("cpu"),
        )
        assert deformation_field.shape == (2, sample_image.shape[0], 1, 1)

    def test_frequency_range_parameter(self, sample_image, pixel_spacing):
        """Test that frequency_range parameter works."""
        deformation_field = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            frequency_range=(200, 20),
            device=torch.device("cpu"),
        )
        assert deformation_field.shape == (2, sample_image.shape[0], 1, 1)


class TestEstimateMotionCrossCorrelationPatches:
    """Tests for estimate_motion_cross_correlation_patches function."""

    def test_basic_functionality(self, sample_image, pixel_spacing):
        """Test basic patch-based motion estimation."""
        deformation_field, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sidelength=32,
            device=torch.device("cpu"),
        )
        # Check output shapes
        assert len(deformation_field.shape) == 4  # (2, t, gh, gw)
        assert deformation_field.shape[0] == 2  # y, x
        assert deformation_field.shape[1] == sample_image.shape[0]  # t
        assert len(patch_positions.shape) == 4  # (t, gh, gw, 3)
        assert patch_positions.shape[0] == sample_image.shape[0]  # t
        assert patch_positions is not None

    def test_reference_strategy_middle_frame(self, sample_image, pixel_spacing):
        """Test middle_frame reference strategy."""
        deformation_field, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sidelength=32,
            reference_strategy="middle_frame",
            device=torch.device("cpu"),
        )
        assert deformation_field.shape[0] == 2
        assert patch_positions is not None

    def test_reference_strategy_mean_except_current(self, sample_image, pixel_spacing):
        """Test mean_except_current reference strategy."""
        deformation_field, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sidelength=32,
            reference_strategy="mean_except_current",
            device=torch.device("cpu"),
        )
        assert deformation_field.shape[0] == 2
        assert patch_positions is not None

    def test_sub_pixel_refinement(self, sample_image, pixel_spacing):
        """Test sub-pixel refinement option."""
        deformation_field, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sidelength=32,
            sub_pixel_refinement=True,
            device=torch.device("cpu"),
        )
        assert deformation_field.shape[0] == 2
        assert patch_positions is not None

    def test_temporal_smoothing(self, sample_image, pixel_spacing):
        """Test temporal smoothing option."""
        deformation_field, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sidelength=32,
            temporal_smoothing=True,
            smoothing_window_size=3,
            device=torch.device("cpu"),
        )
        assert deformation_field.shape[0] == 2
        assert patch_positions is not None

    def test_outlier_rejection(self, sample_image, pixel_spacing):
        """Test outlier rejection option."""
        deformation_field, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sidelength=32,
            outlier_rejection=True,
            outlier_threshold=2.0,
            device=torch.device("cpu"),
        )
        assert deformation_field.shape[0] == 2
        assert patch_positions is not None

    def test_with_initial_deformation_field(self, sample_image, pixel_spacing):
        """Test with initial deformation field."""
        # Create a simple initial deformation field
        t = sample_image.shape[0]
        initial_field = torch.zeros((2, t, 1, 1))
        deformation_field, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sidelength=32,
            deformation_field=initial_field,
            device=torch.device("cpu"),
        )
        assert deformation_field.shape[0] == 2
        assert patch_positions is not None


class TestEstimateLocalMotion:
    """Tests for estimate_local_motion function."""

    def test_basic_functionality(self, sample_image, pixel_spacing):
        """Test basic local motion estimation with minimal iterations."""
        deformation_field = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_shape=(32, 32),
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            initial_deformation_field=None,
            device=torch.device("cpu"),
            n_iterations=2,  # Minimal iterations for testing
            optimizer_type="adam",
        )
        # Check output shape: (2, nt, nh, nw)
        assert deformation_field.shape == (2, sample_image.shape[0], 2, 2)
        assert isinstance(deformation_field, torch.Tensor)

    def test_with_initial_deformation_field(self, sample_image, pixel_spacing):
        """Test with initial deformation field."""
        initial_field = torch.zeros((2, sample_image.shape[0], 2, 2))
        deformation_field = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_shape=(32, 32),
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            initial_deformation_field=initial_field,
            device=torch.device("cpu"),
            n_iterations=2,
            optimizer_type="adam",
        )
        assert deformation_field.shape == (2, sample_image.shape[0], 2, 2)

    def test_different_optimizers(self, sample_image, pixel_spacing):
        """Test different optimizer types."""
        for optimizer_type in ["adam", "sgd"]:
            deformation_field = estimate_local_motion(
                image=sample_image,
                pixel_spacing=pixel_spacing,
                patch_shape=(32, 32),
                deformation_field_resolution=(sample_image.shape[0], 2, 2),
                initial_deformation_field=None,
                device=torch.device("cpu"),
                n_iterations=2,
                optimizer_type=optimizer_type,
            )
            assert deformation_field.shape == (2, sample_image.shape[0], 2, 2)

    def test_different_grid_types(self, sample_image, pixel_spacing):
        """Test different grid types."""
        for grid_type in ["catmull_rom", "bspline"]:
            deformation_field = estimate_local_motion(
                image=sample_image,
                pixel_spacing=pixel_spacing,
                patch_shape=(32, 32),
                deformation_field_resolution=(sample_image.shape[0], 2, 2),
                initial_deformation_field=None,
                device=torch.device("cpu"),
                n_iterations=2,
                grid_type=grid_type,
            )
            assert deformation_field.shape == (2, sample_image.shape[0], 2, 2)

    def test_different_loss_types(self, sample_image, pixel_spacing):
        """Test different loss types."""
        for loss_type in ["mse", "ncc"]:
            deformation_field = estimate_local_motion(
                image=sample_image,
                pixel_spacing=pixel_spacing,
                patch_shape=(32, 32),
                deformation_field_resolution=(sample_image.shape[0], 2, 2),
                initial_deformation_field=None,
                device=torch.device("cpu"),
                n_iterations=2,
                loss_type=loss_type,
            )
            assert deformation_field.shape == (2, sample_image.shape[0], 2, 2)

    def test_return_trajectory(self, sample_image, pixel_spacing):
        """Test return_trajectory option."""
        deformation_field, trajectory = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_shape=(32, 32),
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            initial_deformation_field=None,
            device=torch.device("cpu"),
            n_iterations=2,
            return_trajectory=True,
        )
        assert deformation_field.shape == (2, sample_image.shape[0], 2, 2)
        assert trajectory is not None

    def test_optimizer_kwargs(self, sample_image, pixel_spacing):
        """Test custom optimizer kwargs."""
        deformation_field = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_shape=(32, 32),
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            initial_deformation_field=None,
            device=torch.device("cpu"),
            n_iterations=2,
            optimizer_type="adam",
            optimizer_kwargs={"lr": 0.001},
        )
        assert deformation_field.shape == (2, sample_image.shape[0], 2, 2)
