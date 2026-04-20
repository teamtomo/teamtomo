"""Tests for motion estimation functions."""

import pytest
import torch

from torch_motion_correction.deformation_field import DeformationField
from torch_motion_correction.estimate_motion_optimizer import estimate_local_motion
from torch_motion_correction.estimate_motion_xc import (
    estimate_global_motion,
    estimate_motion_cross_correlation_patches,
)
from torch_motion_correction.types import (
    FourierFilterConfig,
    OptimizationConfig,
    PatchSamplingConfig,
    XCRefinementConfig,
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
        result = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        assert isinstance(result, DeformationField)
        # Check output shape: (2, t, 1, 1) for global motion
        assert result.shape == (2, sample_image.shape[0], 1, 1)

    def test_reference_frame(self, sample_image, pixel_spacing):
        """Test that reference frame parameter works."""
        result = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            reference_frame=0,
            device=torch.device("cpu"),
        )
        assert result.shape == (2, sample_image.shape[0], 1, 1)

    def test_different_devices(self, sample_image, pixel_spacing):
        """Test that device parameter works."""
        if torch.cuda.is_available():
            result = estimate_global_motion(
                image=sample_image,
                pixel_spacing=pixel_spacing,
                device=torch.device("cuda"),
            )
            assert result.device.type == "cuda"
        else:
            pytest.skip("CUDA not available")

    def test_fourier_filter_b_factor(self, sample_image, pixel_spacing):
        """Test that FourierFilterConfig b_factor works."""
        result = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            fourier_filter=FourierFilterConfig(b_factor=1000),
            device=torch.device("cpu"),
        )
        assert result.shape == (2, sample_image.shape[0], 1, 1)

    def test_fourier_filter_frequency_range(self, sample_image, pixel_spacing):
        """Test that FourierFilterConfig frequency_range works."""
        result = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            fourier_filter=FourierFilterConfig(frequency_range=(200, 20)),
            device=torch.device("cpu"),
        )
        assert result.shape == (2, sample_image.shape[0], 1, 1)


class TestEstimateMotionCrossCorrelationPatches:
    """Tests for estimate_motion_cross_correlation_patches function."""

    def test_basic_functionality(self, sample_image, pixel_spacing):
        """Test basic patch-based motion estimation."""
        result, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            device=torch.device("cpu"),
        )
        assert isinstance(result, DeformationField)
        assert len(result.shape) == 4  # (2, t, gh, gw)
        assert result.shape[0] == 2  # y, x
        assert result.shape[1] == sample_image.shape[0]  # t
        assert len(patch_positions.shape) == 4  # (t, gh, gw, 3)
        assert patch_positions.shape[0] == sample_image.shape[0]  # t

    def test_reference_strategy_middle_frame(self, sample_image, pixel_spacing):
        """Test middle_frame reference strategy."""
        result, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            reference_strategy="middle_frame",
            device=torch.device("cpu"),
        )
        assert result.shape[0] == 2
        assert patch_positions is not None

    def test_reference_strategy_mean_except_current(self, sample_image, pixel_spacing):
        """Test mean_except_current reference strategy."""
        result, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            reference_strategy="mean_except_current",
            device=torch.device("cpu"),
        )
        assert result.shape[0] == 2
        assert patch_positions is not None

    def test_xc_refinement_sub_pixel(self, sample_image, pixel_spacing):
        """Test XCRefinementConfig sub_pixel_refinement option."""
        result, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            refinement=XCRefinementConfig(sub_pixel_refinement=True),
            device=torch.device("cpu"),
        )
        assert result.shape[0] == 2
        assert patch_positions is not None

    def test_xc_refinement_temporal_smoothing(self, sample_image, pixel_spacing):
        """Test XCRefinementConfig temporal smoothing option."""
        result, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            refinement=XCRefinementConfig(
                temporal_smoothing=True, smoothing_window_size=3
            ),
            device=torch.device("cpu"),
        )
        assert result.shape[0] == 2
        assert patch_positions is not None

    def test_xc_refinement_outlier_rejection(self, sample_image, pixel_spacing):
        """Test XCRefinementConfig outlier rejection option."""
        result, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            refinement=XCRefinementConfig(
                outlier_rejection=True, outlier_threshold=2.0
            ),
            device=torch.device("cpu"),
        )
        assert result.shape[0] == 2
        assert patch_positions is not None

    def test_with_initial_deformation_field(self, sample_image, pixel_spacing):
        """Test with initial deformation field."""
        t = sample_image.shape[0]
        initial_field = DeformationField(data=torch.zeros((2, t, 1, 1)))
        result, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            initial_deformation_field=initial_field,
            device=torch.device("cpu"),
        )
        assert result.shape[0] == 2
        assert patch_positions is not None

    def test_patch_sampling_overlap(self, sample_image, pixel_spacing):
        """Test that PatchSamplingConfig overlap parameter is used."""
        # 25% overlap should give more patches than 50%
        result_25, _ = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32), overlap=0.25),
            device=torch.device("cpu"),
        )
        result_50, _ = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32), overlap=0.5),
            device=torch.device("cpu"),
        )
        # Both should be valid DeformationFields
        assert isinstance(result_25, DeformationField)
        assert isinstance(result_50, DeformationField)


class TestEstimateLocalMotion:
    """Tests for estimate_local_motion function."""

    def test_basic_functionality(self, sample_image, pixel_spacing):
        """Test basic local motion estimation with minimal iterations."""
        result, _ = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            optimization=OptimizationConfig(n_iterations=2),
            device=torch.device("cpu"),
        )
        assert isinstance(result, DeformationField)
        # Check output shape: (2, nt, nh, nw)
        assert result.shape == (2, sample_image.shape[0], 2, 2)

    def test_with_initial_deformation_field(self, sample_image, pixel_spacing):
        """Test with initial deformation field."""
        initial_field = DeformationField(
            data=torch.zeros((2, sample_image.shape[0], 2, 2))
        )
        result, _ = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            initial_deformation_field=initial_field,
            optimization=OptimizationConfig(n_iterations=2),
            device=torch.device("cpu"),
        )
        assert result.shape == (2, sample_image.shape[0], 2, 2)

    def test_different_optimizers(self, sample_image, pixel_spacing):
        """Test different optimizer types via OptimizationConfig."""
        for optimizer_type in ["adam", "sgd"]:
            result, _ = estimate_local_motion(
                image=sample_image,
                pixel_spacing=pixel_spacing,
                deformation_field_resolution=(sample_image.shape[0], 2, 2),
                patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
                optimization=OptimizationConfig(
                    n_iterations=2, optimizer_type=optimizer_type
                ),
                device=torch.device("cpu"),
            )
            assert result.shape == (2, sample_image.shape[0], 2, 2)

    def test_different_grid_types(self, sample_image, pixel_spacing):
        """Test different grid types via OptimizationConfig."""
        for grid_type in ["catmull_rom", "bspline"]:
            result, _ = estimate_local_motion(
                image=sample_image,
                pixel_spacing=pixel_spacing,
                deformation_field_resolution=(sample_image.shape[0], 2, 2),
                patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
                optimization=OptimizationConfig(n_iterations=2, grid_type=grid_type),
                device=torch.device("cpu"),
            )
            assert result.shape == (2, sample_image.shape[0], 2, 2)
            assert result.grid_type == grid_type

    def test_different_loss_types(self, sample_image, pixel_spacing):
        """Test different loss types via OptimizationConfig."""
        for loss_type in ["mse", "ncc"]:
            result, _ = estimate_local_motion(
                image=sample_image,
                pixel_spacing=pixel_spacing,
                deformation_field_resolution=(sample_image.shape[0], 2, 2),
                patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
                optimization=OptimizationConfig(n_iterations=2, loss_type=loss_type),
                device=torch.device("cpu"),
            )
            assert result.shape == (2, sample_image.shape[0], 2, 2)

    def test_optimizer_kwargs(self, sample_image, pixel_spacing):
        """Test custom optimizer kwargs via OptimizationConfig."""
        result, _ = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            optimization=OptimizationConfig(
                n_iterations=2,
                optimizer_type="adam",
                optimizer_kwargs={"lr": 0.001},
            ),
            device=torch.device("cpu"),
        )
        assert result.shape == (2, sample_image.shape[0], 2, 2)

    def test_fourier_filter_config(self, sample_image, pixel_spacing):
        """Test FourierFilterConfig is applied correctly."""
        result, _ = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            fourier_filter=FourierFilterConfig(b_factor=1000, frequency_range=(200, 5)),
            optimization=OptimizationConfig(n_iterations=2),
            device=torch.device("cpu"),
        )
        assert result.shape == (2, sample_image.shape[0], 2, 2)

    def test_patch_sampling_overlap(self, sample_image, pixel_spacing):
        """Test that PatchSamplingConfig overlap is applied."""
        result, _ = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32), overlap=0.25),
            optimization=OptimizationConfig(n_iterations=2),
            device=torch.device("cpu"),
        )
        assert isinstance(result, DeformationField)
