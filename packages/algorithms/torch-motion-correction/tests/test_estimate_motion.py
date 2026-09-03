"""Tests for motion estimation functions."""

import pytest
import torch

from torch_motion_correction.deformation_field import DeformationField
from torch_motion_correction.estimate_motion_optimizer import (
    _compute_loss,
    estimate_local_motion,
)
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

    @pytest.mark.parametrize(
        "reference_strategy", ["mean_except_current", "middle_frame"]
    )
    def test_different_devices(self, sample_image, pixel_spacing, reference_strategy):
        """Test that a CPU-resident image can be processed on a CUDA device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        assert sample_image.device.type == "cpu"
        result, patch_positions = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            reference_strategy=reference_strategy,
            device=torch.device("cuda"),
        )
        assert result.data.device.type == "cuda"
        assert patch_positions is not None
        assert sample_image.device.type == "cpu"

    def test_mean_except_current_matches_naive_leave_one_out(
        self, sample_image, pixel_spacing
    ):
        """The O(t) precomputed-sum reference must match a naive O(t^2) average.

        Guards the algebraic refactor that derives the leave-one-out reference
        via `(total - current) / (t - 1)` instead of re-summing the other
        `t - 1` frames' patches for every frame: computes the naive
        leave-one-out average directly (independent re-implementation) and
        compares its cross-correlation peak shifts against the function's
        output.
        """
        import einops
        from torch_grid_utils.patch_grid import patch_grid_lazy

        from torch_motion_correction.estimate_motion_xc import (
            _apply_sub_pixel_refinement,
        )
        from torch_motion_correction.utils import prepare_patch_filters

        patch_sampling = PatchSamplingConfig(patch_shape=(32, 32))
        refinement = XCRefinementConfig(
            temporal_smoothing=False, outlier_rejection=False
        )
        fourier_filter = FourierFilterConfig()

        result, _ = estimate_motion_cross_correlation_patches(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            patch_sampling=patch_sampling,
            reference_strategy="mean_except_current",
            refinement=refinement,
            fourier_filter=fourier_filter,
            device=torch.device("cpu"),
        )

        # Independent naive re-implementation: re-sum the other t - 1 frames'
        # patches for every frame instead of using the precomputed total.
        t, h, w = sample_image.shape
        ph, pw = patch_sampling.patch_shape
        hl, hu = int(0.25 * h), int(0.75 * h)
        wl, wu = int(0.25 * w), int(0.75 * w)
        norm_std, norm_mean = torch.std_mean(sample_image[:, hl:hu, wl:wu])

        lazy_patch_grid, data_patch_positions = patch_grid_lazy(
            images=sample_image,
            patch_shape=(1, ph, pw),
            patch_step=(1, *patch_sampling.patch_step),
            distribute_patches=patch_sampling.distribute_patches,
        )
        gh, gw = data_patch_positions.shape[1:3]
        mask, b_factor_envelope, bandpass = prepare_patch_filters(
            shape=(ph, pw),
            pixel_spacing=pixel_spacing,
            fourier_filter=fourier_filter,
            device=torch.device("cpu"),
        )
        combined_filter = bandpass * b_factor_envelope

        def get_patches(idx: int) -> torch.Tensor:
            patches = lazy_patch_grid[idx]
            patches = einops.rearrange(patches, "1 gh gw 1 ph pw -> gh gw ph pw")
            return (patches - norm_mean) / norm_std

        expected_field = torch.zeros((2, t, gh, gw))
        for frame_idx in range(t):
            frame_patches = get_patches(frame_idx)
            others = [get_patches(i) for i in range(t) if i != frame_idx]
            ref_patches = torch.stack(others, dim=0).sum(dim=0) / (t - 1)

            ref_fft = torch.fft.rfftn(ref_patches * mask, dim=(-2, -1))
            ref_fft = ref_fft * combined_filter
            frame_fft = torch.fft.rfftn(frame_patches * mask, dim=(-2, -1))
            frame_fft = frame_fft * combined_filter

            cross_corr = torch.fft.irfftn(torch.conj(ref_fft) * frame_fft, s=(ph, pw))
            cross_corr_flat = cross_corr.view(gh * gw, ph * pw)
            peak_indices = torch.argmax(cross_corr_flat, dim=1)
            peak_y, peak_x = _apply_sub_pixel_refinement(
                cross_corr_flat, peak_indices, ph, pw
            )
            shift_y = torch.where(peak_y <= ph // 2, peak_y, peak_y - ph)
            shift_x = torch.where(peak_x <= pw // 2, peak_x, peak_x - pw)
            expected_field[0, frame_idx] = shift_y.view(gh, gw) * pixel_spacing
            expected_field[1, frame_idx] = shift_x.view(gh, gw) * pixel_spacing

        expected_field = expected_field - torch.mean(expected_field)
        assert torch.allclose(result.data, expected_field, atol=1e-4)


class TestEstimateLocalMotion:
    """Tests for estimate_local_motion function."""

    def test_basic_functionality(self, sample_image, pixel_spacing):
        """Test basic local motion estimation with minimal iterations."""
        result, _ = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            optimization=OptimizationConfig(max_iterations=2),
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
            optimization=OptimizationConfig(max_iterations=2),
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
                    max_iterations=2, optimizer_type=optimizer_type
                ),
                device=torch.device("cpu"),
            )
            assert result.shape == (2, sample_image.shape[0], 2, 2)

    def test_lbfgs_optimizer(self, sample_image, pixel_spacing):
        """Test the LBFGS optimizer path (uses a separate closure/batching path)."""
        result, _ = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            optimization=OptimizationConfig(max_iterations=2, optimizer_type="lbfgs"),
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
                optimization=OptimizationConfig(max_iterations=2, grid_type=grid_type),
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
                optimization=OptimizationConfig(max_iterations=2, loss_type=loss_type),
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
                max_iterations=2,
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
            optimization=OptimizationConfig(max_iterations=2),
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
            optimization=OptimizationConfig(max_iterations=2),
            device=torch.device("cpu"),
        )
        assert isinstance(result, DeformationField)


class TestComputeLossIrfftLinearity:
    """Regression tests for the single-irfft ncc/cc path in ``_compute_loss``."""

    @staticmethod
    def _make_batch(b, t, ph, pw, seed):
        torch.manual_seed(seed)
        shifted = torch.randn(b, t, ph, pw // 2 + 1, dtype=torch.complex64)
        total_sum = shifted.sum(dim=1, keepdim=True)
        reference = (total_sum - shifted) / (t - 1)
        return shifted, reference

    @pytest.mark.parametrize("loss_type", ["ncc", "cc"])
    def test_fast_path_matches_fallback(self, loss_type):
        b, t, ph, pw = 4, 6, 32, 32
        shifted, reference = self._make_batch(b, t, ph, pw, seed=0)

        fast = _compute_loss(shifted, reference, ph, pw, loss_type=loss_type, t=t)
        fallback = _compute_loss(
            shifted, reference, ph, pw, loss_type=loss_type, t=None
        )
        torch.testing.assert_close(fast, fallback, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("loss_type", ["ncc", "cc"])
    def test_single_frame_edge_case(self, loss_type):
        # t == 1: reference_patches == shifted_patches, both code paths must agree.
        b, ph, pw = 4, 32, 32
        torch.manual_seed(1)
        shifted = torch.randn(b, 1, ph, pw // 2 + 1, dtype=torch.complex64)
        reference = shifted

        with_t = _compute_loss(shifted, reference, ph, pw, loss_type=loss_type, t=1)
        without_t = _compute_loss(
            shifted, reference, ph, pw, loss_type=loss_type, t=None
        )
        torch.testing.assert_close(with_t, without_t, atol=1e-5, rtol=1e-5)

    def test_gradients_flow_through_fast_path(self):
        b, t, ph, pw = 4, 6, 32, 32
        shifted, _ = self._make_batch(b, t, ph, pw, seed=2)
        shifted.requires_grad_(True)
        total_sum = shifted.sum(dim=1, keepdim=True)
        reference = (total_sum - shifted) / (t - 1)

        loss = _compute_loss(shifted, reference, ph, pw, loss_type="ncc", t=t)
        loss.backward()

        assert shifted.grad is not None
        assert torch.isfinite(shifted.grad).all()
        assert shifted.grad.abs().sum() > 0


class TestFourierResolutionCropping:
    """Regression tests for cropping patches to the bandpass's resolution cutoff."""

    def test_crop_applied_by_default_still_converges(self, sample_image, pixel_spacing):
        """Default FourierFilterConfig (frequency_range=(300, 10)) at
        pixel_spacing=1.0 triggers cropping (crop spacing 5.0 > 1.0). Optimization
        should still reduce loss on an image with injected motion.
        """
        result, tracker = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            optimization=OptimizationConfig(
                max_iterations=15, loss_type="ncc", optimizer_kwargs={"lr": 0.05}
            ),
            device=torch.device("cpu"),
        )
        assert result.shape == (2, sample_image.shape[0], 2, 2)
        losses = [cp.loss for cp in tracker.checkpoints]
        assert losses[-1] < losses[0]

    def test_crop_skipped_when_not_beneficial(self, sample_image, pixel_spacing):
        """A tight frequency_range whose resolution cutoff is finer than the
        native pixel spacing must not trigger cropping (crop_pixel_spacing <
        pixel_spacing), and optimization should still work.
        """
        result, _ = estimate_local_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            deformation_field_resolution=(sample_image.shape[0], 2, 2),
            patch_sampling=PatchSamplingConfig(patch_shape=(32, 32)),
            fourier_filter=FourierFilterConfig(frequency_range=(300, 1.0)),
            optimization=OptimizationConfig(max_iterations=2),
            device=torch.device("cpu"),
        )
        assert result.shape == (2, sample_image.shape[0], 2, 2)

    def test_cropped_and_uncropped_give_similar_final_field(
        self, sample_image, pixel_spacing
    ):
        """Sanity check that cropping doesn't change *what* is being optimized:
        a run with cropping disabled (tight frequency_range) and a run with
        cropping enabled (default range) should converge to roughly the same
        deformation field on the same synthetic motion.
        """
        torch.manual_seed(0)
        common_kwargs = {
            "image": sample_image,
            "pixel_spacing": pixel_spacing,
            "deformation_field_resolution": (sample_image.shape[0], 2, 2),
            "patch_sampling": PatchSamplingConfig(patch_shape=(32, 32)),
            "optimization": OptimizationConfig(
                max_iterations=25, loss_type="ncc", optimizer_kwargs={"lr": 0.05}
            ),
            "device": torch.device("cpu"),
        }

        torch.manual_seed(0)
        result_cropped, _ = estimate_local_motion(
            fourier_filter=FourierFilterConfig(frequency_range=(300, 10)),
            **common_kwargs,
        )
        torch.manual_seed(0)
        result_uncropped, _ = estimate_local_motion(
            fourier_filter=FourierFilterConfig(frequency_range=(300, 1.0)),
            **common_kwargs,
        )

        # Both should recover a similar overall shift pattern; loose tolerance
        # since the two runs operate at different effective resolutions.
        torch.testing.assert_close(
            result_cropped.data, result_uncropped.data, atol=0.5, rtol=0.5
        )
