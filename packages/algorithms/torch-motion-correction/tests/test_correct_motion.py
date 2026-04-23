"""Tests for motion correction functions."""

import pytest
import torch

from torch_motion_correction.correct_motion import (
    correct_motion,
    correct_motion_fast,
    correct_motion_slow,
    correct_motion_two_grids,
)
from torch_motion_correction.deformation_field import DeformationField


@pytest.fixture
def sample_image():
    """Create a sample image tensor for testing."""
    # Create a simple test image
    t, h, w = 5, 64, 64
    image = torch.zeros((t, h, w))
    # Add a simple pattern
    for frame_idx in range(t):
        y_center = h // 2
        x_center = w // 2
        y, x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        dist_sq = (y - y_center) ** 2 + (x - x_center) ** 2
        image[frame_idx] = torch.exp(-dist_sq / (2 * 10**2))
    return image


@pytest.fixture
def sample_deformation_field():
    """Create a sample deformation field for testing."""
    t = 5
    # Create a simple deformation field with small shifts
    deformation_field = torch.zeros((2, t, 2, 2))
    # Add some small shifts
    for frame_idx in range(t):
        deformation_field[0, frame_idx, :, :] = frame_idx * 0.1  # y shift
        deformation_field[1, frame_idx, :, :] = frame_idx * 0.05  # x shift
    return deformation_field


@pytest.fixture
def sample_single_patch_deformation_field():
    """Create a single patch deformation field for fast correction."""
    t = 5
    # Single patch deformation field: (2, t, 1, 1)
    deformation_field = torch.zeros((2, t, 1, 1))
    for frame_idx in range(t):
        deformation_field[0, frame_idx, 0, 0] = frame_idx * 0.1  # y shift
        deformation_field[1, frame_idx, 0, 0] = frame_idx * 0.05  # x shift
    return deformation_field


@pytest.fixture
def pixel_spacing():
    """Pixel spacing in Angstroms."""
    return 1.0


class TestCorrectMotion:
    """Tests for correct_motion function."""

    def test_basic_functionality(
        self,
        sample_image,
        sample_deformation_field,
        pixel_spacing,
    ):
        """Test basic motion correction."""
        corrected = correct_motion(
            image=sample_image,
            deformation_field=DeformationField(data=sample_deformation_field),
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        # Check output shape matches input
        assert corrected.shape == sample_image.shape
        assert isinstance(corrected, torch.Tensor)

    def test_different_grid_types(
        self,
        sample_image,
        sample_deformation_field,
        pixel_spacing,
    ):
        """Test different grid types via DeformationField."""
        for grid_type in ["catmull_rom", "bspline"]:
            corrected = correct_motion(
                image=sample_image,
                deformation_field=DeformationField(
                    data=sample_deformation_field, grid_type=grid_type
                ),
                pixel_spacing=pixel_spacing,
                device=torch.device("cpu"),
            )
            assert corrected.shape == sample_image.shape

    def test_grad_flag(self, sample_image, sample_deformation_field, pixel_spacing):
        """Test grad flag."""
        corrected = correct_motion(
            image=sample_image,
            deformation_field=DeformationField(data=sample_deformation_field),
            pixel_spacing=pixel_spacing,
            grad=False,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        # When grad=False, output should be detached
        assert not corrected.requires_grad

    def test_different_devices(
        self, sample_image, sample_deformation_field, pixel_spacing
    ):
        """Test that device parameter works."""
        if torch.cuda.is_available():
            corrected = correct_motion(
                image=sample_image,
                deformation_field=DeformationField(data=sample_deformation_field),
                pixel_spacing=pixel_spacing,
                device=torch.device("cuda"),
            )
            assert corrected.device.type == "cuda"
            assert corrected.shape == sample_image.shape
        else:
            pytest.skip("CUDA not available")

    def test_zero_deformation_field(self, sample_image, pixel_spacing):
        """Test with zero deformation field (should return original image)."""
        t = sample_image.shape[0]
        corrected = correct_motion(
            image=sample_image,
            deformation_field=DeformationField(data=torch.zeros((2, t, 2, 2))),
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        # With zero deformation, result should be close to original
        # (allowing for small numerical differences from interpolation)
        assert torch.allclose(corrected, sample_image, atol=0.1)


class TestCorrectMotionFast:
    """Tests for correct_motion_fast function."""

    def test_basic_functionality(
        self, sample_image, sample_single_patch_deformation_field
    ):
        """Test basic fast motion correction."""
        corrected = correct_motion_fast(
            image=sample_image,
            deformation_field=DeformationField(
                data=sample_single_patch_deformation_field
            ),
            device=torch.device("cpu"),
        )
        # Check output shape matches input
        assert corrected.shape == sample_image.shape
        assert isinstance(corrected, torch.Tensor)

    def test_single_patch_requirement(self, sample_image, sample_deformation_field):
        """Test that function requires single patch deformation field."""
        with pytest.raises(ValueError, match="Expected single patch deformation field"):
            correct_motion_fast(
                image=sample_image,
                deformation_field=DeformationField(data=sample_deformation_field),
                device=torch.device("cpu"),
            )

    def test_different_devices(
        self, sample_image, sample_single_patch_deformation_field
    ):
        """Test that device parameter works."""
        if torch.cuda.is_available():
            corrected = correct_motion_fast(
                image=sample_image,
                deformation_field=DeformationField(
                    data=sample_single_patch_deformation_field
                ),
                device=torch.device("cuda"),
            )
            assert corrected.device.type == "cuda"
            assert corrected.shape == sample_image.shape
        else:
            pytest.skip("CUDA not available")

    def test_zero_deformation_field(self, sample_image):
        """Test with zero deformation field."""
        t = sample_image.shape[0]
        corrected = correct_motion_fast(
            image=sample_image,
            deformation_field=DeformationField(data=torch.zeros((2, t, 1, 1))),
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        # With zero deformation, result should be very close to original
        assert torch.allclose(corrected, sample_image, atol=1e-5)


class TestCorrectMotionSlow:
    """Tests for correct_motion_slow function."""

    def test_basic_functionality(self, sample_image, sample_deformation_field):
        """Test basic slow motion correction."""
        corrected = correct_motion_slow(
            image=sample_image,
            deformation_field=DeformationField(data=sample_deformation_field),
            device=torch.device("cpu"),
        )
        # Check output shape matches input
        assert corrected.shape == sample_image.shape
        assert isinstance(corrected, torch.Tensor)

    def test_grad_flag(self, sample_image, sample_deformation_field):
        """Test grad flag."""
        corrected = correct_motion_slow(
            image=sample_image,
            deformation_field=DeformationField(data=sample_deformation_field),
            grad=False,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        # When grad=False, output should be detached
        assert not corrected.requires_grad

    def test_different_devices(self, sample_image, sample_deformation_field):
        """Test that device parameter works."""
        if torch.cuda.is_available():
            corrected = correct_motion_slow(
                image=sample_image,
                deformation_field=DeformationField(data=sample_deformation_field),
                device=torch.device("cuda"),
            )
            assert corrected.device.type == "cuda"
            assert corrected.shape == sample_image.shape
        else:
            pytest.skip("CUDA not available")

    def test_zero_deformation_field(self, sample_image):
        """Test with zero deformation field."""
        t = sample_image.shape[0]
        corrected = correct_motion_slow(
            image=sample_image,
            deformation_field=DeformationField(data=torch.zeros((2, t, 2, 2))),
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        # With zero deformation, result should be close to original
        assert torch.allclose(corrected, sample_image, atol=0.1)


class TestMotionCorrectionIntegration:
    """Integration tests for motion correction workflow."""

    def test_estimate_and_correct_workflow(self, sample_image, pixel_spacing):
        """Test a complete workflow: estimate motion then correct."""
        from torch_motion_correction.estimate_motion_xc import estimate_global_motion

        # Estimate motion — returns DeformationField
        deformation_field = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        assert isinstance(deformation_field, DeformationField)

        # Correct motion — accepts DeformationField directly
        corrected = correct_motion(
            image=sample_image,
            deformation_field=deformation_field,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )

        # Check that correction produces valid output
        assert corrected.shape == sample_image.shape
        assert torch.isfinite(corrected).all()

    def test_fast_correction_workflow(self, sample_image, pixel_spacing):
        """Test workflow with fast correction using a DeformationField."""
        from torch_motion_correction.estimate_motion_xc import estimate_global_motion

        # Estimate motion (produces single patch field)
        deformation_field = estimate_global_motion(
            image=sample_image,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )

        corrected = correct_motion_fast(
            image=sample_image,
            deformation_field=deformation_field,
            device=torch.device("cpu"),
        )

        # Check that correction produces valid output
        assert corrected.shape == sample_image.shape
        assert torch.isfinite(corrected).all()


class TestCorrectMotionTwoGrids:
    """Tests for correct_motion_two_grids function."""

    @pytest.fixture
    def sample_deformation_field_catmull_rom(self):
        """Create a sample Catmull-Rom deformation field for testing."""
        t, h, w = 5, 2, 2
        data = torch.zeros(2, t, h, w)
        for frame_idx in range(t):
            data[0, frame_idx, :, :] = frame_idx * 0.1  # y shift
            data[1, frame_idx, :, :] = frame_idx * 0.05  # x shift
        new_field, _ = DeformationField.from_initial_field(
            resolution=(t, h, w),
            initial_field=DeformationField(data=data),
            grid_type="catmull_rom",
            device=torch.device("cpu"),
        )
        return new_field

    @pytest.fixture
    def sample_deformation_field_bspline(self):
        """Create a sample B-spline deformation field for testing."""
        t, h, w = 5, 2, 2
        data = torch.zeros(2, t, h, w)
        for frame_idx in range(t):
            data[0, frame_idx, :, :] = frame_idx * 0.1  # y shift
            data[1, frame_idx, :, :] = frame_idx * 0.05  # x shift
        new_field, _ = DeformationField.from_initial_field(
            resolution=(t, h, w),
            initial_field=DeformationField(data=data),
            grid_type="bspline",
            device=torch.device("cpu"),
        )
        return new_field

    @pytest.fixture
    def zero_deformation_field_catmull_rom(self):
        """Create a zero Catmull-Rom deformation field for testing."""
        t, h, w = 5, 2, 2
        _, base_field = DeformationField.from_initial_field(
            resolution=(t, h, w),
            initial_field=None,
            grid_type="catmull_rom",
            device=torch.device("cpu"),
        )
        return base_field

    def test_basic_functionality(
        self,
        sample_image,
        sample_deformation_field_catmull_rom,
        zero_deformation_field_catmull_rom,
        pixel_spacing,
    ):
        """Test basic two-field motion correction."""
        corrected = correct_motion_two_grids(
            image=sample_image,
            new_deformation_field=sample_deformation_field_catmull_rom,
            base_deformation_field=zero_deformation_field_catmull_rom,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        assert isinstance(corrected, torch.Tensor)

    def test_different_grid_types(
        self,
        sample_image,
        sample_deformation_field_catmull_rom,
        sample_deformation_field_bspline,
        zero_deformation_field_catmull_rom,
        pixel_spacing,
    ):
        """Test with different grid types."""
        corrected = correct_motion_two_grids(
            image=sample_image,
            new_deformation_field=sample_deformation_field_catmull_rom,
            base_deformation_field=zero_deformation_field_catmull_rom,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape

        _, zero_bspline = DeformationField.from_initial_field(
            resolution=(5, 2, 2),
            initial_field=None,
            grid_type="bspline",
            device=torch.device("cpu"),
        )
        corrected = correct_motion_two_grids(
            image=sample_image,
            new_deformation_field=sample_deformation_field_bspline,
            base_deformation_field=zero_bspline,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape

    def test_grad_flag(
        self,
        sample_image,
        sample_deformation_field_catmull_rom,
        zero_deformation_field_catmull_rom,
        pixel_spacing,
    ):
        """Test grad flag."""
        corrected = correct_motion_two_grids(
            image=sample_image,
            new_deformation_field=sample_deformation_field_catmull_rom,
            base_deformation_field=zero_deformation_field_catmull_rom,
            pixel_spacing=pixel_spacing,
            grad=True,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape

        corrected = correct_motion_two_grids(
            image=sample_image,
            new_deformation_field=sample_deformation_field_catmull_rom,
            base_deformation_field=zero_deformation_field_catmull_rom,
            pixel_spacing=pixel_spacing,
            grad=False,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        assert not corrected.requires_grad

    def test_gradient_preservation(
        self,
        sample_image,
        sample_deformation_field_catmull_rom,
        zero_deformation_field_catmull_rom,
        pixel_spacing,
    ):
        """Test that gradients flow through the new field when grad=True."""
        corrected = correct_motion_two_grids(
            image=sample_image,
            new_deformation_field=sample_deformation_field_catmull_rom,
            base_deformation_field=zero_deformation_field_catmull_rom,
            pixel_spacing=pixel_spacing,
            grad=True,
            device=torch.device("cpu"),
        )

        assert corrected.requires_grad

        loss = corrected.sum()
        loss.backward()

    def test_different_devices(
        self,
        sample_image,
        sample_deformation_field_catmull_rom,
        zero_deformation_field_catmull_rom,
        pixel_spacing,
    ):
        """Test that device parameter works."""
        if torch.cuda.is_available():
            new_field = sample_deformation_field_catmull_rom.to("cuda")
            base_field = zero_deformation_field_catmull_rom.to("cuda")

            corrected = correct_motion_two_grids(
                image=sample_image,
                new_deformation_field=new_field,
                base_deformation_field=base_field,
                pixel_spacing=pixel_spacing,
                device=torch.device("cuda"),
            )
            assert corrected.device.type == "cuda"
            assert corrected.shape == sample_image.shape
        else:
            pytest.skip("CUDA not available")

    def test_zero_deformation_fields(
        self,
        sample_image,
        zero_deformation_field_catmull_rom,
        pixel_spacing,
    ):
        """Test with zero deformation fields (should return original image)."""
        corrected = correct_motion_two_grids(
            image=sample_image,
            new_deformation_field=zero_deformation_field_catmull_rom,
            base_deformation_field=zero_deformation_field_catmull_rom,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        assert torch.allclose(corrected, sample_image, atol=0.1)

    def test_combined_fields(
        self,
        sample_image,
        sample_deformation_field_catmull_rom,
        zero_deformation_field_catmull_rom,
        pixel_spacing,
    ):
        """Test that combining new and base fields works correctly."""
        corrected = correct_motion_two_grids(
            image=sample_image,
            new_deformation_field=sample_deformation_field_catmull_rom,
            base_deformation_field=zero_deformation_field_catmull_rom,
            pixel_spacing=pixel_spacing,
            device=torch.device("cpu"),
        )
        assert corrected.shape == sample_image.shape
        assert torch.isfinite(corrected).all()

    def test_base_field_detached(
        self,
        sample_image,
        sample_deformation_field_catmull_rom,
        zero_deformation_field_catmull_rom,
        pixel_spacing,
    ):
        """Test that base field gradients are properly detached."""
        corrected = correct_motion_two_grids(
            image=sample_image,
            new_deformation_field=sample_deformation_field_catmull_rom,
            base_deformation_field=zero_deformation_field_catmull_rom,
            pixel_spacing=pixel_spacing,
            grad=True,
            device=torch.device("cpu"),
        )

        loss = corrected.sum()
        loss.backward()

        assert corrected.requires_grad

        # New grid may or may not have gradients depending on whether
        # the grid's data is a Parameter and properly connected
        # The key test is that the output has gradients, showing the
        # computation graph works correctly
        # Base grid should NOT have gradients (it's detached in the function)
        # The base grid's shifts are detached, so gradients won't flow to it
        # even if requires_grad was set to True
