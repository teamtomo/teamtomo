import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torch_fourier_rescale import fourier_rescale_2d, fourier_rescale_3d


def test_fourier_upscale_2d(circle):
    rescaled, new_spacing = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=0.5
    )
    assert tuple(circle.shape) == (28, 28)
    assert tuple(rescaled.shape) == (56, 56)

    # test upscale with uneven image
    rescaled, new_spacing = fourier_rescale_2d(
        image=F.pad(circle, (0, 1, 0, 1)), source_spacing=1, target_spacing=0.5
    )
    assert tuple(rescaled.shape) == (58, 58)


def test_fourier_downscale_2d(circle):
    rescaled, new_spacing = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=2
    )
    assert tuple(circle.shape) == (28, 28)
    assert tuple(rescaled.shape) == (14, 14)

    # test downscale with uneven image
    rescaled, new_spacing = fourier_rescale_2d(
        image=F.pad(circle, (0, 1, 0, 1)), source_spacing=1, target_spacing=2
    )
    assert tuple(rescaled.shape) == (14, 14)


def test_fourier_upscale_3d(sphere):
    rescaled, new_spacing = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=0.5
    )
    assert tuple(sphere.shape) == (28, 28, 28)
    assert tuple(rescaled.shape) == (56, 56, 56)

    # test upscale with uneven box
    rescaled, new_spacing = fourier_rescale_3d(
        image=F.pad(sphere, (0, 1, 0, 1, 0, 1)), source_spacing=1, target_spacing=0.5
    )
    assert tuple(rescaled.shape) == (58, 58, 58)


def test_fourier_downscale_3d(sphere):
    rescaled, new_spacing = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=2
    )
    assert tuple(sphere.shape) == (28, 28, 28)
    assert tuple(rescaled.shape) == (14, 14, 14)

    # test downscale with uneven box
    rescaled, new_spacing = fourier_rescale_3d(
        image=F.pad(sphere, (0, 1, 0, 1, 0, 1)), source_spacing=1, target_spacing=2
    )
    assert tuple(rescaled.shape) == (14, 14, 14)


@pytest.mark.parametrize("spacing", [0.5, 2.0])
def test_fourier_rescale_2d_mean(circle, spacing):
    rescaled, new_spacing = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=spacing
    )
    assert rescaled.mean() == pytest.approx(circle.mean())

    rescaled, new_spacing = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=spacing, preserve_mean=False
    )
    assert rescaled.mean() != pytest.approx(circle.mean())


@pytest.mark.parametrize("spacing", [0.5, 2.0])
def test_fourier_rescale_3d_mean(sphere, spacing):
    rescaled, new_spacing = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=spacing
    )
    assert rescaled.mean() == pytest.approx(sphere.mean())

    rescaled, new_spacing = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=spacing, preserve_mean=False
    )
    assert rescaled.mean() != pytest.approx(sphere.mean())


@pytest.mark.parametrize(
    "dtype", [int, float, np.float32, np.float64, torch.float32, torch.float64]
)
def test_pixel_spacing_scalar_dtypes(dtype, circle, sphere):
    # Smoke test - just verify functions don't crash with different scalar dtypes
    if dtype == int:
        source_spacing = 1
        target_spacing = 2
    elif dtype == float:
        source_spacing = 1.0
        target_spacing = 0.5
    elif dtype in [np.float32, np.float64]:
        source_spacing = dtype(1.0)
        target_spacing = dtype(0.5)
    else:  # torch dtypes
        source_spacing = torch.tensor(1.0, dtype=dtype).item()
        target_spacing = torch.tensor(0.5, dtype=dtype).item()

    # Test 2D with dtype for both source and target spacing
    rescaled_2d, new_spacing_2d = fourier_rescale_2d(
        image=circle, source_spacing=source_spacing, target_spacing=target_spacing
    )
    assert rescaled_2d is not None
    assert new_spacing_2d is not None

    # Test 3D with dtype for both source and target spacing
    rescaled_3d, new_spacing_3d = fourier_rescale_3d(
        image=sphere, source_spacing=source_spacing, target_spacing=target_spacing
    )
    assert rescaled_3d is not None
    assert new_spacing_3d is not None


def test_fourier_rescale_2d_target_shape(circle):
    # Test upscaling with target_shape
    rescaled, new_spacing = fourier_rescale_2d(image=circle, target_shape=(56, 56))
    assert tuple(circle.shape) == (28, 28)
    assert tuple(rescaled.shape) == (56, 56)
    assert new_spacing == pytest.approx((0.5, 0.5))

    # Test downscaling with target_shape
    rescaled, new_spacing = fourier_rescale_2d(image=circle, target_shape=(14, 14))
    assert tuple(rescaled.shape) == (14, 14)
    assert new_spacing == pytest.approx((2.0, 2.0))

    # Test with non-uniform target shape
    rescaled, new_spacing = fourier_rescale_2d(image=circle, target_shape=(56, 14))
    assert tuple(rescaled.shape) == (56, 14)
    assert new_spacing[0] == pytest.approx(0.5)
    assert new_spacing[1] == pytest.approx(2.0)


def test_fourier_rescale_3d_target_shape(sphere):
    # Test upscaling with target_shape
    rescaled, new_spacing = fourier_rescale_3d(image=sphere, target_shape=(56, 56, 56))
    assert tuple(sphere.shape) == (28, 28, 28)
    assert tuple(rescaled.shape) == (56, 56, 56)
    assert new_spacing == pytest.approx((0.5, 0.5, 0.5))

    # Test downscaling with target_shape
    rescaled, new_spacing = fourier_rescale_3d(image=sphere, target_shape=(14, 14, 14))
    assert tuple(rescaled.shape) == (14, 14, 14)
    assert new_spacing == pytest.approx((2.0, 2.0, 2.0))

    # Test with non-uniform target shape
    rescaled, new_spacing = fourier_rescale_3d(image=sphere, target_shape=(56, 14, 28))
    assert tuple(rescaled.shape) == (56, 14, 28)
    assert new_spacing[0] == pytest.approx(0.5)
    assert new_spacing[1] == pytest.approx(2.0)
    assert new_spacing[2] == pytest.approx(1.0)


def test_target_shape_vs_target_spacing_2d(circle):
    # Verify that using target_shape gives the same result as target_spacing
    rescaled_shape, spacing_from_shape = fourier_rescale_2d(
        image=circle, target_shape=(56, 56)
    )
    rescaled_spacing, spacing_from_spacing = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=0.5
    )

    assert torch.allclose(rescaled_shape, rescaled_spacing)
    assert spacing_from_shape == pytest.approx(spacing_from_spacing)


def test_target_shape_vs_target_spacing_3d(sphere):
    # Verify that using target_shape gives the same result as target_spacing
    rescaled_shape, spacing_from_shape = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_shape=(56, 56, 56)
    )
    rescaled_spacing, spacing_from_spacing = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=0.5
    )

    assert torch.allclose(rescaled_shape, rescaled_spacing)
    assert spacing_from_shape == pytest.approx(spacing_from_spacing)


def test_providing_target_spacing_and_shape_2d():
    # Test that specifying both target_shape and target_spacing raises an error
    image = torch.randn(28, 28)

    with pytest.raises(
        ValueError, match="Cannot specify both target_spacing and target_shape"
    ):
        fourier_rescale_2d(
            image=image, source_spacing=1, target_spacing=0.5, target_shape=(56, 56)
        )

    # Test that specifying target_spacing without source_spacing raises an error
    with pytest.raises(
        ValueError, match="source_spacing is required when target_spacing is specified"
    ):
        fourier_rescale_2d(image=image, target_spacing=0.5, target_shape=(56, 56))

    # Test that specifying neither raises an error
    with pytest.raises(ValueError, match="Either target_spacing or target_shape"):
        fourier_rescale_2d(image=image, source_spacing=1)


def test_providing_target_spacing_and_shape_3d():
    # Test that specifying both target_shape and target_spacing raises an error
    image = torch.randn(28, 28, 28)

    with pytest.raises(
        ValueError, match="Cannot specify both target_spacing and target_shape"
    ):
        fourier_rescale_3d(
            image=image, source_spacing=1, target_spacing=0.5, target_shape=(56, 56, 56)
        )

    # Test that specifying target_spacing without source_spacing raises an error
    with pytest.raises(
        ValueError, match="source_spacing is required when target_spacing is specified"
    ):
        fourier_rescale_3d(image=image, target_spacing=0.5, target_shape=(56, 56, 56))

    # Test that specifying neither raises an error
    with pytest.raises(ValueError, match="Either target_spacing or target_shape"):
        fourier_rescale_3d(image=image, source_spacing=1)


def test_preserve_mean_with_target_shape_2d(circle):
    # Test preserve_mean with target_shape
    rescaled, _ = fourier_rescale_2d(
        image=circle, target_shape=(56, 56), preserve_mean=True
    )
    assert rescaled.mean() == pytest.approx(circle.mean())

    rescaled_no_preserve, _ = fourier_rescale_2d(
        image=circle, target_shape=(56, 56), preserve_mean=False
    )
    assert rescaled_no_preserve.mean() != pytest.approx(circle.mean())
