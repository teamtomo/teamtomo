import numbers

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torch_fourier_rescale import (
    fourier_rescale_2d,
    fourier_rescale_3d,
    fourier_rescale_rfft_2d,
    fourier_rescale_rfft_3d,
)


def test_fourier_upscale_2d(circle):
    rescaled, _ = fourier_rescale_2d(image=circle, source_spacing=1, target_spacing=0.5)
    assert tuple(circle.shape) == (28, 28)
    assert tuple(rescaled.shape) == (56, 56)

    # test upscale with uneven image
    rescaled, _ = fourier_rescale_2d(
        image=F.pad(circle, (0, 1, 0, 1)), source_spacing=1, target_spacing=0.5
    )
    assert tuple(rescaled.shape) == (58, 58)


def test_fourier_downscale_2d(circle):
    rescaled, _ = fourier_rescale_2d(image=circle, source_spacing=1, target_spacing=2)
    assert tuple(circle.shape) == (28, 28)
    assert tuple(rescaled.shape) == (14, 14)

    # test downscale with uneven image
    rescaled, _ = fourier_rescale_2d(
        image=F.pad(circle, (0, 1, 0, 1)), source_spacing=1, target_spacing=2
    )
    assert tuple(rescaled.shape) == (14, 14)


def test_fourier_upscale_3d(sphere):
    rescaled, _ = fourier_rescale_3d(image=sphere, source_spacing=1, target_spacing=0.5)
    assert tuple(sphere.shape) == (28, 28, 28)
    assert tuple(rescaled.shape) == (56, 56, 56)

    # test upscale with uneven box
    rescaled, _ = fourier_rescale_3d(
        image=F.pad(sphere, (0, 1, 0, 1, 0, 1)), source_spacing=1, target_spacing=0.5
    )
    assert tuple(rescaled.shape) == (58, 58, 58)


def test_fourier_downscale_3d(sphere):
    rescaled, _ = fourier_rescale_3d(image=sphere, source_spacing=1, target_spacing=2)
    assert tuple(sphere.shape) == (28, 28, 28)
    assert tuple(rescaled.shape) == (14, 14, 14)

    # test downscale with uneven box
    rescaled, _ = fourier_rescale_3d(
        image=F.pad(sphere, (0, 1, 0, 1, 0, 1)), source_spacing=1, target_spacing=2
    )
    assert tuple(rescaled.shape) == (14, 14, 14)


@pytest.mark.parametrize("spacing", [0.5, 2.0])
def test_fourier_rescale_2d_mean(circle, spacing):
    rescaled, _ = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=spacing
    )
    assert rescaled.mean() == pytest.approx(circle.mean())

    rescaled, _ = fourier_rescale_2d(
        image=circle, source_spacing=1, target_spacing=spacing, preserve_mean=False
    )
    assert rescaled.mean() != pytest.approx(circle.mean())


@pytest.mark.parametrize("spacing", [0.5, 2.0])
def test_fourier_rescale_3d_mean(sphere, spacing):
    rescaled, _ = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=spacing
    )
    assert rescaled.mean() == pytest.approx(sphere.mean())

    rescaled, _ = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=spacing, preserve_mean=False
    )
    assert rescaled.mean() != pytest.approx(sphere.mean())


# Cover every parity combination of source / target / direction. Pre-fix, the
# 2D fast path collapsed the mean to ~0 when source and target had different
# parity (e.g. even source, odd target — the common case when rescaling a
# 4096^2 image to a non-power-of-two spacing). Regression test ensures the
# fast path keeps the DC component in the correct frequency bin for every
# parity combination.
@pytest.mark.parametrize(
    "source_shape,target_shape",
    [
        # 2D downscale: even -> even (works pre-fix)
        ((16, 16), (8, 8)),
        # 2D downscale: even -> odd (BROKEN pre-fix)
        ((16, 16), (7, 7)),
        # 2D downscale: odd -> even
        ((15, 15), (8, 8)),
        # 2D downscale: odd -> odd
        ((15, 15), (7, 7)),
        # 2D upscale: even -> odd
        ((8, 8), (15, 15)),
        # 2D upscale: odd -> even
        ((7, 7), (16, 16)),
        # Asymmetric source/target (cryo-EM tilt-stack-like aspect ratio)
        ((4096, 4096), (609, 609)),
    ],
)
def test_fourier_rescale_2d_preserves_mean_for_all_parities(source_shape, target_shape):
    torch.manual_seed(0)
    image = torch.full(source_shape, 5.0) + 0.01 * torch.randn(source_shape)
    rescaled, _ = fourier_rescale_2d(
        image=image, target_shape=target_shape, preserve_mean=True
    )
    assert tuple(rescaled.shape) == target_shape
    assert rescaled.mean().item() == pytest.approx(image.mean().item(), abs=1e-3)


@pytest.mark.parametrize(
    "source_shape,target_shape",
    [
        ((16, 16, 16), (7, 7, 7)),
        ((15, 15, 15), (8, 8, 8)),
        ((7, 7, 7), (16, 16, 16)),
    ],
)
def test_fourier_rescale_3d_preserves_mean_for_all_parities(source_shape, target_shape):
    torch.manual_seed(0)
    image = torch.full(source_shape, 5.0) + 0.01 * torch.randn(source_shape)
    rescaled, _ = fourier_rescale_3d(
        image=image, target_shape=target_shape, preserve_mean=True
    )
    assert tuple(rescaled.shape) == target_shape
    assert rescaled.mean().item() == pytest.approx(image.mean().item(), abs=1e-3)


@pytest.mark.parametrize(
    "dtype", [int, float, np.float32, np.float64, torch.float32, torch.float64]
)
def test_pixel_spacing_scalar_dtypes(dtype, circle, sphere):
    # Smoke test - just verify functions don't crash with different scalar dtypes
    if isinstance(dtype, type) and issubclass(dtype, numbers.Number):
        source_spacing = 1
        target_spacing = 2
    elif dtype in [int, float]:
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


# Tests for fourier_rescale_rfft_2d and fourier_rescale_rfft_3d with even/odd arrays
@pytest.mark.parametrize(
    "source_shape,target_shape",
    [
        # even -> even
        ((16, 16), (8, 8)),
        # even -> odd (the common cryo-EM case)
        ((16, 16), (7, 7)),
        # odd -> even
        ((15, 15), (8, 8)),
        # odd -> odd
        ((15, 15), (7, 7)),
        # even -> odd upscale
        ((8, 8), (15, 15)),
        # odd -> even upscale
        ((7, 7), (16, 16)),
    ],
)
def test_fourier_rescale_rfft_2d_with_even_odd_arrays(source_shape, target_shape):
    """Test fourier_rescale_rfft_2d with pre-computed rfft for even/odd sizes."""
    torch.manual_seed(0)
    image = torch.full(source_shape, 5.0) + 0.01 * torch.randn(source_shape)

    # Compute rfft as done internally in fourier_rescale_2d
    image_shifted = torch.fft.fftshift(image, dim=(-2, -1))
    dft = torch.fft.rfftn(image_shifted, dim=(-2, -1))
    dft_shifted = torch.fft.fftshift(dft, dim=(-2,))

    # Call the rfft function directly
    dft_rescaled = fourier_rescale_rfft_2d(
        dft=dft_shifted,
        image_shape=source_shape,
        target_shape=target_shape,
    )

    # Verify output shape is correct
    # For rfft, the last dimension should be target_shape[-1] // 2 + 1
    expected_rfft_width = target_shape[-1] // 2 + 1
    expected_shape = (target_shape[-2], expected_rfft_width)
    assert dft_rescaled.shape[-2:] == expected_shape


@pytest.mark.parametrize(
    "source_shape,target_shape",
    [
        # even -> even
        ((16, 16, 16), (8, 8, 8)),
        # even -> odd
        ((16, 16, 16), (7, 7, 7)),
        # odd -> even
        ((15, 15, 15), (8, 8, 8)),
        # odd -> odd
        ((15, 15, 15), (7, 7, 7)),
        # upscale: odd -> even
        ((7, 7, 7), (16, 16, 16)),
        # upscale: even -> odd
        ((8, 8, 8), (15, 15, 15)),
    ],
)
def test_fourier_rescale_rfft_3d_with_even_odd_arrays(source_shape, target_shape):
    """Test fourier_rescale_rfft_3d with pre-computed rfft for even/odd sizes."""
    torch.manual_seed(0)
    image = torch.full(source_shape, 5.0) + 0.01 * torch.randn(source_shape)

    # Compute rfft as done internally in fourier_rescale_3d
    image_shifted = torch.fft.fftshift(image, dim=(-3, -2, -1))
    dft = torch.fft.rfftn(image_shifted, dim=(-3, -2, -1))
    dft_shifted = torch.fft.fftshift(dft, dim=(-3, -2))

    # Call the rfft function directly
    dft_rescaled = fourier_rescale_rfft_3d(
        dft=dft_shifted,
        image_shape=source_shape,
        target_shape=target_shape,
    )

    # Verify output shape is correct
    # For rfft, the last dimension should be target_shape[-1] // 2 + 1
    expected_rfft_width = target_shape[-1] // 2 + 1
    expected_shape = (target_shape[-3], target_shape[-2], expected_rfft_width)
    assert dft_rescaled.shape[-3:] == expected_shape
