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
    assert tuple(sphere.shape) == (14, 14, 14)
    assert tuple(rescaled.shape) == (28, 28, 28)

    # test upscale with uneven box
    rescaled, new_spacing = fourier_rescale_3d(
        image=F.pad(sphere, (0, 1, 0, 1, 0, 1)), source_spacing=1, target_spacing=0.5
    )
    assert tuple(rescaled.shape) == (30, 30, 30)


def test_fourier_downscale_3d(sphere):
    rescaled, new_spacing = fourier_rescale_3d(
        image=sphere, source_spacing=1, target_spacing=2
    )
    assert tuple(sphere.shape) == (14, 14, 14)
    assert tuple(rescaled.shape) == (6, 6, 6)

    # test downscale with uneven box
    rescaled, new_spacing = fourier_rescale_3d(
        image=F.pad(sphere, (0, 1, 0, 1, 0, 1)), source_spacing=1, target_spacing=2
    )
    assert tuple(rescaled.shape) == (8, 8, 8)


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


@pytest.mark.parametrize("dtype", [
    int,
    float,
    np.float32,
    np.float64,
    torch.float32,
    torch.float64
])
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
        image=circle,
        source_spacing=source_spacing,
        target_spacing=target_spacing
    )
    assert rescaled_2d is not None
    assert new_spacing_2d is not None

    # Test 3D with dtype for both source and target spacing
    rescaled_3d, new_spacing_3d = fourier_rescale_3d(
        image=sphere,
        source_spacing=source_spacing,
        target_spacing=target_spacing
    )
    assert rescaled_3d is not None
    assert new_spacing_3d is not None
