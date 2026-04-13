import torch

from torch_fourier_shift.phase_shift_grids import (
    fftfreq_grid_3d_with_cache,
    phase_shift_grid_3d,
)


def test_cache_3d():
    """Test caching for 3D phase shift grids."""
    shape = (16, 16, 16)
    shifts = torch.randn(1, 3)

    # Clear cache before test
    fftfreq_grid_3d_with_cache.cache_clear()

    # Call once to populate the cache
    phase_shift_grid_3d(
        shifts, image_shape=shape, rfft=False, fftshift=False, cache_intermediates=True
    )
    assert fftfreq_grid_3d_with_cache.cache_info().hits == 0
    assert fftfreq_grid_3d_with_cache.cache_info().misses == 1

    # Call again to get a cache hit
    phase_shift_grid_3d(
        shifts, image_shape=shape, rfft=False, fftshift=False, cache_intermediates=True
    )
    assert fftfreq_grid_3d_with_cache.cache_info().hits == 1
    assert fftfreq_grid_3d_with_cache.cache_info().misses == 1


def test_cached_vs_uncached_3d():
    """Test that cached and uncached 3D phase shift grids are the same."""
    shape = (16, 16, 16)
    shifts = torch.randn(1, 3)

    # Get cached result
    cached_result = phase_shift_grid_3d(
        shifts, image_shape=shape, rfft=False, fftshift=False, cache_intermediates=True
    )

    # Get uncached result
    uncached_result = phase_shift_grid_3d(
        shifts, image_shape=shape, rfft=False, fftshift=False, cache_intermediates=False
    )

    torch.testing.assert_close(cached_result, uncached_result)
