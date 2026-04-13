import torch

from torch_fourier_shift.phase_shift_grids import (
    fftfreq_grid_1d_with_cache,
    phase_shift_grid_1d,
)


def test_cache_1d():
    """Test caching for 1D phase shift grids."""
    shape = (16,)
    shifts = torch.randn(1, 1)

    # Clear cache before test
    fftfreq_grid_1d_with_cache.cache_clear()

    # Call once to populate the cache
    phase_shift_grid_1d(
        shifts, image_shape=shape, rfft=False, fftshift=False, cache_intermediates=True
    )
    assert fftfreq_grid_1d_with_cache.cache_info().hits == 0
    assert fftfreq_grid_1d_with_cache.cache_info().misses == 1

    # Call again to get a cache hit
    phase_shift_grid_1d(
        shifts, image_shape=shape, rfft=False, fftshift=False, cache_intermediates=True
    )
    assert fftfreq_grid_1d_with_cache.cache_info().hits == 1
    assert fftfreq_grid_1d_with_cache.cache_info().misses == 1


def test_cached_vs_uncached_1d():
    """Test that cached and uncached 1D phase shift grids are the same."""
    shape = (16,)
    shifts = torch.randn(1, 1)

    # Get cached result
    cached_result = phase_shift_grid_1d(
        shifts, image_shape=shape, rfft=False, fftshift=False, cache_intermediates=True
    )

    # Get uncached result
    uncached_result = phase_shift_grid_1d(
        shifts, image_shape=shape, rfft=False, fftshift=False, cache_intermediates=False
    )

    torch.testing.assert_close(cached_result, uncached_result)
