"""Tests for `torch_so3.hierarchical_s2_grid`."""

import platform

import healpy as hp
import numpy as np
import pytest

from torch_so3.hierarchical_s2_grid import GridLevel, HierarchicalS2Grid

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)


# ---------------------------------------------------------------------------
# GridLevel tests
# ---------------------------------------------------------------------------


def test_grid_level_npix():
    level = GridLevel(nside=4, depth=0)
    assert level.npix == 12 * 4**2


def test_grid_level_s2_step_deg():
    level = GridLevel(nside=4, depth=0)
    assert level.s2_step_deg == pytest.approx(58.6 / 4)


def test_grid_level_angle_from_index_matches_healpy():
    level = GridLevel(nside=8, depth=0)
    ipix = np.arange(level.npix)

    theta, phi = level.angle_from_index(ipix)
    expected_theta, expected_phi = hp.pix2ang(8, ipix, nest=True)

    np.testing.assert_allclose(theta, expected_theta)
    np.testing.assert_allclose(phi, expected_phi)


def test_grid_level_index_from_angle_roundtrip():
    level = GridLevel(nside=8, depth=0)
    ipix = np.arange(level.npix)

    theta, phi = level.angle_from_index(ipix)
    recovered = level.index_from_angle(theta, phi)

    np.testing.assert_array_equal(recovered, ipix)


def test_grid_level_all_indices_and_all_angles():
    level = GridLevel(nside=2, depth=0)
    indices = level.all_indices()
    assert indices.shape == (level.npix,)
    np.testing.assert_array_equal(indices, np.arange(level.npix))

    theta, phi = level.all_angles()
    expected_theta, expected_phi = level.angle_from_index(indices)
    np.testing.assert_array_equal(theta, expected_theta)
    np.testing.assert_array_equal(phi, expected_phi)


# ---------------------------------------------------------------------------
# HierarchicalS2Grid construction/validation tests
# ---------------------------------------------------------------------------


def test_init_validates_nside_finest_power_of_two():
    with pytest.raises(ValueError, match="power of 2"):
        HierarchicalS2Grid(nside_finest=3, n_levels=1)


def test_init_validates_divisibility():
    # nside_finest=4 is not divisible by 2**(n_levels-1)=8
    with pytest.raises(ValueError, match="divisible"):
        HierarchicalS2Grid(nside_finest=4, n_levels=4)


def test_levels_ordering_coarsest_to_finest():
    grid = HierarchicalS2Grid(nside_finest=8, n_levels=3)
    nsides = [level.nside for level in grid.levels]
    assert nsides == [2, 4, 8]
    depths = [level.depth for level in grid.levels]
    assert depths == [2, 1, 0]


def test_coarsest_and_finest_level_properties():
    grid = HierarchicalS2Grid(nside_finest=8, n_levels=3)
    assert grid.coarsest_level == 0
    assert grid.finest_level == 2


def test_get_level_out_of_bounds_raises():
    grid = HierarchicalS2Grid(nside_finest=4, n_levels=1)
    with pytest.raises(ValueError, match="Level must be between"):
        grid.get_level(1)
    with pytest.raises(ValueError, match="Level must be between"):
        grid.get_level(-1)


def test_repr_shows_nsides():
    grid = HierarchicalS2Grid(nside_finest=8, n_levels=3)
    assert repr(grid) == "HierarchicalS2Grid(n_levels=3, nsides=[2, 4, 8])"


def test_from_target_step_deg_picks_nearest_power_of_two_nside():
    grid = HierarchicalS2Grid.from_target_step_deg(target_step_deg=14.65, n_levels=1)
    # 58.6 / 14.65 ~= 4.0 -> nearest power of two is 4
    assert grid.nside_finest == 4


def test_from_target_step_deg_too_large_raises():
    with pytest.raises(ValueError, match="too large"):
        HierarchicalS2Grid.from_target_step_deg(target_step_deg=1000.0, n_levels=1)


# ---------------------------------------------------------------------------
# Angle <-> index conversion tests
# ---------------------------------------------------------------------------


def test_get_level_orientations_degrees_vs_radians():
    grid = HierarchicalS2Grid(nside_finest=4, n_levels=1)
    theta_deg, phi_deg = grid.get_level_orientations(0, degrees=True)
    theta_rad, phi_rad = grid.get_level_orientations(0, degrees=False)

    np.testing.assert_allclose(theta_deg, np.degrees(theta_rad))
    np.testing.assert_allclose(phi_deg, np.degrees(phi_rad))


def test_angle_from_index_matches_get_level():
    grid = HierarchicalS2Grid(nside_finest=4, n_levels=1)
    ipix = np.arange(grid.get_level(0).npix)

    theta, phi = grid.angle_from_index(ipix, level=0, degrees=True)
    expected_theta, expected_phi = grid.get_level(0).angle_from_index(ipix)

    np.testing.assert_allclose(theta, np.degrees(expected_theta))
    np.testing.assert_allclose(phi, np.degrees(expected_phi))


def test_index_from_angle_roundtrip_degrees():
    grid = HierarchicalS2Grid(nside_finest=4, n_levels=1)
    ipix = np.arange(grid.get_level(0).npix)

    theta, phi = grid.angle_from_index(ipix, level=0, degrees=True)
    recovered = grid.index_from_angle(theta, phi, level=0, degrees=True)

    np.testing.assert_array_equal(recovered, ipix)


# ---------------------------------------------------------------------------
# convert_index tests
# ---------------------------------------------------------------------------


def test_convert_index_same_level_is_identity():
    grid = HierarchicalS2Grid(nside_finest=8, n_levels=2)
    ipix = np.array([0, 1, 2, 3])
    result = grid.convert_index(ipix, from_level=1, to_level=1)
    np.testing.assert_array_equal(result, ipix)


def test_convert_index_to_coarser_is_many_to_one():
    grid = HierarchicalS2Grid(nside_finest=8, n_levels=2)  # nsides = [4, 8]
    # Every group of 4 fine (nested) pixels maps to a single coarse pixel.
    fine_ipix = np.arange(grid.get_level(1).npix)
    coarse_ipix = grid.convert_index(fine_ipix, from_level=1, to_level=0)

    expected = fine_ipix // 4
    np.testing.assert_array_equal(coarse_ipix, expected)


def test_convert_index_to_finer_is_one_to_many():
    grid = HierarchicalS2Grid(nside_finest=8, n_levels=2)  # nsides = [4, 8]
    coarse_ipix = np.array([5])
    fine_ipix = grid.convert_index(coarse_ipix, from_level=0, to_level=1)

    assert fine_ipix.shape == (4,)
    np.testing.assert_array_equal(np.sort(fine_ipix), np.arange(20, 24))


def test_convert_index_roundtrip_covers_all_fine_pixels():
    grid = HierarchicalS2Grid(nside_finest=8, n_levels=2)
    coarse_ipix = grid.get_level(0).all_indices()
    fine_ipix = grid.convert_index(coarse_ipix, from_level=0, to_level=1)

    # Every fine pixel is reached exactly once across all coarse parents.
    assert np.array_equal(np.sort(fine_ipix), grid.get_level(1).all_indices())

    back_to_coarse = grid.convert_index(fine_ipix, from_level=1, to_level=0)
    np.testing.assert_array_equal(np.unique(back_to_coarse), np.unique(coarse_ipix))


# ---------------------------------------------------------------------------
# neighbor_indices / followup tests
# ---------------------------------------------------------------------------


def test_neighbor_indices_zero_rings_returns_seed_only():
    grid = HierarchicalS2Grid(nside_finest=4, n_levels=1)
    result = grid.neighbor_indices(ipix=5, level=0, rings=0)
    np.testing.assert_array_equal(result, np.array([5]))


def test_neighbor_indices_matches_healpy_get_all_neighbours():
    grid = HierarchicalS2Grid(nside_finest=4, n_levels=1)
    ipix = 5
    result = grid.neighbor_indices(ipix, level=0, rings=1)

    expected_neighbors = hp.get_all_neighbours(4, ipix, nest=True)
    expected_neighbors = expected_neighbors[expected_neighbors >= 0]
    expected = np.unique(np.concatenate([[ipix], expected_neighbors]))

    np.testing.assert_array_equal(result, expected)


def test_neighbor_indices_more_rings_grows_the_result():
    grid = HierarchicalS2Grid(nside_finest=8, n_levels=1)
    small = grid.neighbor_indices(ipix=10, level=0, rings=1)
    large = grid.neighbor_indices(ipix=10, level=0, rings=2)
    assert set(small.tolist()).issubset(set(large.tolist()))
    assert len(large) >= len(small)


def test_followup_returns_finer_level_indices():
    grid = HierarchicalS2Grid(nside_finest=8, n_levels=2)  # nsides = [4, 8]
    result = grid.followup(ipix=5, source_level=0, target_level=1, rings=0)

    # rings=0 -> just the seed's own children (4 fine pixels).
    expected = np.sort(grid.convert_index([5], from_level=0, to_level=1))
    np.testing.assert_array_equal(np.sort(result), expected)


# ---------------------------------------------------------------------------
# sector_child_angles / sector_bounds_mask tests
# ---------------------------------------------------------------------------


def test_sector_child_angles_shape_is_equal_area():
    grid = HierarchicalS2Grid(nside_finest=4, n_levels=3)  # nsides = [1, 2, 4]
    theta, phi = grid.sector_child_angles(
        grid.coarsest_level, grid.finest_level, degrees=True
    )
    n_sectors = grid.get_level(grid.coarsest_level).npix
    k = grid.get_level(grid.finest_level).npix // n_sectors

    assert theta.shape == (n_sectors, k)
    assert phi.shape == (n_sectors, k)


def test_sector_child_angles_matches_manual_grouping():
    grid = HierarchicalS2Grid(nside_finest=4, n_levels=3)
    theta, phi = grid.sector_child_angles(
        grid.coarsest_level, grid.finest_level, degrees=True
    )

    coarse = grid.get_level(grid.coarsest_level)
    fine_ipix = grid.convert_index(
        coarse.all_indices(),
        from_level=grid.coarsest_level,
        to_level=grid.finest_level,
    ).reshape(coarse.npix, -1)
    expected_theta, expected_phi = grid.angle_from_index(
        fine_ipix.reshape(-1), level=grid.finest_level, degrees=True
    )
    expected_theta = expected_theta.reshape(fine_ipix.shape)
    expected_phi = expected_phi.reshape(fine_ipix.shape)

    np.testing.assert_array_equal(theta, expected_theta)
    np.testing.assert_array_equal(phi, expected_phi)


def test_sector_bounds_mask_keeps_all_for_full_sphere():
    grid = HierarchicalS2Grid(nside_finest=4, n_levels=3)
    keep_mask, theta, _phi = grid.sector_bounds_mask(
        grid.coarsest_level, grid.finest_level
    )
    assert keep_mask.shape == (theta.shape[0],)
    assert np.all(keep_mask)


def test_sector_bounds_mask_matches_brute_force_any_child_in_range():
    grid = HierarchicalS2Grid(nside_finest=4, n_levels=3)
    theta_min, theta_max, phi_min, phi_max = 0.0, 90.0, 0.0, 180.0

    keep_mask, theta, phi = grid.sector_bounds_mask(
        grid.coarsest_level,
        grid.finest_level,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
    )

    expected = np.array(
        [
            np.any(
                (theta[i] >= theta_min)
                & (theta[i] <= theta_max)
                & (phi[i] >= phi_min)
                & (phi[i] <= phi_max)
            )
            for i in range(theta.shape[0])
        ]
    )
    np.testing.assert_array_equal(keep_mask, expected)
    assert 0 < keep_mask.sum() < keep_mask.shape[0]


def test_sector_bounds_mask_empty_range_keeps_none():
    grid = HierarchicalS2Grid(nside_finest=4, n_levels=3)
    keep_mask, _, _ = grid.sector_bounds_mask(
        grid.coarsest_level,
        grid.finest_level,
        phi_min=400.0,
        phi_max=410.0,
    )
    assert not np.any(keep_mask)
