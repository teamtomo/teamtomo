"""Tests for `torch_angular_search` package."""

import platform

import numpy as np
import pytest

from torch_so3.base_s2_grid import (
    _nside_from_theta_step,
    healpix_base_grid,
    healpix_sectored_base_grid,
)
from torch_so3.local_so3_sampling import (
    get_local_high_resolution_angles,
    get_roll_angles,
)
from torch_so3.uniform_so3_sampling import (
    get_sectored_euler_angles,
    get_uniform_euler_angles,
)

# TODO: Check actual values of returned tensors


def test_get_uniform_euler_angles():
    # Test the angle generator
    angles = get_uniform_euler_angles(base_grid_method="uniform")
    assert angles.shape == (1584480, 3)

    # Ensure that the angles are within the desired (default) range
    assert (angles[:, 0] >= 0).all()
    assert (angles[:, 0] <= 360).all()
    assert (angles[:, 1] >= 0).all()
    assert (angles[:, 1] <= 180).all()
    assert (angles[:, 2] >= 0).all()
    assert (angles[:, 2] <= 360).all()


def test_get_uniform_euler_angles_includes_zero():
    """Test that (0, 0, 0) Euler angles are included in the output."""
    angles = get_uniform_euler_angles(base_grid_method="uniform")

    # Check if any row matches (0, 0, 0) within tolerance
    zero_angle = np.array([0.0, 0.0, 0.0])
    # Convert to numpy for comparison
    angles_np = angles.numpy()

    # Check if any row is close to (0, 0, 0) by computing differences
    differences = np.abs(angles_np - zero_angle)
    matches = np.all(differences < 1e-5, axis=1)
    assert np.any(matches), "Euler angles (0, 0, 0) should be included in the output"


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_get_uniform_euler_angles_healpix():
    angles = get_uniform_euler_angles(base_grid_method="healpix")
    assert angles.shape == (1658880, 3)


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_healpix_base_grid_fine_theta_step():
    """Fine theta_step is allowed without the old nside < 36 cap."""
    nside = _nside_from_theta_step(0.5)
    assert nside >= 36
    grid = healpix_base_grid(theta_step=0.5)
    assert grid.shape == (12 * nside**2, 2)


def test_get_local_high_resolution_angles():
    local_angles = get_local_high_resolution_angles()
    assert local_angles.shape == (1581, 3)

    # range tests for angles
    assert (local_angles[:, 0] >= -1.51).all()
    assert (local_angles[:, 0] <= 1.51).all()
    assert (local_angles[:, 1] >= -2.51).all()
    assert (local_angles[:, 1] <= 2.51).all()
    assert np.allclose(local_angles[:, 2].min().item(), -1.50)
    assert np.allclose(local_angles[:, 2].max().item(), 1.50)


def test_get_local_high_resolution_angles_includes_zero():
    """Test that (0, 0, 0) Euler angles are included in the output."""
    local_angles = get_local_high_resolution_angles()

    # Check if any row matches (0, 0, 0) within tolerance
    zero_angle = np.array([0.0, 0.0, 0.0])
    # Convert to numpy for comparison
    local_angles_np = local_angles.numpy()

    # Check if any row is close to (0, 0, 0) by computing differences
    differences = np.abs(local_angles_np - zero_angle)
    matches = np.all(differences < 1e-5, axis=1)
    assert np.any(matches), "Euler angles (0, 0, 0) should be included in the output"


def test_get_roll_angles():
    roll_angles = get_roll_angles()
    assert roll_angles.shape == (151290, 3)

    # range tests for angles
    assert (roll_angles[:, 1] >= -10.01).all()
    assert (roll_angles[:, 1] <= 10.01).all()


# ---------------------------------------------------------------------------
# healpix_sectored_base_grid tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_healpix_sectored_base_grid_shape():
    """nside_coarse=1, nside_fine=2 -> 12 sectors x 4 fine pixels x 2 angles."""
    grid = healpix_sectored_base_grid(nside_coarse=1, nside_fine=2)
    assert grid.shape == (12, 4, 2)


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_healpix_sectored_base_grid_equal_area():
    """Every sector has the same number of fine pixels."""
    grid = healpix_sectored_base_grid(nside_coarse=2, nside_fine=4)
    n_sectors = grid.shape[0]
    k2 = grid.shape[1]
    assert n_sectors == 12 * 2**2  # 48 sectors
    assert k2 == (4 // 2) ** 2  # 4 fine pixels per sector


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_healpix_sectored_base_grid_theta_step():
    """nside_fine inferred from theta_step covers the expected resolution."""
    grid = healpix_sectored_base_grid(nside_coarse=1, theta_step=30.0)
    assert grid.ndim == 3
    assert grid.shape[0] == 12  # 12 coarse sectors for nside_coarse=1
    assert grid.shape[2] == 2  # (phi, theta)


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_healpix_sectored_base_grid_nside_fine_lt_coarse_raises():
    with pytest.raises(ValueError, match="nside_fine"):
        healpix_sectored_base_grid(nside_coarse=4, nside_fine=2)


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_healpix_sectored_base_grid_narrow_range_fewer_sectors():
    """A narrow angular window typically keeps fewer coarse sectors than full sphere."""
    full = healpix_sectored_base_grid(nside_coarse=1, nside_fine=2)
    narrow = healpix_sectored_base_grid(
        nside_coarse=1,
        nside_fine=2,
        theta_min=0.0,
        theta_max=15.0,
        phi_min=0.0,
        phi_max=30.0,
    )
    assert full.shape[0] == 12
    assert narrow.shape[0] < 12
    assert narrow.shape[1] == 4


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_healpix_sectored_base_grid_impossible_range_empty():
    """No fine centre in range -> no sectors."""
    grid = healpix_sectored_base_grid(
        nside_coarse=1,
        nside_fine=2,
        phi_min=400.0,
        phi_max=410.0,
    )
    assert grid.shape == (0, 4, 2)


# ---------------------------------------------------------------------------
# get_sectored_euler_angles tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_get_sectored_euler_angles_shape():
    """nside_coarse=1, nside_fine=2, psi_step=1.5 -> (12, k2*n_psi, 3)."""
    psi_step = 1.5
    angles = get_sectored_euler_angles(nside_coarse=1, nside_fine=2, psi_step=psi_step)
    n_psi = len(np.arange(0.0, 360.0, psi_step))
    k2 = (2 // 1) ** 2  # 4
    assert angles.shape == (12, k2 * n_psi, 3)


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_get_sectored_euler_angles_total_count():
    """Total angle triples equals n_sectors * k2 * n_psi."""
    psi_step = 5.0
    angles = get_sectored_euler_angles(nside_coarse=1, nside_fine=2, psi_step=psi_step)
    n_psi = len(np.arange(0.0, 360.0, psi_step))
    k2 = 4
    assert angles.numel() // 3 == 12 * k2 * n_psi


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_get_sectored_euler_angles_psi_sweep_per_sector():
    """Within each sector, each S2 direction has the full psi sweep."""
    psi_step = 5.0
    psi_min, psi_max = 0.0, 360.0
    angles = get_sectored_euler_angles(
        nside_coarse=1,
        nside_fine=2,
        psi_step=psi_step,
        psi_min=psi_min,
        psi_max=psi_max,
    )
    expected_psi = np.arange(psi_min, psi_max, psi_step)
    n_psi = len(expected_psi)
    k2 = 4

    for sector in range(angles.shape[0]):
        sector_angles = angles[sector]  # (k2*n_psi, 3)
        for i in range(k2):
            # Rows for fine pixel i: i*n_psi : (i+1)*n_psi
            psi_vals = sector_angles[i * n_psi : (i + 1) * n_psi, 2].numpy()
            assert np.allclose(psi_vals, expected_psi, atol=1e-9)
            # phi and theta are constant within this block
            phi_theta = sector_angles[i * n_psi : (i + 1) * n_psi, :2]
            assert (phi_theta == phi_theta[0]).all()


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_get_sectored_euler_angles_psi_min_ge_psi_max_single_psi():
    """When ``psi_min >= psi_max``, use a single ``psi`` value."""
    angles = get_sectored_euler_angles(
        nside_coarse=1,
        nside_fine=2,
        psi_min=42.0,
        psi_max=42.0,
    )
    assert angles.shape == (12, 4, 3)
    assert (angles[..., 2] == 42.0).all()


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_get_sectored_euler_angles_angle_range():
    """Output phi in [0, 360), theta in [0, 180], psi in [psi_min, psi_max)."""
    angles = get_sectored_euler_angles(nside_coarse=1, nside_fine=2, psi_step=5.0)
    phi, theta, psi = angles[..., 0], angles[..., 1], angles[..., 2]
    assert (phi >= 0).all() and (phi < 360).all()
    assert (theta >= 0).all() and (theta <= 180).all()
    assert (psi >= 0).all() and (psi < 360).all()
