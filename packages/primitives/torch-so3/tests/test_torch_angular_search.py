"""Tests for `torch_angular_search` package."""

import platform

import numpy as np
import pytest

from torch_so3.local_so3_sampling import (
    get_local_high_resolution_angles,
    get_roll_angles,
)
from torch_so3.uniform_so3_sampling import get_uniform_euler_angles

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


@pytest.mark.skipif(
    platform.system() == "Windows", reason="healpy is not supported on Windows"
)
def test_get_uniform_euler_angles_healpix():
    angles = get_uniform_euler_angles(base_grid_method="healpix")
    assert angles.shape == (1658880, 3)


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


def test_get_roll_angles():
    roll_angles = get_roll_angles()
    assert roll_angles.shape == (151290, 3)

    # range tests for angles
    assert (roll_angles[:, 1] >= -10.01).all()
    assert (roll_angles[:, 1] <= 10.01).all()
