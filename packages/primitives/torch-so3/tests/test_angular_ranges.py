import pytest

from torch_so3.angular_ranges import SymmetryRanges, get_symmetry_ranges


def test_default_symmetry():
    result = get_symmetry_ranges()
    expected = SymmetryRanges(
        phi_min=0.0,
        phi_max=360.0,
        theta_min=0.0,
        theta_max=180.0,
        psi_min=0.0,
        psi_max=360.0,
    )
    assert result == expected


def test_symmetry_group_C():
    result = get_symmetry_ranges(symmetry_group="C", symmetry_order=2)
    expected = SymmetryRanges(
        phi_min=0.0,
        phi_max=180.0,
        theta_min=0.0,
        theta_max=180.0,
        psi_min=0.0,
        psi_max=360.0,
    )
    assert result == expected


def test_symmetry_group_D():
    result = get_symmetry_ranges(symmetry_group="D", symmetry_order=2)
    expected = SymmetryRanges(
        phi_min=0.0,
        phi_max=180.0,
        theta_min=0.0,
        theta_max=90.0,
        psi_min=0.0,
        psi_max=360.0,
    )
    assert result == expected


def test_symmetry_group_T():
    result = get_symmetry_ranges(symmetry_group="T")
    expected = SymmetryRanges(
        phi_min=0.0,
        phi_max=180.0,
        theta_min=0,
        theta_max=54.7356,
        psi_min=0.0,
        psi_max=360.0,
    )
    assert result == expected


def test_symmetry_group_O():
    result = get_symmetry_ranges(symmetry_group="O")
    expected = SymmetryRanges(
        phi_min=0.0,
        phi_max=90.0,
        theta_min=0.0,
        theta_max=54.7356,
        psi_min=0.0,
        psi_max=360.0,
    )
    assert result == expected


def test_symmetry_group_I():
    result = get_symmetry_ranges(symmetry_group="I")
    expected = SymmetryRanges(
        phi_min=0.0,
        phi_max=180.0,
        theta_min=0.0,
        theta_max=31.7,
        psi_min=0.0,
        psi_max=360.0,
    )
    assert result == expected


def test_invalid_symmetry_group():
    with pytest.raises(ValueError, match="Symmetry group not recognized"):
        get_symmetry_ranges(symmetry_group="invalid")
