"""Generate Euler angles ranges applying symmetry operations."""

from typing import NamedTuple


class SymmetryRanges(NamedTuple):
    """NamedTuple child class for 'get_symmetry_ranges' return type."""

    phi_min: float
    phi_max: float
    theta_min: float
    theta_max: float
    psi_min: float
    psi_max: float


def get_symmetry_ranges(
    symmetry_group: str = "C",
    symmetry_order: int = 1,
) -> SymmetryRanges:
    """
    Generate Euler angle ranges based on symmetry group.

    Parameters
    ----------
    symmetry_group : str, optional
        Symmetry group of the particle (C, D, T, O, I). Default is "C".
    symmetry_order : int, optional
        Order of the symmetry group. Default is 1.

    Returns
    -------
    SymmetryRanges
        NamedTuple containing the min/max for phi, theta, and psi.
    """
    # Convert to upper case
    symmetry_group = symmetry_group.upper()
    phi_max = 360.0
    theta_max = 180.0
    psi_max = 360.0
    if symmetry_group == "C":
        phi_max = 360 / float(symmetry_order)
    elif symmetry_group == "D":
        phi_max = 360 / float(symmetry_order)
        theta_max = 90.0
    elif symmetry_group == "T":
        phi_max = 180.0
        theta_max = 54.7356
    elif symmetry_group == "O":
        phi_max = 90.0
        theta_max = 54.7356
    elif symmetry_group == "I":
        phi_max = 180.0
        theta_max = 31.7
    else:
        raise ValueError("Symmetry group not recognized")

    return SymmetryRanges(
        phi_min=0.0,
        phi_max=phi_max,
        theta_min=0.0,
        theta_max=theta_max,
        psi_min=0.0,
        psi_max=psi_max,
    )
