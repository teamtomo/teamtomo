"""Solvent electrostatic potentials in volts."""

from __future__ import annotations

import torch

from .occupancy import vdw_probe_occupancy
from .shang_sigworth import shang_sigworth_density

DEFAULT_ICE_POTENTIAL_V = 3.6


def constant_solvent_potential(
    occupancy: torch.Tensor,
    *,
    ice_potential_V: float = DEFAULT_ICE_POTENTIAL_V,
) -> torch.Tensor:
    """Constant bulk-ice potential outside the protein.

    Parameters
    ----------
    occupancy : torch.Tensor
        Solvent occupancy in ``[0, 1]`` (``1`` = full solvent).
    ice_potential_V : float
        Bulk amorphous-ice mean inner potential in volts.

    Returns
    -------
    torch.Tensor
        Solvent potential in volts, same shape as ``occupancy``.
    """
    return occupancy * ice_potential_V


def shang_sigworth_solvent_potential(
    dist_map: torch.Tensor,
    nearest_atomic_numbers: torch.Tensor,
    *,
    ice_potential_V: float = DEFAULT_ICE_POTENTIAL_V,
    probe_radius: float = 1.4,
) -> torch.Tensor:
    """Shang–Sigworth continuum solvent potential in volts.

    Density is scaled by ``ice_potential_V`` and zeroed where
    ``dist_map < probe_radius`` (probe-excluded protein volume).

    Parameters
    ----------
    dist_map : torch.Tensor
        Surface distances in Angstroms.
    nearest_atomic_numbers : torch.Tensor
        Nearest-atom Z map.
    ice_potential_V : float
        Bulk ice MIP in volts (multiplies normalized density).
    probe_radius : float
        Probe radius used to zero the protein interior.

    Returns
    -------
    torch.Tensor
        Solvent potential in volts.
    """
    density = shang_sigworth_density(dist_map, nearest_atomic_numbers)
    occupancy = vdw_probe_occupancy(dist_map, probe_radius=probe_radius)
    return ice_potential_V * density * occupancy
