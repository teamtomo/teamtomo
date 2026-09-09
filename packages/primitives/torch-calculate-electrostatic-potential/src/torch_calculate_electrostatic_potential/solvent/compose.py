"""High-level solvent and solvated-potential helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ..structure import potential_from_structure_3d
from .geometry import distance_to_surface
from .occupancy import _require_unbatched_structure, vdw_probe_occupancy
from .potential import (
    DEFAULT_ICE_POTENTIAL_V,
    constant_solvent_potential,
    shang_sigworth_solvent_potential,
)

if TYPE_CHECKING:
    from torch import Tensor
    from torch_structure_manipulation import AtomicStructure

    from ..grid import GridConfig
    from ..utils.peng_model import BondedFallback, ScatteringFactors

SolventModel = Literal["constant", "shang_sigworth"]


def solvent_potential_from_structure_3d(
    structure: AtomicStructure,
    grid_config: GridConfig,
    *,
    model: SolventModel = "shang_sigworth",
    probe_radius: float = 1.4,
    r_asymptote: float = 7.5,
    ice_potential_V: float = DEFAULT_ICE_POTENTIAL_V,
    batch_size: int = 256,
) -> Tensor:
    """Solvent-only electrostatic potential in volts.

    Parameters
    ----------
    structure : AtomicStructure
        Unbatched structure (``positions_zyx`` shape ``(n_atoms, 3)``).
    grid_config : GridConfig
        Three-dimensional grid (same as dry ESP).
    model : {"constant", "shang_sigworth"}
        ``shang_sigworth`` (default) = Shang–Sigworth hydration density times
        bulk ice MIP.
        ``constant`` = VdW+probe occupancy times bulk ice MIP.
    probe_radius : float
        Probe radius in Angstroms.
    r_asymptote : float
        Neighborhood radius for distance updates.
    ice_potential_V : float
        Bulk amorphous-ice mean inner potential in volts.
    batch_size : int
        Passed to distance geometry.

    Returns
    -------
    Tensor
        Solvent potential volume in volts, ZYX order.
    """
    _require_unbatched_structure(structure)
    dist_map, nearest_z = distance_to_surface(
        structure.positions_zyx,
        structure.atomic_numbers,
        grid_config,
        r_asymptote=r_asymptote,
        batch_size=batch_size,
    )
    if model == "constant":
        occupancy = vdw_probe_occupancy(dist_map, probe_radius=probe_radius)
        return constant_solvent_potential(occupancy, ice_potential_V=ice_potential_V)
    if model == "shang_sigworth":
        return shang_sigworth_solvent_potential(
            dist_map,
            nearest_z,
            ice_potential_V=ice_potential_V,
            probe_radius=probe_radius,
        )
    raise ValueError(
        f"unknown solvent model {model!r}; expected 'constant' or 'shang_sigworth'"
    )


def solvated_potential_from_structure_3d(
    structure: AtomicStructure,
    grid_config: GridConfig,
    *,
    model_water_potential: bool = False,
    solvent_model: SolventModel = "shang_sigworth",
    ice_potential_V: float = DEFAULT_ICE_POTENTIAL_V,
    probe_radius: float = 1.4,
    r_asymptote: float = 7.5,
    scattering_factors: ScatteringFactors = "peng_elemental",
    bonded_fallback: BondedFallback = "elemental",
    per_voxel_averaging: bool = True,
    batch_size: int = 4096,
    solvent_batch_size: int = 256,
) -> Tensor:
    """Dry atomic ESP plus optional continuum solvent, in volts.

    Parameters
    ----------
    structure : AtomicStructure
        Unbatched when ``model_water_potential`` is True.
    grid_config : GridConfig
        Three-dimensional grid.
    model_water_potential : bool
        If False (default), return dry ``potential_from_structure_3d``.
        If True, add continuum ice using ``solvent_model``.
    solvent_model : {"constant", "shang_sigworth"}
        Water model used when ``model_water_potential`` is True.
        Defaults to Shang & Sigworth (2012) continuum hydration.
    ice_potential_V : float
        Bulk ice MIP in volts.
    probe_radius : float
        Probe radius in Angstroms.
    r_asymptote : float
        Neighborhood radius for distance updates.
    scattering_factors, bonded_fallback, per_voxel_averaging, batch_size
        Passed to :func:`potential_from_structure_3d`.
    solvent_batch_size : int
        Atom chunk hint for solvent geometry.

    Returns
    -------
    Tensor
        Total potential in volts (atomic + solvent when applicable).
    """
    atomic = potential_from_structure_3d(
        structure,
        grid_config,
        scattering_factors=scattering_factors,
        bonded_fallback=bonded_fallback,
        per_voxel_averaging=per_voxel_averaging,
        batch_size=batch_size,
    )
    if not model_water_potential:
        return atomic
    solvent = solvent_potential_from_structure_3d(
        structure,
        grid_config,
        model=solvent_model,
        probe_radius=probe_radius,
        r_asymptote=r_asymptote,
        ice_potential_V=ice_potential_V,
        batch_size=solvent_batch_size,
    )
    return atomic + solvent
