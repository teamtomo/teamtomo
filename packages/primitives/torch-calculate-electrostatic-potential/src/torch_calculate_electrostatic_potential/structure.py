"""High-level electrostatic-potential calculation from atomic structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .potential import (
    calculate_scattering_potential_2d,
    calculate_scattering_potential_3d,
)
from .utils.peng_model import (
    BondedFallback,
    ScatteringFactors,
    resolve_scattering_parameters,
)

if TYPE_CHECKING:
    from torch import Tensor
    from torch_structure_manipulation import AtomicStructure

    from .grid import GridConfig


def potential_from_structure_3d(
    structure: AtomicStructure,
    grid_config: GridConfig,
    *,
    scattering_factors: ScatteringFactors = "peng_elemental",
    bonded_fallback: BondedFallback = "elemental",
    per_voxel_averaging: bool = True,
    batch_size: int = 4096,
) -> Tensor:
    """Calculate a 3D potential in volts from an ``AtomicStructure``.

    Parameters
    ----------
    structure : AtomicStructure
        Atomic coordinates and metadata. Positions are in Angstroms and ZYX order.
        Batched leading dimensions are supported for elemental factors. For bonded
        factors, ``atomic_numbers`` must remain one-dimensional ``(n_atoms,)`` and
        bonding tuples must describe the same chemistry for every batch member.
    grid_config : GridConfig
        Three-dimensional output grid.
    scattering_factors : {"peng_elemental", "peng_bonded"} or mapping
        Bundled model, or custom ``BondedScatteringFactorTable`` instances keyed
        by molecule type. Bonding metadata is ignored for the default elemental
        model.
    bonded_fallback : {"elemental", "error"}
        Behavior for unsupported bonded providers or environment keys.
    per_voxel_averaging : bool
        Average over each voxel instead of sampling its center.
    batch_size : int
        Number of atoms evaluated per chunk.

    Returns
    -------
    Tensor
        Potential volume in volts, in ZYX order.
    """
    parameters_a, parameters_b = _parameters_from_structure(
        structure, grid_config, scattering_factors, bonded_fallback
    )
    return calculate_scattering_potential_3d(
        structure.positions_zyx,
        structure.b_factors,
        parameters_a,
        parameters_b,
        grid_config,
        atom_occupancies=structure.occupancies,
        per_voxel_averaging=per_voxel_averaging,
        batch_size=batch_size,
    )


def potential_from_structure_2d(
    structure: AtomicStructure,
    grid_config: GridConfig,
    *,
    scattering_factors: ScatteringFactors = "peng_elemental",
    bonded_fallback: BondedFallback = "elemental",
    per_voxel_averaging: bool = True,
    batch_size: int = 4096,
) -> Tensor:
    """Calculate a projected 2D potential in volt-Angstroms.

    Parameters
    ----------
    structure : AtomicStructure
        Atomic coordinates and metadata. The Z coordinate is analytically projected.
        Batched leading dimensions are supported for elemental factors. For bonded
        factors, ``atomic_numbers`` must remain one-dimensional ``(n_atoms,)`` and
        bonding tuples must describe the same chemistry for every batch member.
    grid_config : GridConfig
        Two-dimensional output grid in YX order.
    scattering_factors : {"peng_elemental", "peng_bonded"} or mapping
        Bundled model, or custom ``BondedScatteringFactorTable`` instances keyed
        by molecule type. Bonding metadata is ignored for the default elemental
        model.
    bonded_fallback : {"elemental", "error"}
        Behavior for unsupported bonded providers or environment keys.
    per_voxel_averaging : bool
        Average over each pixel instead of sampling its center.
    batch_size : int
        Number of atoms evaluated per chunk.

    Returns
    -------
    Tensor
        Projected potential image in V Angstrom, in YX order.
    """
    parameters_a, parameters_b = _parameters_from_structure(
        structure, grid_config, scattering_factors, bonded_fallback
    )
    return calculate_scattering_potential_2d(
        structure.positions_zyx[..., 1:],
        structure.b_factors,
        parameters_a,
        parameters_b,
        grid_config,
        atom_occupancies=structure.occupancies,
        per_voxel_averaging=per_voxel_averaging,
        batch_size=batch_size,
    )


def _parameters_from_structure(
    structure: AtomicStructure,
    grid_config: GridConfig,
    scattering_factors: ScatteringFactors,
    bonded_fallback: BondedFallback,
) -> tuple[Tensor, Tensor]:
    return resolve_scattering_parameters(
        structure.atomic_numbers,
        scattering_factors=scattering_factors,
        bonded_environments=structure.bonded_environments,
        molecule_types=structure.molecule_types,
        bonded_fallback=bonded_fallback,
        device=grid_config.device,
        dtype=grid_config.dtype,
    )
