"""Cryo-EM Electrostatic Potential computation with PyTorch."""

from importlib.metadata import PackageNotFoundError, version

from .atom_stack import AtomStack
from .grid import GridConfig, default_sublattice_radius
from .potential import (
    PENG_SCATTERING_TO_POTENTIAL,
    calculate_scattering_potential_2d,
    calculate_scattering_potential_3d,
)
from .solvent import (
    DEFAULT_ICE_POTENTIAL_V,
    VDW_RADII_A,
    constant_solvent_potential,
    distance_to_surface,
    shang_sigworth_density,
    shang_sigworth_solvent_potential,
    solvent_occupancy_from_structure_3d,
    solvent_potential_from_structure_3d,
    solvated_potential_from_structure_3d,
    vdw_probe_occupancy,
    vdw_radii_for_atomic_numbers,
    voxel_centers_zyx,
)
from .structure import potential_from_structure_2d, potential_from_structure_3d
from .utils.peng_model import (
    BondedScatteringFactorTable,
    get_peng_scattering_parameters,
    resolve_scattering_parameters,
)

try:
    __version__ = version("torch-calculate-electrostatic-potential")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "DEFAULT_ICE_POTENTIAL_V",
    "PENG_SCATTERING_TO_POTENTIAL",
    "VDW_RADII_A",
    "AtomStack",
    "BondedScatteringFactorTable",
    "GridConfig",
    "constant_solvent_potential",
    "default_sublattice_radius",
    "distance_to_surface",
    "__version__",
    "calculate_scattering_potential_2d",
    "calculate_scattering_potential_3d",
    "get_peng_scattering_parameters",
    "potential_from_structure_2d",
    "potential_from_structure_3d",
    "resolve_scattering_parameters",
    "shang_sigworth_density",
    "shang_sigworth_solvent_potential",
    "solvent_occupancy_from_structure_3d",
    "solvent_potential_from_structure_3d",
    "solvated_potential_from_structure_3d",
    "vdw_probe_occupancy",
    "vdw_radii_for_atomic_numbers",
    "voxel_centers_zyx",
]
