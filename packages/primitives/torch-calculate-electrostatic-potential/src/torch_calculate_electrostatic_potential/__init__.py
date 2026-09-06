"""Cryo-EM Electrostatic Potential computation with PyTorch."""

from importlib.metadata import PackageNotFoundError, version

from .atom_stack import AtomStack
from .grid import GridConfig, default_sublattice_radius
from .potential import (
    PENG_SCATTERING_TO_POTENTIAL,
    calculate_scattering_potential_2d,
    calculate_scattering_potential_3d,
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
    "PENG_SCATTERING_TO_POTENTIAL",
    "AtomStack",
    "BondedScatteringFactorTable",
    "GridConfig",
    "default_sublattice_radius",
    "__version__",
    "calculate_scattering_potential_2d",
    "calculate_scattering_potential_3d",
    "get_peng_scattering_parameters",
    "potential_from_structure_2d",
    "potential_from_structure_3d",
    "resolve_scattering_parameters",
]
