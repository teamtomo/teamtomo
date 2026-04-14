"""Generate uniform 3D euler angles (ZYZ)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-so3")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"

from .angular_ranges import SymmetryRanges, get_symmetry_ranges
from .base_s2_grid import (
    cartesian_base_grid,
    healpix_base_grid,
    healpix_sectored_base_grid,
    uniform_base_grid,
)
from .local_so3_sampling import get_local_high_resolution_angles, get_roll_angles
from .uniform_so3_sampling import get_sectored_euler_angles, get_uniform_euler_angles

__all__ = [
    "SymmetryRanges",
    "cartesian_base_grid",
    "get_local_high_resolution_angles",
    "get_roll_angles",
    "get_sectored_euler_angles",
    "get_symmetry_ranges",
    "get_uniform_euler_angles",
    "healpix_base_grid",
    "healpix_sectored_base_grid",
    "uniform_base_grid",
]
