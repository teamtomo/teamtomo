"""Generate uniform 3D euler angles (ZYZ)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-so3")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"

from .angular_ranges import SymmetryRanges, get_symmetry_ranges
from .base_s2_grid import cartesian_base_grid, healpix_base_grid, uniform_base_grid
from .local_so3_sampling import get_local_high_resolution_angles, get_roll_angles
from .uniform_so3_sampling import get_uniform_euler_angles

__all__ = [
    "get_uniform_euler_angles",
    "get_local_high_resolution_angles",
    "get_roll_angles",
    "get_symmetry_ranges",
    "SymmetryRanges",
    "uniform_base_grid",
    "healpix_base_grid",
    "cartesian_base_grid",
]
