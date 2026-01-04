"""Peak finding and fitting using torch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-find-peaks")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Johannes Elferich"
__email__ = "jojotux123@hotmail.com"

from .find_peaks import find_peaks_2d, find_peaks_3d
from .gaussians import Gaussian2D, Gaussian3D
from .refine_peaks import refine_peaks_2d, refine_peaks_3d

__all__ = [
    "Gaussian2D",
    "Gaussian3D",
    "__version__",
    "find_peaks_2d",
    "find_peaks_3d",
    "refine_peaks_2d",
    "refine_peaks_3d",
]
