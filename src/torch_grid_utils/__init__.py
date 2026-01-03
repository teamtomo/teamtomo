"""Grids for 2D/3D image manipulations in PyTorch."""

from importlib.metadata import PackageNotFoundError, version

from torch_grid_utils.coordinate_grid import coordinate_grid, image_center
from torch_grid_utils.fftfreq_grid import (
    dft_center,
    fftfreq_grid,
    fftfreq_to_spatial_frequency,
    fftshift_1d,
    fftshift_2d,
    fftshift_3d,
    ifftshift_1d,
    ifftshift_2d,
    ifftshift_3d,
    rfft_shape,
    spatial_frequency_to_fftfreq,
    transform_fftfreq_grid,
)
from torch_grid_utils.patch_grid import (
    patch_grid,
    patch_grid_centers,
    patch_grid_indices,
    patch_grid_lazy,
)
from torch_grid_utils.polar_grid import (
    cartesian_to_polar,
    fftfreq_grid_polar,
    normalize_polar_grid,
    polar_grid,
    polar_to_cartesian,
)
from torch_grid_utils.shapes_2d import circle, rectangle, square, wedge
from torch_grid_utils.shapes_3d import cone, cube, cuboid, sphere

try:
    __version__ = version("torch-grid-utils")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

__all__ = [
    "cartesian_to_polar",
    "circle",
    "cone",
    "coordinate_grid",
    "cube",
    "cuboid",
    "dft_center",
    "fftfreq_grid",
    "fftfreq_grid_polar",
    "fftfreq_to_spatial_frequency",
    "fftshift_1d",
    "fftshift_2d",
    "fftshift_3d",
    "ifftshift_1d",
    "ifftshift_2d",
    "ifftshift_3d",
    "image_center",
    "normalize_polar_grid",
    "patch_grid",
    "patch_grid_centers",
    "patch_grid_indices",
    "patch_grid_lazy",
    "polar_grid",
    "polar_to_cartesian",
    "rectangle",
    "rfft_shape",
    "spatial_frequency_to_fftfreq",
    "sphere",
    "square",
    "transform_fftfreq_grid",
    "wedge",
]
