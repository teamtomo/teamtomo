"""Utilities for affine transformations of 2d/3d coordinates in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-affine-utils")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_affine_utils.utils import homogenise_coordinates

__all__ = [
    "homogenise_coordinates",
]