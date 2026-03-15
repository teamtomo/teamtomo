"""Erase masked regions from cryo-EM images"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-affine-utils")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from .erase import erase_region_2d

__all__ = ["erase_region_2d"]
