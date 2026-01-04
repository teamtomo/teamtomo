"""Extract 2D/3D subimages with subpixel precision in PyTorch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-subpixel-crop")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from .subpixel_crop_2d import subpixel_crop_2d
from .subpixel_crop_3d import subpixel_crop_3d
