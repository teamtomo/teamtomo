"""Fit tilt-series alignment to tracked 2D positions of 3D points."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-tsa-fit")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Josh Dickerson"
__email__ = "jdickerson@mrc-lmb.cam.ac.uk"

from torch_tsa_fit.fit_tilt_series import TiltAlignResult, fit_tilt_series
from torch_tsa_fit.triangulate_points import affine_projection, triangulate_points

__all__ = [
    "TiltAlignResult",
    "affine_projection",
    "fit_tilt_series",
    "triangulate_points",
]
