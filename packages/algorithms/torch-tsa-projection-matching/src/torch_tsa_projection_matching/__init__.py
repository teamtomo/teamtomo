"""Marker-free tilt-series alignment by projection matching."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-tsa-projection-matching")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Josh Dickerson"
__email__ = "jdickerson@mrc-lmb.cam.ac.uk"

from torch_tsa_projection_matching.backprojection import backproject, splat
from torch_tsa_projection_matching.projection_matching import (
    ProjectionMatchingResult,
    align_by_projection_matching,
)

__all__ = [
    "ProjectionMatchingResult",
    "align_by_projection_matching",
    "backproject",
    "splat",
]
