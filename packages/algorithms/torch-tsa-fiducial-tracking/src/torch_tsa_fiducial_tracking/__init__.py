"""Gold fiducial tracking through cryo-ET tilt series."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-tsa-fiducial-tracking")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Josh Dickerson"
__email__ = "jdickerson@mrc-lmb.cam.ac.uk"

from torch_tsa_fiducial_tracking.bead_correlation import (
    bead_correlation_maps,
    bead_template,
)
from torch_tsa_fiducial_tracking.track_fiducials import FiducialTracks, track_fiducials

__all__ = [
    "FiducialTracks",
    "bead_correlation_maps",
    "bead_template",
    "track_fiducials",
]
