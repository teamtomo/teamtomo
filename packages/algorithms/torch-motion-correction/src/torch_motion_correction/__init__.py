"""Motion estimation and correction in PyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-motion-correction")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_motion_correction.correct_motion import (
    correct_motion,
    correct_motion_fast,
    correct_motion_slow,
    correct_motion_two_grids,
    get_pixel_shifts,
)
from torch_motion_correction.data_io import (
    read_deformation_field_from_csv,
    write_deformation_field_to_csv,
)
from torch_motion_correction.deformation_field_utils import (
    evaluate_deformation_field,
)
from torch_motion_correction.estimate_motion_optimizer import estimate_local_motion
from torch_motion_correction.estimate_motion_xc import (
    estimate_global_motion,
    estimate_motion_cross_correlation_patches,
)

__all__ = [
    "correct_motion",
    "correct_motion_fast",
    "correct_motion_slow",
    "correct_motion_two_grids",
    "estimate_global_motion",
    "estimate_local_motion",
    "estimate_motion_cross_correlation_patches",
    "evaluate_deformation_field",
    "get_pixel_shifts",
    "read_deformation_field_from_csv",
    "write_deformation_field_to_csv",
]
