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
from torch_motion_correction.deformation_field import DeformationField
from torch_motion_correction.estimate_motion_optimizer import estimate_local_motion
from torch_motion_correction.estimate_motion_xc import (
    estimate_global_motion,
    estimate_motion_cross_correlation_patches,
)
from torch_motion_correction.optimization_state import (
    OptimizationState,
    OptimizationTracker,
)
from torch_motion_correction.types import (
    FourierFilterConfig,
    OptimizationConfig,
    PatchSamplingConfig,
    XCRefinementConfig,
)

__all__ = [
    "DeformationField",
    "FourierFilterConfig",
    "OptimizationConfig",
    "OptimizationState",
    "OptimizationTracker",
    "PatchSamplingConfig",
    "XCRefinementConfig",
    "correct_motion",
    "correct_motion_fast",
    "correct_motion_slow",
    "correct_motion_two_grids",
    "estimate_global_motion",
    "estimate_local_motion",
    "estimate_motion_cross_correlation_patches",
    "get_pixel_shifts",
]
