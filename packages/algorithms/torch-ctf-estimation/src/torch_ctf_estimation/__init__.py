"""Contrast transfer function estimation for cryo-EM images in PyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-ctf-estimation")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_ctf_estimation.estimate_ctf import (
    CTFEstimationResult,
    estimate_ctf,
    estimate_ctf_and_thickness,
)
from torch_ctf_estimation.estimate_ctf_1d import (
    estimate_ctf_1d,
    estimate_thickness_1d,
    fit_background_spline_1d,
    refine_defocus_and_thickness_1d,
)
from torch_ctf_estimation.estimate_ctf_2d import (
    estimate_ctf_2d,
    estimate_defocus_2d_at_1x1,
    estimate_thickness_2d,
    refine_defocus_and_thickness_2d,
)
from torch_ctf_estimation.models import (
    CTFFittingParams,
    Defocus1DResults,
    Defocus2DResults,
    LaserParams,
    OpticalParams,
    Thickness1DResults,
    Thickness2DResults,
    ThicknessParams,
)
from torch_ctf_estimation.utils.defocus_field_from_1d import defocus_field_from_1d_fits
from torch_ctf_estimation.utils.laser_axis_mask import (
    apply_laser_axis_mask,
    build_laser_axis_mask,
)
from torch_ctf_estimation.utils.patches import (
    compute_patch_power_spectra,
    extract_ctf_patches,
    normalised_patch_positions,
)
from torch_ctf_estimation.utils.prepare_image import prepare_image_for_ctf

__all__ = [
    "CTFEstimationResult",
    "CTFFittingParams",
    "Defocus1DResults",
    "Defocus2DResults",
    "LaserParams",
    "OpticalParams",
    "Thickness1DResults",
    "Thickness2DResults",
    "ThicknessParams",
    "apply_laser_axis_mask",
    "build_laser_axis_mask",
    "compute_patch_power_spectra",
    "defocus_field_from_1d_fits",
    "estimate_ctf",
    "estimate_ctf_1d",
    "estimate_ctf_2d",
    "estimate_ctf_and_thickness",
    "estimate_defocus_2d_at_1x1",
    "estimate_thickness_1d",
    "estimate_thickness_2d",
    "extract_ctf_patches",
    "fit_background_spline_1d",
    "normalised_patch_positions",
    "prepare_image_for_ctf",
    "refine_defocus_and_thickness_1d",
    "refine_defocus_and_thickness_2d",
]
