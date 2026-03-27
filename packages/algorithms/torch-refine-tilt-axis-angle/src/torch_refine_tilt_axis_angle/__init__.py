"""Tilt-axis angle optimization using common lines in PyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-refine-tilt-axis-angle")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Marten Chaillet"
__email__ = "martenchaillet@gmail.com"


from torch_refine_tilt_axis_angle.refine_tilt_axis_angle import refine_tilt_axis_angle

__all__ = ["refine_tilt_axis_angle"]
