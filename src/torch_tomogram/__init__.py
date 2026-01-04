"""(sub-)Tomogram reconstruction, subtilt extraction for cryo-ET."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-tomogram")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Marten Chaillet"
__email__ = "martenchaillet@gmail.com"

from torch_tomogram.tomogram import Tomogram

__all__ = ["Tomogram"]
