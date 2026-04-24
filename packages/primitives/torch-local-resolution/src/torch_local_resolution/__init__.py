"""Core functionality for local resolution estimation of cryo-EM half-maps."""

from importlib.metadata import PackageNotFoundError, version

from .estimate_local_resolution import (
    compute_correlation,
    compute_resolution,
    estimate_local_resolution,
)
from .input_models import ComputeResolutionInput

try:
    __version__ = version("torch-local-resolution")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "ComputeResolutionInput",
    "compute_correlation",
    "compute_resolution",
    "estimate_local_resolution",
]
