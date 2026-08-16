"""Multislice electron scattering simulation in PyTorch."""

from importlib.metadata import PackageNotFoundError, version

from torch_scattering._core import (
    chunk_slices,
    fresnel_propagator,
    interaction_parameter,
    multislice_step,
    transmission_function,
)
from torch_scattering.firstborn import firstborn
from torch_scattering.multislice import multislice
from torch_scattering.projection import projection
from torch_scattering.rytov import rytov

try:
    __version__ = version("torch-scattering")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "chunk_slices",
    "firstborn",
    "fresnel_propagator",
    "interaction_parameter",
    "multislice",
    "multislice_step",
    "projection",
    "rytov",
    "transmission_function",
]
