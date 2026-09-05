"""Tilt series data structure, projection and subtilt extraction for cryo-ET."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-tilt-series")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Marten Chaillet, Davide Torre"
__email__ = "martenchaillet@gmail.com, davidetorre99@gmail.com"

from torch_tilt_series.io import (
    from_aretomo_output,
    from_etomo_directory,
    load_tilt_series_images,
)
from torch_tilt_series.preprocessing import preprocess_tilt_series_images
from torch_tilt_series.tilt_series import TiltSeries
from torch_tilt_series.utils import normalize_on_central_crop, subtract_plane

TiltSeries.from_aretomo_output = classmethod(
    lambda cls, *args, **kwargs: from_aretomo_output(*args, **kwargs)
)
TiltSeries.from_etomo_directory = classmethod(
    lambda cls, *args, **kwargs: from_etomo_directory(*args, **kwargs)
)

__all__ = [
    "TiltSeries",
    "load_tilt_series_images",
    "normalize_on_central_crop",
    "preprocess_tilt_series_images",
    "subtract_plane",
]
