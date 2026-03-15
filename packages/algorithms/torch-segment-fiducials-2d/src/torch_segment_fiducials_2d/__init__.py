"""A small pre-trained residual U-Net for gold fiducial segmentation in cryo-EM images"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-segment-fiducials-2d")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from torch_segment_fiducials_2d._cli import cli
from torch_segment_fiducials_2d.dataset.download import download_training_data
from torch_segment_fiducials_2d.train import train_fiducial_segmentation_model

from torch_segment_fiducials_2d.predict import (
    predict_fiducial_probabilities,
    predict_fiducial_mask,
)

# add CLI commands
cli.command(name="download", no_args_is_help=True)(download_training_data)
cli.command(name="train", no_args_is_help=True)(train_fiducial_segmentation_model)

# expose only prediction API at top level
__all__ = [
    "predict_fiducial_probabilities",
    "predict_fiducial_mask",
]
