"""PyTorch Lightning application for tomographic boundary segmentation.

This package provides a complete pipeline for training and applying deep learning
models to detect slab boundaries in 3D tomographic volumes. It includes data
processing, model training, and inference capabilities built on PyTorch Lightning
and segmentation-models-pytorch.

Main Components:

- TomoSlabTrainer: Handles model training with PyTorch Lightning
- TomoSlabPredictor: Performs inference and generates boundary masks
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-tomo-slab")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Pranav NM Shah"
__email__ = "p.shah.lab@gmail.com"
