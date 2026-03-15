"""MONAI-based model factory for tomographic boundary segmentation.

This module provides simplified MONAI U-Net models for boundary detection.
"""

import torch.nn as nn
from monai.networks.nets import UNet


def create_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    channels: tuple = (32, 64, 128, 256, 512),
    strides: tuple = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Create a MONAI U-Net model for 2D boundary segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels (classes)
    channels : tuple
        Number of channels in each level
    strides : tuple
        Stride values for downsampling
    num_res_units : int
        Number of residual units per level
    dropout : float
        Dropout probability

    Returns
    -------
    nn.Module
        MONAI U-Net model
    """
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        dropout=dropout,
    )

    return model
