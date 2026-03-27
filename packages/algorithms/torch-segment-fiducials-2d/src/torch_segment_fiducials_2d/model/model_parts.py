from typing import Optional

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F


class Conv3x3(nn.Conv2d):
    """3x3 convolution module."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
            bias=False,
            dilation=1,
        )


class Conv1x1(nn.Conv2d):
    """1x1 convolution module."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )


class ResBlock(nn.Module):
    (
        """Residual block.

    input -> Conv2d (3x3) -> BatchNorm2d -> ReLU -> Conv2d (3x3)
                                                       |
             +input                                    v
    ReLU    <-------  [optional downsampling]  <-  BatchNorm2d
    """
        ""
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample downsample the input when stride != 1
        self.conv1 = Conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class DoubleConv(nn.Module):
    """(convolution -> [BN] -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Multiple ResidualBlocks with optional downsampling in the first block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_residual_blocks: int,
        stride: int = 1,
    ):
        super().__init__()
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        for i in range(n_residual_blocks):
            if i == 0:
                layers.append(
                    ResBlock(
                        in_channels,
                        out_channels,
                        stride=stride,
                        downsample=self.downsample,
                    )
                )
            else:
                layers.append(ResBlock(out_channels, out_channels))

        self.forward_impl = nn.Sequential(*layers)

    def forward(self, x):
        return self.forward_impl(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.post_up_conv = Conv1x1(in_channels, in_channels // 2)
        self.post_cat_conv = DoubleConv(in_channels, out_channels, in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.post_up_conv(x1)
        # pad input to (b, c, h, w)
        dy = x2.size()[-2] - x1.size()[-2]
        dx = x2.size()[-1] - x1.size()[-1]
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.post_cat_conv(x)
