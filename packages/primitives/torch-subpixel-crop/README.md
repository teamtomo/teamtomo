# torch-subpixel-crop

[![License](https://img.shields.io/pypi/l/torch-subpixel-crop.svg?color=green)](https://github.com/teamtomo/teamtomo/blob/main/packages/primitives/torch-subpixel-crop/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-subpixel-crop.svg?color=green)](https://pypi.org/project/torch-subpixel-crop)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-subpixel-crop.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml)

Crop 2D/3D images with subpixel precision in PyTorch.

Each patch is extracted around the nearest whole pixel and then shifted in
Fourier space by the remaining subpixel offset, so the requested position ends
up exactly at the centre of the patch (pixel `sidelength // 2`).

## Installation

```bash
pip install torch-subpixel-crop
```

## Usage

```python
import torch
from torch_subpixel_crop import subpixel_crop_2d, subpixel_crop_3d

image = torch.rand(1024, 1024)
positions = torch.tensor([[100.5, 200.25], [512.0, 512.0]])  # (n, 2) yx, in pixels

# (n, 64, 64) patches centred on each position, to subpixel precision
patches = subpixel_crop_2d(image=image, positions=positions, sidelength=64)

volume = torch.rand(128, 128, 128)
subvolumes = subpixel_crop_3d(image=volume, positions=torch.tensor([[64.5, 60.0, 70.2]]), sidelength=32)
```

Positions are in pixels and ordered `yx` (2D) or `zyx` (3D); regions of a patch
that fall outside the image are filled with zeros. An optional `mask`
is applied to each patch, and `return_rfft=True` returns the patches' rfft
instead, saving a transform when the next step works in Fourier space.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
