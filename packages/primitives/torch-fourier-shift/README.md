# torch-phase-shift

[![License](https://img.shields.io/pypi/l/torch-fourier-shift.svg?color=green)](https://github.com/teamtomo/teamtomo/blob/main/packages/primitives/torch-fourier-shift/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-fourier-shift.svg?color=green)](https://pypi.org/project/torch-fourier-shift)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-fourier-shift.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml)

*torch-fourier-shift* is a package for shifting 1D, 2D and 3D images with subpixel precision 
by applying phase shifts to Fourier transforms in PyTorch.

<p align="center" width="100%">
  <img src="./docs/assets/shift_2d_image.png" alt="A 2D image and the shifted result" width="50%">
</p>

```python
import torch
from torch_fourier_shift import fourier_shift_image_2d

# create a dummy image
my_image = torch.tensor(
    [[0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]]
)

# shift the image by 1 pixel in dim 0, 2 pixels in dim 1
shifted_image = fourier_shift_image_2d(image=my_image, shifts=torch.tensor([1, 2]))
```

API's are equivalent for 1D and 3D images.

## Installation

*torch-fourier-shift* is available on PyPI.

```shell
pip install torch-fourier-shift
```

## Usage

Please check the the docs at [teamtomo.org/torch-fourier-shift](https://teamtomo.org/torch-fourier-shift/)



