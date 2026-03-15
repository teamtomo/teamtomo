# torch-2dtm

[![License](https://img.shields.io/pypi/l/torch-2dtm.svg?color=green)](https://github.com/teamtomo/torch-2dtm/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-2dtm.svg?color=green)](https://pypi.org/project/torch-2dtm)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-2dtm.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-2dtm/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-2dtm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-2dtm/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-2dtm)

## Overview

torch-2dtm is a Python package for efficient templating matching of 
2D projections of a 3D template with a 2D image in PyTorch.

This is implemented for cryo-EM applications, see 
[Rickgauer et al. 2017 eLife](https://doi.org/10.7554/eLife.25648) for details.

## Features

- Fast 2D template matching using Fourier transforms
- Batch processing over orientations
- Batch processing over Fourier space filters (e.g. for defocus sweeps)
- GPU acceleration through PyTorch

Projections are calculated on-the-fly using 
[*torch-fourier-slice*](https://github.com/teamtomo/torch-fourier-slice).

## Installation

```bash
pip install torch-2dtm
```

## Basic Usage

```python
import torch
import torch_2dtm
from scipy.stats import special_ortho_group

# Create random test data
# 1. Create a random image and compute its FFT
image_size = (128, 128)
image = torch.randn(*image_size, dtype=torch.float32)
image_dft = torch.fft.rfftn(image, dim=(0, 1))  # Shape: (128, 65)

# 2. Create a random 3D template and compute its FFT
template_size = (64, 64, 64)
template = torch.randn(*template_size, dtype=torch.float32)
template_dft = torch.fft.rfftn(template, dim=(0, 1, 2))  # Shape: (64, 64, 33)

# 3. Create a batch of random rotation matrices with shape (b, 3, 3)
num_orientations = 10
rotation_matrices = torch.tensor(special_ortho_group.rvs(size=num_orientations, dim=3), dtype=torch.float32)

# 4. Create an arbitrary stack of Fourier space filters (identity filter in this example)
# These filters operate on rffts of the 2D projection images
# Filter shape: (..., h, w // 2 + 1)
filters = torch.ones(template_size[0], template_size[1] // 2 + 1, dtype=torch.complex64)

# Perform template matching
cross_correlation = torch_2dtm.match_template_dft_2d(
    image_dft=image_dft,
    template_dft=template_dft,
    rotation_matrices=rotation_matrices,
    filters=filters
)
# The result has shape (..., num_orientations, image_height, image_width)
```

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.