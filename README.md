# torch-fourier-slice

[![License](https://img.shields.io/pypi/l/torch-fourier-slice.svg?color=green)](https://github.com/teamtomo/torch-fourier-slice/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-fourier-slice.svg?color=green)](https://pypi.org/project/torch-fourier-slice)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-fourier-slice.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-fourier-slice/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-fourier-slice/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-fourier-slice/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-fourier-slice)

Fourier slice extraction/insertion from 2D images and 3D volumes in PyTorch.

## Overview

This package provides a simple API for back projection (reconstruction) and forward projection of 3D volumes using Fourier slice insertion and extraction. This can be done for

* single volumes with `project_3d_to_2d()` and `backproject_2d_to_3d()`
* and multichannel volumes with `project_3d_to_2d_multichannel()` and `backproject_2d_to_3d_multichannel()`

There are also some lower order layers in the package that run directly on Fourier transforms of volumes/images which can be relevant if the fourier transform can be precalculated:

* `extract_central_slices_rfft_3d()`, `extract_central_slices_rfft_3d_multichannel()`
* `insert_central_slices_rfft_3d()`, `insert_central_slices_rfft_3d_multichannel()`

The package also provides a use case for extracting common lines from 2D images with `project_2d_to_1d` which can be useful for tilt-axis angle optimization in cryo-ET.

## Installation

```bash
pip install torch-fourier-slice
```

## Usage

### Single volume
```python
import torch
from scipy.stats import special_ortho_group
from torch_fourier_slice import project_3d_to_2d, backproject_2d_to_3d

# start with a volume
volume = torch.rand((30, 30, 30))

# and some random rotations
rotation_matrices = torch.tensor(special_ortho_group.rvs(dim=3, size=10))
# shape is (10, 3, 3)

# forward project the volume, provides 10 projection images
projections = project_3d_to_2d(volume, rotation_matrices)
# shape is (10, 30, 30)

# we can backproject the 10 images to get the original volume back
reconstruction = backproject_2d_to_3d(projections, rotation_matrices)
# shape is (30, 30, 30)

# we can have an arbitrary number of leading dimensions for the rotations
rotation_matrices = torch.rand(3, 10, 3, 3)
projections = project_3d_to_2d(volume, rotation_matrices)
# shape is (3, 10, 30, 30)

# but for reconstruction it needs to match up with the projections
reconstruction = backproject_2d_to_3d(
    projections,  # (3, 10, 30, 30) 
    rotation_matrices  # (3, 10, 3, 3)
)
# shape is (30, 30, 30
```

### Multichannel volumes
```python
import torch
from scipy.stats import special_ortho_group
from torch_fourier_slice import project_3d_to_2d_multichannel, backproject_2d_to_3d_multichannel

# now we start with a multichannel 3d volume
volume = torch.rand((5, 30, 30, 30))

# and some random rotations
rotation_matrices = torch.tensor(special_ortho_group.rvs(dim=3, size=10))
# shape is (10, 3, 3)

# forward project the volume, provides 10 projection images with 5 channels each
projections = project_3d_to_2d_multichannel(volume, rotation_matrices)
# shape is (10, 5, 30, 30)

# we can backproject the 10 multichannel images to get the original multichannel volume back
reconstruction = backproject_2d_to_3d_multichannel(projections, rotation_matrices)
# shape is (5, 30, 30, 30)

# we can have an arbitrary number of trailing dimensions as well for multichannel data
rotation_matrices = torch.rand(3, 10, 3, 3)
projections = project_3d_to_2d_multichannel(volume, rotation_matrices)
# shape is (3, 10, 5, 30, 30)

# but for reconstruction it needs to match up with the projections
reconstruction = backproject_2d_to_3d_multichannel(
    projections,  # (3, 10, 5, 30, 30) 
    rotation_matrices  # (3, 10, 3, 3)
)
# shape is (5, 30, 30, 30)
```

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
