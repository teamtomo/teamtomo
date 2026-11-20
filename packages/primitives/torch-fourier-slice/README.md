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

## Rotation convention

The two layers interpret a rotation matrix `R` in opposite directions.

* **Real-space functions** (`project_3d_to_2d()`, `backproject_2d_to_3d()`, their
  multichannel versions, `project_2d_to_1d()`, and the real-space functions in
  `torch_fourier_slice.experimental`) take an **active** rotation. The projection is
  of the volume rotated by `R`, viewed along z (or along y for `project_2d_to_1d`).
  Backprojection takes the same matrices that produced the images.
* **Fourier-space functions** (`extract_central_slices_rfft_*()`,
  `insert_central_slices_rfft_*()`, and their experimental equivalents) take a
  **passive** rotation. `R` rotates the central slice's frequency coordinates from
  the image frame into the volume frame, so the slice extracted with `R` is that
  of the volume rotated by `R.T`.

For the same image, the Fourier-space functions therefore need the transpose of the
matrix the real-space functions take:

```python
project_3d_to_2d(volume, R)  # volume rotated by R, projected along z
extract_central_slices_rfft_3d(volume_rfft, R.mT)  # the same projection, in Fourier space
```

Matrices left-multiply column vectors containing xyz coordinates (zyx with
`zyx_matrices=True`; the experimental Mojo functions only take zyx matrices). If you
build `R` from ZYZ Euler angles, e.g. with
`scipy.spatial.transform.Rotation.from_euler("ZYZ", (phi, theta, psi))`, then
`R.T` corresponds to `("ZYZ", (-psi, -theta, -phi))`.

> **Changed in 0.7:** the real-space functions used to take the passive rotation,
> the same as the Fourier-space functions. To keep results from older code, pass
> `R.mT` (the transpose) to the real-space functions.

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
