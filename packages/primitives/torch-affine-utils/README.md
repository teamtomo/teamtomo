# torch-affine-utils

[![License](https://img.shields.io/pypi/l/torch-affine-utils.svg?color=green)](https://github.com/teamtomo/torch-affine-utils/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-affine-utils.svg?color=green)](https://pypi.org/project/torch-affine-utils)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-affine-utils.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-affine-utils/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-affine-utils/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-affine-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-affine-utils)

A small utility library for generating affine matrices for 2D and 3D coordinates.

## Overview

`torch-affine-utils` provides an easy, intuitive API for generating affine transformation matrices in PyTorch. 
These matrices are often used in computer graphics and imaging applications where coordinate transformations are frequent.

The library supports:
- 2D transformations (rotation, translation, scaling)
- 3D transformations (rotation around X, Y, Z axes, translation, scaling)
- Batched operations for efficient processing
- Support for different coordinate conventions (xyw/yxw for 2D, xyzw/zyxw for 3D)

## Installation

```bash
pip install torch-affine-utils
```

## Usage Examples

### 2D Transformations

```python
import einops
import torch
from torch_affine_utils.transforms_2d import R, T, S

# Create a rotation matrix (45 degrees)
rotation = R(torch.tensor([45.0]))

# Create a translation matrix
translation = T(torch.tensor([[2.0, 3.0]]))

# Create a scaling matrix
scaling = S(torch.tensor([[2.0, 3.0]]))

# Chain transformations (apply scaling, then rotation, then translation)
transform = translation @ rotation @ scaling

# Apply to a batch of 2D coordinates
coords = torch.tensor([
    [1.0, 1.0, 1.0],  # Homogeneous coordinates (x, y, w)
    [1.0, 0.0, 1.0]
])  
coords = einops.rearrange(coords, 'b xyw -> b xyw 1')
transformed_coords = transform @ coords
```

### 3D Transformations

```python
import einops
import torch
from torch_affine_utils.transforms_3d import Rx, Ry, Rz, T, S

# Create rotation matrices around each axis
rot_x = Rx(torch.tensor([30.0]))  # 30 degrees around X axis
rot_y = Ry(torch.tensor([45.0]))  # 45 degrees around Y axis
rot_z = Rz(torch.tensor([60.0]))  # 60 degrees around Z axis

# Create a translation and scaling matrix
translation = T(torch.tensor([[1.0, 2.0, 3.0]]))
scaling = S(torch.tensor([[2.0, 2.0, 2.0]]))

# Chain transformations
transform = translation @ rot_z @ rot_y @ rot_x @ scaling

# Apply to a batch of 3D coordinates
coords = torch.tensor([
    [1.0, 1.0, 1.0, 1.0],  # Homogeneous coordinates (x, y, z, w)
    [1.0, 0.0, 0.0, 1.0]
])
coords = einops.rearrange(coords, 'b xyzw -> b xyzw 1')
transformed_coords = transform @ coords
```

### Batched Operations

The library supports batched operations for efficient processing:

```python
import torch
from torch_affine_utils.transforms_2d import R

# Batch of rotation angles
angles = torch.tensor([0.0, 30.0, 45.0, 60.0, 90.0])

# Create batch of 2D rotation matrices
rotation_matrices = R(angles)  # Shape: (5, 3, 3)

from torch_affine_utils.transforms_3d import T

# Batch of 3D translations
translations = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
])

# Create batch of translation matrices
translation_matrices = T(translations)  # Shape: (4, 4, 4)
```

## Homogeneous Coordinates
The package provides a helper function to convert standard coordinates to homogeneous coordinates

```python
import torch
from torch_affine_utils import homogenise_coordinates

# For 2D points
points_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
homogeneous_2d = homogenise_coordinates(points_2d)
# Result: tensor([[1.0, 2.0, 1.0],
#                 [3.0, 4.0, 1.0],
#                 [5.0, 6.0, 1.0]])

# For 3D points
points_3d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
homogeneous_3d = homogenise_coordinates(points_3d)
# Result: tensor([[1.0, 2.0, 3.0, 1.0],
#                 [4.0, 5.0, 6.0, 1.0]])

# Works with any batch dimensions
points_batched = torch.randn(2, 3, 5, 3)  # Shape: (2, 3, 5, 3) - batch of 3D points
homogeneous_batched = homogenise_coordinates(points_batched)
# Result shape: (2, 3, 5, 4) - added homogeneous coordinate
```

## Coordinate Systems

The library supports multiple coordinate conventions:

- For 2D:
  - `xyw` (default): Standard Cartesian coordinates
  - `yxw`: Alternative ordering (useful for 2D image coordinates)

- For 3D:
  - `xyzw` (default): Standard right-handed Cartesian coordinates
  - `zyxw`: Alternative ordering (useful for 3D image coordinates)

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
