# torch-transform-image

[![License](https://img.shields.io/pypi/l/torch-transform-image.svg?color=green)](https://github.com/teamtomo/torch-transform-image/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-transform-image.svg?color=green)](https://pypi.org/project/torch-transform-image)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-transform-image.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-transform-image/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-transform-image/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-transform-image/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-transform-image)

Real space transformations of 2D/3D images in PyTorch

## Motivation

This package provides a simple, consistent API for applying affine transformations to 2D/3D images in PyTorch. 
It enables efficient, GPU-accelerated geometric transformations of images.

## Installation

```bash
pip install torch-transform-image
```

## Features

- Apply arbitrary affine transformations to 2D and 3D images
- Support for various interpolation methods (nearest, bilinear, bicubic for 2D; nearest, trilinear for 3D)
- Batched operations for efficient processing
- Fully differentiable operations compatible with PyTorch's autograd

## Coordinate System

This package uses the same coordinate system as NumPy/PyTorch array indexing:
- For 2D images: coordinates are ordered as `[y, x]` for dimensions `(height, width)`
- For 3D images: coordinates are ordered as `[z, y, x]` for dimensions `(depth, height, width)`

Transformation matrices left-multiply homogeneous pixel coordinates (`[y, x, 1]` for 2D and `[z, y, x, 1]` for 3D).

### Generating Transformation Matrices

The companion package [torch-affine-utils](https://github.com/teamtomo/torch-affine-utils) provides convenient functions 
to generate transformation matrices that work with homogenous pixel coordinates (`yxw`/`zyxw`):

```python
from torch_affine_utils.transforms_2d import R, T, S  # Rotation, Translation, Scale for 2D
from torch_affine_utils.transforms_3d import Rx, Ry, Rz, T, S  # Rotation, Translation, Scale for 3D
```

## Usage

### 2D Transformations

```python
import torch
from torch_transform_image import affine_transform_image_2d
from torch_affine_utils.transforms_2d import R, T, S  # Rotation, Translation, Scale

# Create a test image (28×28)
image = torch.zeros((28, 28), dtype=torch.float32)
image[14, 14] = 1  # Place a dot at the center

# Create a transformation matrix to translate coordinates 4 pixels in y direction
translation = T([4, 0])  # Uses [y, x] coordinate order matching dimensions (h, w)

# Apply the transformation
result = affine_transform_image_2d(
    image=image,
    matrices=translation, 
    interpolation='bilinear',  # Options: 'nearest', 'bilinear', 'bicubic'
    yx_matrices=True,  # The generated translations have [y, x] order
)

# Compose multiple transformations
# First translate to origin, then rotate, then translate back
T1 = T([-14, -14])  # Move center to origin
R1 = R(45, yx=True)  # Rotate 45 degrees
T2 = T([14, 14])  # Move back
transform = T2 @ R1 @ T1  # Matrix composition (applied right-to-left)

# Apply the composed transformation
rotated = affine_transform_image_2d(
    image=image,
    matrices=transform,
    interpolation='bicubic',
    yx_matrices=True,
)
```

### 3D Transformations

```python
import torch
from torch_transform_image import affine_transform_image_3d
from torch_affine_utils.transforms_3d import R, T, S  # Rotation, Translation, Scale

# Create a test volume (64×64×64)
volume = torch.zeros((64, 64, 64), dtype=torch.float32)
volume[32, 32, 32] = 1  # Place a dot at the center

# Create a transformation matrix (translate coordinates 5 voxels in z direction)
translation = T([5, 0, 0])  # Uses [z, y, x] coordinate order matching dimensions (d, h, w)

# Apply the transformation
result = affine_transform_image_3d(
    image=volume,
    matrices=translation, 
    interpolation='trilinear',  # Options: 'nearest', 'trilinear'
    zyx_matrices=True,  # The generated translations have [z, y, x] order
)
```

## How It Works

Under the hood, the package:
1. Creates a coordinate grid for the output image
2. Applies the transformation matrix to these coordinates
3. Samples the input image at the transformed coordinates using the specified interpolation method

This can be unintuitive at times, in the 2D example above the dot will move to `[10,14]` in the output.
If you want the input image to be transformed according to the matrix, you can use `torch.linalg.inv` to inverse the direction of the transform.

All operations are performed in PyTorch, making them fully differentiable and GPU-compatible.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.