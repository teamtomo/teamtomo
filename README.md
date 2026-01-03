# torch-image-interpolation

[![License](https://img.shields.io/pypi/l/torch-image-interpolation.svg?color=green)](https://github.com/teamtomo/torch-image-interpolation/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-image-interpolation.svg?color=green)](https://pypi.org/project/torch-image-interpolation)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-image-interpolation.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-image-interpolation/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-image-interpolation/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-image-interpolation/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-image-interpolation)

1D/2D/3D image interpolation routines in PyTorch.

## Why?

This package provides a simple, consistent API for

- sampling values from 1D/2D/3D images (`sample_image_1d()`/`sample_image_2d()`/`sample_image_3d()`)
- inserting values into 2D/3D images (`insert_into_image_1d()`/`insert_into_image_2d()`/`insert_into_image_3d()`)

Operations are differentiable, multichannel data and complex valued images are supported.

[`torch.nn.functional.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)
is used under the hood for sampling.

# Installation

```shell
pip install torch-image-interpolation
```

# Usage

## Coordinate System

This library uses an array-like coordinate system where coordinate values span from `0`
to `dim_size - 1` for each dimension.
Fractional coordinates are supported and values are interpolated appropriately.

### 1D Images

For 1D images (vectors) with shape `(w, )` or `(c, w)` 
coordinates are a `(...)` array of coordinates in the `w` dimension.

For example, in a `(28, )` image, valid coordinates range from `0` to `27`.

### 2D Images

For 2D images with shape `(h, w)` or `(c, h, w)`:

Coordinates are ordered as `[y, x]` where:

- `y` is the position in the height dimension (first dimension of shape)
- `x` is the position in the width dimension (second dimension of shape)

For example, in a `(28, 28)` image, valid coordinates range from `[0, 0]` to `[27, 27]`

### 3D Images

For 3D images with shape `(d, h, w)` or `(c, d, h, w)`:

Coordinates are ordered as `[z, y, x]` where:

- `z` is the position in the depth dimension (first dimension of shape)
- `y` is the position in the height dimension (second dimension of shape)
- `x` is the position in the width dimension (third dimension of shape)

For example, in a `(28, 28, 28)` volume, valid coordinates range from `[0, 0, 0]` to
`[27, 27, 27]`.

## Sample from image

```python
import torch
import numpy as np
from torch_image_interpolation import sample_image_2d

# example (h, w) image
image = torch.rand((28, 28))

# make an arbitrary stack (..., 2) of 2d coords
coords = torch.tensor(np.random.uniform(low=0, high=27, size=(6, 7, 8, 2))).float()

# sampling returns a (6, 7, 8) array of samples
# using bilinear interpolation (the default)
sample_bilinear = sample_image_2d(image=image, coordinates=coords, interpolation='bilinear')

# or a different interpolation mode...
samples_nearest = sample_image_2d(image=image, coordinates=coords, interpolation='nearest')
samples_bicubic = sample_image_2d(image=image, coordinates=coords, interpolation='bicubic')
```

The API is identical for 1D (`w`) or 3D `(d, h, w)` images but takes `(...)` or `(..., 3)` arrays of
coordinates respectively.

Sampling is supported for multichannel images in 1D `(c, w)`, 2D `(c, h, w)` and 3D `(c, d, h, w)`. 
Sampling multichannel images returns `(..., c)` arrays of values.

## Insert into image

```python
import torch
import numpy as np
from torch_image_interpolation import insert_into_image_2d

# example (h, w) image
image = torch.zeros((28, 28))

# make an arbitrary stack (..., 2) of 2d coords
coords = torch.tensor(np.random.uniform(low=0, high=27, size=(3, 4, 2)))

# generate random values to place at coords
values = torch.rand(size=(3, 4))

# insert values into the image with bilinear interpolation (the default)
image_bilinear, weights_bilinear = insert_into_image_2d(
    values, image=image, coordinates=coords, interpolation='bilinear'
)

# you can specify a different interpolation mode
image_nearest, weights_nearest = insert_into_image_2d(
    values, image=image, coordinates=coords, interpolation='nearest'
)
```

The API is identical for 1D `(w, )` and 3D `(d, h, w)` images but requires `(...)` and (..., 3)` arrays of
coordinates respectively.

Insertion is supported for multichannel images in 1D `(c, w)`, 2D `(c, h, w)` and 3D `(c, d, h, w)`. 
Inserting into multichannel images requires `(..., c)` arrays of values.


## Similar packages

- https://github.com/balbasty/torch-interpol




