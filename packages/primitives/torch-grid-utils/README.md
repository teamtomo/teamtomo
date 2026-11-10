# torch-grid-utils

[![License](https://img.shields.io/pypi/l/torch-grid-utils.svg?color=green)](https://github.com/teamtomo/teamtomo/blob/main/packages/primitives/torch-grid-utils/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-grid-utils.svg?color=green)](https://pypi.org/project/torch-grid-utils)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-grid-utils.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml)

*torch-grid-utils* provides grids for 2D/3D image manipulations in PyTorch:
coordinate grids, Fourier-space frequency grids, masks of simple shapes, polar
grids, and patch grids for splitting images into (overlapping) tiles.

All grids follow the TeamTomo conventions: arrays are `(h, w)` or `(d, h, w)`,
coordinates are ordered `yx` / `zyx`, and frequencies are in cycles per pixel.

## Installation

```bash
pip install torch-grid-utils
```

## Usage

```python
import torch
from torch_grid_utils import circle, coordinate_grid, fftfreq_grid, patch_grid, sphere

# (h, w, 2) array of yx coordinates, centred on the middle of a 64 x 64 image
coords = coordinate_grid(image_shape=(64, 64), center=(32, 32))
distances = torch.linalg.norm(coords, dim=-1)  # (64, 64) distance from the centre

# masks with a soft edge
disc = circle(radius=20, image_shape=(64, 64), smoothing_radius=4)  # (64, 64)
ball = sphere(radius=20, image_shape=(64, 64, 64))  # (64, 64, 64)

# frequency of each element of the rfft of a 64 x 64 image, in cycles/pixel
freqs = fftfreq_grid(image_shape=(64, 64), rfft=True)  # (64, 33, 2)
freq_magnitude = fftfreq_grid(image_shape=(64, 64), rfft=True, norm=True)  # (64, 33)

# split a stack of images into overlapping patches
images = torch.rand(10, 512, 512)
patches, patch_centers = patch_grid(images, patch_shape=(128, 128), patch_step=(64, 64))
```

### What's included

- **Coordinate grids:** `coordinate_grid`, `image_center`.
- **Fourier-space grids:** `fftfreq_grid`, `fftfreq_grid_polar`, `transform_fftfreq_grid`,
  `dft_center`, `rfft_shape`, `next_fft_size`, and `fftshift_*d`/`ifftshift_*d` that
  also handle `rfft` arrays.
- **Shapes:** `circle`, `rectangle`, `square`, `wedge` (2D) and `sphere`, `cuboid`,
  `cube`, `cone` (3D), each with optional soft edges.
- **Polar grids:** `polar_grid`, `cartesian_to_polar`, `polar_to_cartesian`.
- **Patch grids:** `patch_grid`, `patch_grid_lazy`, `patch_grid_centers`,
  `patch_grid_indices`.
- **Unit conversion:** `fftfreq_to_spatial_frequency`, `spatial_frequency_to_fftfreq`.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
