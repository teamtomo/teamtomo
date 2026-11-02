# torch-find-peaks

[![License](https://img.shields.io/pypi/l/torch-find-peaks.svg?color=green)](https://github.com/teamtomo/teamtomo/blob/main/packages/primitives/torch-find-peaks/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-find-peaks.svg?color=green)](https://pypi.org/project/torch-find-peaks)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-find-peaks.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml)

Peak finding and fitting in PyTorch.

`find_peaks_2d` and `find_peaks_3d` detect local maxima in images and volumes,
and `refine_peaks_2d` and `refine_peaks_3d` refine each peak to subpixel
precision by fitting a Gaussian with gradient descent. Typical uses are picking
gold fiducials, or locating peaks in template matching or cross-correlation
maps.

## Installation

```bash
pip install torch-find-peaks
```

## Usage

```python
import torch
from torch_find_peaks import Gaussian2D, find_peaks_2d, refine_peaks_2d
from torch_grid_utils import coordinate_grid

# an image with three Gaussian blobs at known positions
grid = coordinate_grid(image_shape=(128, 128))
blobs = Gaussian2D(
    amplitude=torch.tensor([1.0, 0.8, 0.6]),
    center_y=torch.tensor([30.3, 64.0, 100.7]),
    center_x=torch.tensor([40.6, 80.2, 20.0]),
    sigma_y=torch.tensor([2.0, 2.0, 2.0]),
    sigma_x=torch.tensor([2.0, 2.0, 2.0]),
)
image = blobs(grid).sum(dim=0) + 0.05 * torch.randn(128, 128)

# integer peak positions (n_peaks, 2) in yx, and the image value at each
peak_coords, peak_heights = find_peaks_2d(image, min_distance=5, threshold_abs=0.3)

# subpixel positions from fitting a 2D Gaussian around each peak
refined = refine_peaks_2d(
    image, peak_coords, boxsize=11, amplitude=peak_heights, return_as="dataframe"
)
```

Peaks can be returned as tensors, NumPy arrays or pandas DataFrames
(`return_as="torch" | "numpy" | "dataframe"`). The `Gaussian2D` and `Gaussian3D`
modules used for fitting are exported too, for building synthetic test data as
above.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
