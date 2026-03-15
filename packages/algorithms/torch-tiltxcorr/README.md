# torch-tiltxcorr

[![License](https://img.shields.io/pypi/l/torch-tiltxcorr.svg?color=green)](https://github.com/teamtomo/torch-tiltxcorr/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-tiltxcorr.svg?color=green)](https://pypi.org/project/torch-tiltxcorr)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-tiltxcorr.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-tiltxcorr/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-tiltxcorr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-tiltxcorr/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-tiltxcorr)

Cross correlation with image stretching for coarse alignment of cryo-EM tilt series data in PyTorch.

## Overview

torch-tiltxcorr reimplements the 
[IMOD](https://bio3d.colorado.edu/imod/) program 
[tiltxcorr](https://bio3d.colorado.edu/imod/doc/man/tiltxcorr.html) 
in PyTorch.


## Installation

```bash
pip install torch-tiltxcorr
```

## Usage

```python
import torch
from torch_fourier_shift import fourier_shift_image_2d
from torch_tiltxcorr import tiltxcorr

# Load or create your tilt series
# tilt_series shape: (batch, height, width) - batch is number of tilt images
# Example: tilt_series with shape (61, 512, 512) - 61 tilt images of 512x512 pixels
tilt_series = torch.randn(61, 512, 512)

# Define tilt angles (in degrees)
# Shape: (batch,) - one angle per tilt image
tilt_angles = torch.linspace(-60, 60, steps=61)

# Define tilt axis angle (in degrees)
tilt_axis_angle = 45

# Run tiltxcorr
# (for noisy cryo-EM data, apply a lowpass to ~4x pixel size as a starting point)
shifts = tiltxcorr(
   tilt_series=tilt_series,
   tilt_angles=tilt_angles,
   tilt_axis_angle=tilt_axis_angle,
   pixel_spacing_angstroms=10,
   lowpass_angstroms=40,
)
# shifts shape: (batch, 2) - (dy, dx) shifts which center each tilt image

# Apply shifts to align the tilt series
aligned_tilt_series = fourier_shift_image_2d(tilt_series, shifts=shifts)
# aligned_tilt_series shape: (batch, height, width)
```

Use [uv](https://docs.astral.sh/uv/) to run an example with simulated data and visualize the results.

```shell
uv run examples/tiltxcorr_example_simulated_data.py
```

## How It Works

torch-tiltxcorr performs coarse tilt series alignment by:

1. Sorting images by tilt angle
2. Dividing the series into groups of positive and negative tilt angles
3. For each adjacent pair of images in each group:
   - Applying a stretch perpendicular to the tilt axis on the image with the larger tilt angle
   - Calculating cross-correlation between the images
   - Extracting the shift from the position of the correlation peak
   - Transforming the shift to account for the stretch applied to the image
4. Accumulating shifts to align the entire series

### With Sample Tilt Estimation

If a sample is physically rotated +5° around the microscope stage tilt axis, then at nominal 0° stage tilt the beam sees the sample at +5°.

This offset affects the correlations measured by tiltxcorr. 
By finding the offset value that maximizes total correlation, we can estimate the true physical pre-tilt of the sample around the microscope tilt axis.

```python
import torch
from torch_fourier_shift import fourier_shift_image_2d
from torch_tiltxcorr import tiltxcorr_with_sample_tilt_estimation

# Load or create your tilt series
tilt_series = torch.randn(61, 512, 512)
tilt_angles = torch.linspace(-60, 60, steps=61)
tilt_axis_angle = 45

# Run tiltxcorr with sample tilt estimation
# (for noisy cryo-EM data, apply a lowpass to ~4x pixel size as a starting point)
shifts, sample_tilt = tiltxcorr_with_sample_tilt_estimation(
   tilt_series=tilt_series,
   tilt_angles=tilt_angles,
   tilt_axis_angle=tilt_axis_angle,
   pixel_spacing_angstroms=10,
   lowpass_angstroms=40,
   sample_tilt_range=(-30.0, 30.0),  # search range in degrees
)
# shifts shape: (batch, 2) tensor of (dy, dx) shifts which center each tilt image
# sample_tilt: float value for component of sample tilt about the stage in degrees

# Apply shifts to align the tilt series
aligned_tilt_series = fourier_shift_image_2d(tilt_series, shifts=shifts)
```

## License

This package is distributed under the BSD 3-Clause License.
