# torch-refine-tilt-axis-angle

[![License](https://img.shields.io/pypi/l/torch-refine-tilt-axis-angle.svg?color=green)](https://github.com/teamtomo/torch-refine-tilt-axis-angle/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-refine-tilt-axis-angle.svg?color=green)](https://pypi.org/project/torch-refine-tilt-axis-angle)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-refine-tilt-axis-angle.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-refine-tilt-axis-angle/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-refine-tilt-axis-angle/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-refine-tilt-axis-angle/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-refine-tilt-axis-angle)

Tilt-axis angle optimization for tilt series using common lines.

## Overview

torch-refine-tilt-axis-angle provides an implementation for finding the in plane rotation for a tilt-series that aligns the tilt axis with the y-axis. This is done by extracting lines from the 2D Fourier transform of the tilt-series images and minimizing the mean squared error between them, i.e. the common lines. It makes use of the pytorch LBFGS optimizer to find the minimum.

## Installation

```bash
pip install torch-refine-tilt-axis-angle
```

## Usage

```python
import torch
from torch_refine_tilt_axis_angle import refine_tilt_axis_angle

# Load or create your tilt series
# tilt_series shape: (batch, height, width) - batch is number of tilt images
# Example: tilt_series with shape (61, 512, 512) - 61 tilt images of 512x512 pixels
tilt_series = torch.randn(61, 512, 512)

# Specify an initial guess for the tilt axis angle (the default is 0.0)
# This can be the value from an MDOC file.
initial_tilt_axis_angle = 50.0

# Run tilt axis angle refinement.
new_tilt_axis_angle = refine_tilt_axis_angle(
    tilt_series=tilt_series,
    tilt_axis_angle=initial_tilt_axis_angle,
    # grid_points=1,  # optionally increase the spline grid points (default 1)
    # return_single_angle=False,  # optionally write out an angle for each image 
)
```

Use [uv](https://docs.astral.sh/uv/) to run an example with simulated data and visualize the results.

```shell
uv run examples/simulate_tilt_axis_refinement.py
```

## License

This package is distributed under the BSD 3-Clause License.
