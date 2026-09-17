# torch-refine-tilt-axis-angle

[![License](https://img.shields.io/pypi/l/torch-refine-tilt-axis-angle.svg?color=green)](https://github.com/teamtomo/torch-refine-tilt-axis-angle/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-refine-tilt-axis-angle.svg?color=green)](https://pypi.org/project/torch-refine-tilt-axis-angle)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-refine-tilt-axis-angle.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-refine-tilt-axis-angle/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-refine-tilt-axis-angle/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-refine-tilt-axis-angle/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-refine-tilt-axis-angle)

Tilt-axis angle optimization for tilt series using common lines.

## Overview

torch-refine-tilt-axis-angle finds the tilt-axis angle of a translationally
aligned tilt series from its common line: the line through the origin of
Fourier space that all tilt images share, oriented perpendicular to the tilt
axis. Each image's Fourier transform is coherently summed across the stack,
and a two-stage angular grid search (coarse, then a finer search around the
coarse optimum) locates the orientation of highest power in the summed power
spectrum. The whole search is evaluated as batched tensor operations, so
every candidate angle is scored in parallel rather than in a Python loop.

Rectangular images are handled natively, without cropping to square, so no
image data is discarded.

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

# Specify an initial guess for the tilt axis angle (the default is 90.0,
# which searches the full [0, 180] range). This can be the value from an
# MDOC file.
initial_tilt_axis_angle = 50.0

# Run tilt axis angle refinement.
new_tilt_axis_angle = refine_tilt_axis_angle(
    tilt_series=tilt_series,
    tilt_axis_angle=initial_tilt_axis_angle,
)
```

## License

This package is distributed under the BSD 3-Clause License.
