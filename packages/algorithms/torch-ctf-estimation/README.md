# torch-ctf-estimation

Contrast transfer function estimation for cryo-EM images in PyTorch.

## Overview

`torch-ctf-estimation` fits defocus, astigmatism, and (optionally) sample
thickness from a micrograph power spectrum.

- 1D defocus on the mean spectrum, then 2D defocus / astigmatism on patches
- Spatial defocus as a spline grid or a linear tilt model
- 1D thickness grid search against a thickness-modulated CTF, with optional
  thickness-only refinement
- Laser phase plate (LPP) CTF support via `torch-ctf`

This package is the algorithm. Programs such as PICASSO chain these primitives
and write optics / metrics files.

## Installation

This package is part of the [TeamTomo monorepo](https://github.com/teamtomo/teamtomo).
See the main repository README for development setup instructions.

## Usage

```python
import torch
from torch_ctf_estimation import estimate_ctf
from torch_ctf_estimation.models import CTFFittingParams, OpticalParams

image = torch.randn(1024, 1024)
optical = OpticalParams(
    pixel_spacing_angstroms=1.0,
    voltage_kev=300.0,
    spherical_aberration_mm=2.7,
    amplitude_contrast_fraction=0.07,
)
fitting = CTFFittingParams(
    defocus_grid_resolution=(1, 3, 3),
    frequency_fit_range_angstroms=(30.0, 5.0),
    defocus_range_microns=(0.5, 5.0),
    patch_sidelength=128,
)
mean_ps, result1d, result2d = estimate_ctf(image, optical, fitting)
```
