# torch-ctf-estimation

Contrast transfer function estimation for cryo-EM images in PyTorch.

## Overview

`torch-ctf-estimation` fits defocus, astigmatism, and (optionally) sample
thickness from a micrograph power spectrum.

- 1D defocus on the mean spectrum, then 2D defocus / astigmatism on patches
- Spatial defocus as a spline grid or a linear tilt model
- 1D thickness grid search against a thickness-modulated CTF, with optional
  joint defocus+thickness refinement (1D scalar or 2D field)
- Laser phase plate (LPP) CTF support via `torch-ctf`

This package is the algorithm: `estimate_ctf` (defocus only) and
`estimate_ctf_and_thickness` (defocus + thickness) are self-contained
compositions of the lower-level primitives, usable directly from notebooks,
tests, or another program. Downstream tools can chain these primitives
differently and add optics / metrics file I/O.

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

To also estimate sample thickness, use `estimate_ctf_and_thickness` with a
`ThicknessParams`:

```python
from torch_ctf_estimation import estimate_ctf_and_thickness
from torch_ctf_estimation.models import ThicknessParams

thickness = ThicknessParams(refine_dim="2d")
result = estimate_ctf_and_thickness(image, optical, fitting, thickness)
result.result2d, result.thickness1d, result.thickness_joint
```

## Changed in 0.7.0

- Requires `torch-ctf>=0.7.0` and `torch-grid-utils>=0.7.0`.
- `extract_ctf_patches` returns the patch shape as an `(h, w)` tuple instead of a
  sidelength.
- The 2D estimators, `apply_laser_axis_mask`, `warp_rfft_power_isotropic` and the
  tilt-corrected power spectra take an optional `image_shape`. Without it the
  patches are assumed square, and a spectrum whose width is not `h // 2 + 1`
  raises `ValueError`. Pass `image_shape` for non-square patches.
- Odd patch sizes no longer give false astigmatism (2D) or a defocus bias (1D),
  and non-square whole images are fitted with their true shape.
- The returned pixel spacing is the one the rescale achieved, which can differ
  slightly from the target (e.g. 3.00073 Å instead of 3 Å).
- The single-laser axis mask measures `laser_xy_angle_deg` from +ky, like the LPP
  model, so it masks the strip the laser actually covers. Dual-laser masks (the
  default) are unchanged.
- The equiphase 1D profile is referenced to the mean defocus, so astigmatism no
  longer shifts the fitted 1D defocus.
