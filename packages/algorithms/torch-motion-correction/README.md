# torch-motion-correction

[![License](https://img.shields.io/pypi/l/torch-motion-correction.svg?color=green)](https://github.com/teamtomo/teamtomo/blob/main/packages/algorithms/torch-motion-correction/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-motion-correction.svg?color=green)](https://pypi.org/project/torch-motion-correction)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-motion-correction.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml)

Movie motion estimation and correction using PyTorch. Implements a spline-based deformation field to model the beam-induced motion over movie collection.

## Implemented algorithms

- Cross-correlation-based global motion estimation (see `estimate_global_motion` in [estimate_motion_xc.py](src/torch_motion_correction/estimate_motion_xc.py))
- Spline-based local motion estimation (see `estimate_local_motion` in [estimate_motion_optimizer.py](src/torch_motion_correction/estimate_motion_optimizer.py))
- Motion correction of movie (not summation to final micrograph) using estimated deformation field (see `correct_motion` in [correct_motion.py](src/torch_motion_correction/correct_motion.py))

## Installation

```bash
pip install torch-motion-correction
```

## Usage

Motion is estimated as a `DeformationField` (shifts in Å over time and space),
first for whole frames by cross-correlation, then refined to local motion with a
spline model, and finally applied to every frame. The example below runs on a
synthetic movie drifting half a pixel per frame:

```python
import torch
from torch_fourier_shift import fourier_shift_image_2d
from torch_motion_correction import (
    PatchSamplingConfig,
    correct_motion,
    estimate_global_motion,
    estimate_local_motion,
)

# A synthetic 10-frame movie (t, h, w) drifting 0.5 px per frame in y and x
frame = torch.nn.functional.avg_pool2d(torch.rand(1, 1, 512, 512), 5, 1, 2)[0, 0]
drift = torch.linspace(0, 4.5, 10)[:, None].expand(10, 2)  # (t, 2) yx shifts, px
movie = fourier_shift_image_2d(frame, shifts=drift)
movie = movie + 0.1 * torch.randn_like(movie)
pixel_spacing = 1.0  # Å

# 1. Whole-frame (global) motion by cross-correlation
global_field = estimate_global_motion(movie, pixel_spacing=pixel_spacing)

# 2. Refine to local motion, starting from the global estimate
local_field, tracker = estimate_local_motion(
    movie,
    pixel_spacing=pixel_spacing,
    deformation_field_resolution=(10, 4, 4),  # (t, gh, gw) spline control points
    patch_sampling=PatchSamplingConfig(patch_shape=(128, 128)),
    initial_deformation_field=global_field,
)

# 3. Correct every frame, then sum to a micrograph
corrected = correct_motion(movie, local_field, pixel_spacing=pixel_spacing)
micrograph = corrected.mean(dim=0)
```

Pass `device=torch.device("cuda")` to the estimation and correction functions to
run on a GPU; the movie itself can stay on the CPU.

## Authors

- Alister Burt (started when working at MRC-LMB, continued at Genentech)
- Josh Dickerson (UC Berkeley)
- Matthew Giammar (UC Berkeley)

