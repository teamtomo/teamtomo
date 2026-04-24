# torch-local-resolution

Core functionality for local resolution estimation of cryo-EM half-maps.

## Overview

`torch-local-resolution` estimates local resolution by computing the local cosine similarity between two bandpass-filtered half-maps. For each resolution shell, the correlation is measured within a sliding spherical window across the map. Statistical significance is assessed by comparing the observed correlations against a null distribution derived from phase-permuted or voxel-shuffled surrogates.

- Supports 2D (single-slice) and 3D half-maps
- Batched processing of multiple half-map pairs
- Configurable resolution shells, window sizes, and step sizes
- Statistical p-value maps or raw correlation output

## Installation

This package is part of the [TeamTomo monorepo](https://github.com/teamtomo/teamtomo). See the main repository README for development setup instructions.

## Usage

```python
import torch
from torch_local_resolution import estimate_local_resolution

# Load your half-maps as (B, Z, Y, X) tensors
# For 2D maps, set Z=1: (B, 1, Y, X)
half_map1 = torch.randn(1, 64, 64, 64)
half_map2 = torch.randn(1, 64, 64, 64)

# Define resolution shells and matching window radii
apix=1.0
resolutions = [10.0, 8.0, 6.0, 4.0]  # in Ångström
windows_radii = [10.5, 9.1, 7.2, 4.5]  # in voxels, one per shell

# Compute local resolution p-value map
pvalue_map = estimate_local_resolution(
    apix=apix,                    # voxel size in Å/pixel
    windows_radii=windows_radii,
    resolutions=resolutions,
    batch_half_map1=half_map1,
    batch_half_map2=half_map2,
    step_size=3,                 # stride between sampling points
    gpu_id=0,                   # use first CUDA device (None for CPU)
)

# To get raw correlation values instead of p-values
correlation_map = estimate_local_resolution(
    apix=1.0,
    windows_radii=windows_radii,
    resolutions=resolutions,
    batch_half_map1=half_map1,
    batch_half_map2=half_map2,
    step_size=3,
    gpu_id=0,
    skip_statistics=True,
)
```
