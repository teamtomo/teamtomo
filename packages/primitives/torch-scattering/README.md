# torch-scattering

[![License](https://img.shields.io/pypi/l/torch-scattering.svg?color=green)](https://github.com/joelyeois/torch-scattering/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-scattering.svg?color=green)](https://pypi.org/project/torch-scattering)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-scattering.svg?color=green)](https://python.org)
[![CI](https://github.com/joelyeois/torch-scattering/actions/workflows/ci.yml/badge.svg)](https://github.com/joelyeois/torch-scattering/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/joelyeois/torch-scattering/branch/main/graph/badge.svg)](https://codecov.io/gh/joelyeois/torch-scattering)

Multislice electron scattering simulation in PyTorch, for cryo-EM/cryo-ET forward modelling.

## Overview

`torch_scattering` computes the 2D exit wave produced by propagating an electron
beam through a 3D scattering potential. Given a potential of shape `(..., Z, H, W)`,
a pixel size, and a beam energy, each function returns the complex-valued exit
wave of shape `(..., H, W)`.

Four propagation modes are provided, trading physical accuracy for speed:

* `multislice()` - full multislice propagation (Kirkland, *Advanced Computing in
  Electron Microscopy*), alternating transmission through each slice with Fresnel
  propagation to the next. The most accurate mode.
* `rytov()` - Rytov approximation, accumulating phase in the exponent rather than
  the wave itself.
* `firstborn()` - first Born approximation, summing single-scattering
  contributions from each slice.
* `projection()` - projection approximation, treating the specimen as infinitely
  thin and skipping inter-slice propagation entirely. The fastest and least
  accurate mode.

All four share the same signature and can be swapped in for one another. Each
also accepts an `n_slices` argument to coarsen the potential into fewer, thicker
slabs before propagating, trading accuracy for speed.

Lower-level, pure-math primitives (`fresnel_propagator`, `transmission_function`,
`multislice_step`, `chunk_slices`, `interaction_parameter`) are also exposed for
building custom propagation schemes.

## Installation

```shell
pip install torch-scattering
```

## Usage

```python
import torch
from torch_scattering import multislice

# a complex-valued scattering potential, shape (Z, H, W)
potential = torch.zeros((50, 64, 64), dtype=torch.complex64)

# propagate a plane wave through it
exit_wave = multislice(
    potential=potential,
    pixel_size=1.0,   # Angstroms
    energy=300,       # keV
)
# exit_wave.shape is (64, 64)
```

`rytov`, `firstborn`, and `projection` share the same call signature:

```python
from torch_scattering import firstborn, projection, rytov

exit_wave = rytov(potential, pixel_size=1.0, energy=300)
exit_wave = firstborn(potential, pixel_size=1.0, energy=300)
exit_wave = projection(potential, pixel_size=1.0, energy=300)  # n_slices not applicable
```

### Coarsening slices

`n_slices` groups the potential into fewer, thicker slabs before propagating.
By default (`n_slices=None`), every slice of the potential is propagated
individually - the most accurate but slowest setting.

```python
# propagate as 10 chunks instead of all 50 slices individually
exit_wave = multislice(potential, pixel_size=1.0, energy=300, n_slices=10)
```

### Batching

All functions accept arbitrary leading batch dimensions on `potential`:

```python
potential = torch.zeros((8, 50, 64, 64), dtype=torch.complex64)  # batch of 8
exit_wave = multislice(potential, pixel_size=1.0, energy=300)
# exit_wave.shape is (8, 64, 64)
```

## Low-level primitives

For building custom propagation schemes directly on top of the multislice
recurrence:

```python
import torch
from torch_grid_utils import fftfreq_grid
from torch_scattering import (
    fresnel_propagator,
    interaction_parameter,
    multislice_step,
)

frequency_grid = fftfreq_grid(image_shape=(64, 64), rfft=False, spacing=1.0, norm=True)
propagator = fresnel_propagator(frequency_grid, wavelength=0.01969, dz=1.0)
sigma = interaction_parameter(energy=300)

wave = torch.ones((64, 64), dtype=torch.complex64)
potential_slice = torch.zeros((64, 64), dtype=torch.complex64)
wave = multislice_step(wave, potential_slice, propagator, sigma, dz=1.0)
```

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
