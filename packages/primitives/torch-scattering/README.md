# torch-scattering

[![License](https://img.shields.io/pypi/l/torch-scattering.svg?color=green)](https://github.com/joelyeois/torch-scattering/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-scattering.svg?color=green)](https://pypi.org/project/torch-scattering)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-scattering.svg?color=green)](https://python.org)
[![CI](https://github.com/joelyeois/torch-scattering/actions/workflows/ci.yml/badge.svg)](https://github.com/joelyeois/torch-scattering/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/joelyeois/torch-scattering/branch/main/graph/badge.svg)](https://codecov.io/gh/joelyeois/torch-scattering)

Multislice electron scattering simulation in PyTorch, for cryo-EM/cryo-ET forward modelling.

## Overview

`torch_scattering` computes the 2D exit wave produced by propagating an electron
beam through a 3D electrostatic potential in volts. The potential has shape
`(..., Z, H, W)`, where Z is the beam direction. `pixel_size` is the isotropic
voxel spacing in Angstroms, so it specifies both the Y/X pixel spacing and the
Z slice thickness. Every function returns a complex exit wave of shape
`(..., H, W)`.

Real `float32` and `float64` potentials model non-absorbing specimens and can be
passed directly; callers do not need to cast them to complex. Complex potentials
remain supported for modelling absorption.

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

All four share the same required inputs and can be swapped in for one another.
`multislice`, `rytov`, and `firstborn` also accept an `n_slices` argument to
coarsen the potential into fewer, thicker slabs before propagating.

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

# A real electrostatic potential in volts, shape (Z, H, W).
potential = torch.zeros((50, 64, 64), dtype=torch.float32)

# propagate a plane wave through it
exit_wave = multislice(
    potential=potential,
    pixel_size=1.0,   # Angstroms
    voltage=300,      # kV
)
# exit_wave.shape is (64, 64)
# exit_wave.dtype is torch.complex64
```

`rytov`, `firstborn`, and `projection` share the same call signature:

```python
from torch_scattering import firstborn, projection, rytov

exit_wave = rytov(potential, pixel_size=1.0, voltage=300)
exit_wave = firstborn(potential, pixel_size=1.0, voltage=300)
exit_wave = projection(potential, pixel_size=1.0, voltage=300)  # n_slices not applicable
```

### Coarsening slices

`n_slices` groups the potential into fewer, thicker slabs before propagating.
By default (`n_slices=None`), every slice of the potential is propagated
individually - the most accurate but slowest setting.

```python
# propagate as 10 chunks instead of all 50 slices individually
exit_wave = multislice(potential, pixel_size=1.0, voltage=300, n_slices=10)
```

### Batching

All functions accept arbitrary leading batch dimensions on `potential`:

```python
potential = torch.zeros((8, 50, 64, 64), dtype=torch.complex64)  # batch of 8
exit_wave = multislice(potential, pixel_size=1.0, voltage=300)
# exit_wave.shape is (8, 64, 64)
```

## Structure-to-wave pipeline

Structure handling and potential generation are deliberately separate packages.
They are not runtime dependencies of `torch-scattering`; their real tensor
output is passed through the public tensor API:

```python
import pandas as pd
from torch_calculate_electrostatic_potential import (
    GridConfig,
    potential_from_structure_3d,
)
from torch_scattering import multislice
from torch_structure_manipulation import (
    AtomicStructure,
    annotate_bonding_environments,
)

# mmdf-compatible coordinates are in Angstroms.
atoms = pd.DataFrame(
    [
        ("A", 1, "ALA", "C", "C", 0.0, 0.0, 0.0),
        ("A", 1, "ALA", "O", "O", 1.2, 0.0, 0.0),
        ("A", 1, "ALA", "CA", "C", -1.2, 0.0, 0.0),
        ("A", 2, "GLY", "N", "N", 2.4, 0.0, 0.0),
    ],
    columns=[
        "chain", "residue_id", "residue", "atom", "element", "x", "y", "z"
    ],
)
atoms["b_isotropic"] = 10.0  # Angstrom squared
atoms["occupancy"] = 1.0

# Annotate a complete local residue context, then build the desired structure.
annotated = annotate_bonding_environments(atoms, include_hydrogens=False)
structure = AtomicStructure.from_dataframe(annotated.iloc[[0]])

grid = GridConfig.from_grid_shape_and_voxel_size(
    grid_shape=(9, 9, 9),       # Z, Y, X
    voxel_size=(1.0, 1.0, 1.0), # Angstroms; isotropic for scattering
    center_zyx=(0.0, 0.0, 0.0),
    sublattice_radius=4.0,
)
elemental_volts = potential_from_structure_3d(structure, grid)
bonded_volts = potential_from_structure_3d(
    structure,
    grid,
    scattering_factors="peng_bonded",
    bonded_fallback="error",
)

# Both volumes are real tensors in volts and are accepted directly.
elemental_wave = multislice(elemental_volts, pixel_size=1.0, voltage=300.0)
bonded_wave = multislice(bonded_volts, pixel_size=1.0, voltage=300.0)
# Both waves are complex tensors; voltage is in kV.
```

`projection()` is a wave-propagation approximation that numerically sums this
sampled 3D volume along Z. It is distinct from the electrostatic package's
analytic 2D projected-potential calculation and from projection alignment in
`torch-fit-in-map`.

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
sigma = interaction_parameter(voltage=300)

wave = torch.ones((64, 64), dtype=torch.complex64)
potential_slice = torch.zeros((64, 64), dtype=torch.complex64)
wave = multislice_step(wave, potential_slice, propagator, sigma, dz=1.0)
```

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
