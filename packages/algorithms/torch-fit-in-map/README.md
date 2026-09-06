# torch-fit-in-map

[![License](https://img.shields.io/pypi/l/torch-fit-in-map.svg?color=green)](https://github.com/teamtomo/teamtomo/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-fit-in-map.svg?color=green)](https://pypi.org/project/torch-fit-in-map)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-fit-in-map.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml)

## Overview

`torch-fit-in-map` is a PyTorch package for rigid-body volume alignment in
cryo-EM. It finds the rotation and translation that best superimposes a *mobile*
volume onto a *reference* volume using normalised cross-correlation (NCC).

The public API operates purely on **`torch.Tensor`** potential maps and
**`pandas.DataFrame`** atom tables — it does no file I/O. Reading/writing MRC and
PDB/mmCIF files and the command-line tools live in the companion package
[`torch-fit-in-map-cli`](https://github.com/rsanchezgarc/torch-fit-in-map-cli).

Two alignment modes are supported:

- **Map-to-map** — align one map onto another (`fit_map_in_map`).
- **Atoms ↔ map** — simulate an electrostatic potential from a table of
  atoms and align it against a map (`fit_structure_in_map` /
  `fit_map_in_structure`).

## Features

- Exhaustive SO(3) grid search with per-rotation FFT-based optimal translation
- Symmetry-aware search (`C1`, `C4`, `D2`, `T`, `O`, `I`, …) to restrict to the asymmetric unit
- Gradient-based local refinement (L-BFGS or Adam) using PyTorch autograd
- Multi-start refinement to escape local minima
- Optional soft masking and multi-GPU support
- Atom-table transform: map input atomic coordinates into the reference frame
  (`apply_alignment_to_structure`)

## Installation

```bash
pip install torch-fit-in-map
```

The atoms↔map modes use the electrostatic-potential simulator
`torch-calculate-electrostatic-potential` (installed automatically as a
dependency).

## Basic Usage

### Map-to-map alignment

```python
import torch
from torch_fit_in_map import fit_map_in_map, apply_alignment

# (d, h, w) float tensors at the same pixel size
reference = torch.load("reference.pt")
mobile    = torch.load("mobile.pt")

result = fit_map_in_map(mobile, reference, pixel_size_angstroms=1.5)

print(f"NCC score:        {result.score:.4f}")
print(f"Rotation (zyx):\n{result.rotation_matrix}")
print(f"Translation (px): {result.translation_pixels}")

aligned = apply_alignment(mobile, result)
```

If the two maps have different voxel sizes, resample first with
`normalise_voxel_sizes(reference, mobile, ref_px, mob_px)`.

### Atoms ↔ map alignment

Atoms are passed as a DataFrame with columns `x`, `y`, `z` (Å) and `element` —
exactly what [`mmdf`](https://github.com/teamtomo/mmdf) produces:

```python
import mmdf
from torch_fit_in_map import fit_structure_in_map

atoms = mmdf.read("model.pdb")          # pandas DataFrame

result = fit_structure_in_map(
    mobile_atoms=atoms,
    reference_map=experimental,          # (d, h, w) tensor
    pixel_size_angstroms=1.5,
    box_size=128,
)
```

`fit_map_in_structure` does the inverse (fit a map into the frame of an atomic
structure).
Both accept a custom `simulator=` implementing the `PotentialSimulator` protocol
and optional `simulator_config=` for the default electrostatic-potential backend.

#### Simulation contract

The default simulator (`DEFAULT_POTENTIAL_SIMULATOR`) delegates to
`torch-calculate-electrostatic-potential`:

1. Atoms are centred at the cubic simulation-box centre
   (`default_sublattice_radius(pixel_size)` sets the per-atom stencil).
2. A `(box_size, box_size, box_size)` potential in volts is returned (ZYX order).
3. `apply_alignment_to_structure` inverts the same centre/crop geometry before
   applying the alignment transform.

Use `PotentialSimulatorConfig` to select Peng elemental vs bonded scattering
factors. Bonded factors require structure columns
`chain`, `residue_id`, `residue`, and `atom` (or set `annotate_bonding=True`).

The CLI wrappers `torch-fit-in-map`, `torch-fit-in-atomic-model`, and
`torch-simulate-density` in
[`torch-fit-in-map-cli`](https://github.com/rsanchezgarc/torch-fit-in-map-cli)
call the same default simulator path.

### Transforming atoms into the reference frame

```python
from torch_fit_in_map import apply_alignment_to_structure

moved = apply_alignment_to_structure(
    atoms, result,
    pixel_size=1.5,
    box_shape=reference.shape,           # (d, h, w)
)   # returns a DataFrame with transformed x/y/z
```

### Tuning the search

```python
from torch_fit_in_map import (
    fit_map_in_map,
    ExhaustiveSearchConfig,
    GradientRefinementConfig,
)

result = fit_map_in_map(
    mobile,
    reference,
    exhaustive_config=ExhaustiveSearchConfig(
        angular_step_degrees=7.5,   # finer orientation sampling
        symmetry="C4",              # restrict to C4 asymmetric unit
        n_start=5,                  # refine top-5 poses independently
        devices=["cuda:0", "cuda:1"],
    ),
    gradient_config=GradientRefinementConfig(optimizer="lbfgs", n_iterations=200),
    pixel_size_angstroms=1.5,
)
```

Pass `gradient_config=None` to return the exhaustive-search result without refinement.

## AlignmentResult

All alignment functions return an `AlignmentResult`:

| Field | Type | Description |
|---|---|---|
| `rotation_matrix` | `(3, 3)` tensor | Rotation in zyx convention |
| `translation_pixels` | `(3,)` tensor | Translation in zyx pixels |
| `score` | `float` | Peak NCC score (higher is better, max 1.0) |
| `translation_angstroms` | `(3,)` tensor or `None` | Translation in Å (when pixel size is provided) |
| `simulated_potential` | `(d, h, w)` tensor or `None` | Simulated potential when `save_simulated=True` |

Use `apply_alignment(mobile, result)` to produce the aligned volume.

## Command-line tools

The `torch-fit-in-map`, `torch-fit-in-atomic-model` and `torch-simulate-density`
commands (with MRC/PDB file handling) live in
[`torch-fit-in-map-cli`](https://github.com/rsanchezgarc/torch-fit-in-map-cli).
They use the same `DEFAULT_POTENTIAL_SIMULATOR` path as the Python API.

## License

BSD 3-Clause — see the [LICENSE](https://github.com/teamtomo/teamtomo/blob/main/LICENSE) file for details.
