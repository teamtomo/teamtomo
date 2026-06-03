# torch-fit-in-map

[![License](https://img.shields.io/pypi/l/torch-fit-in-map.svg?color=green)](https://github.com/teamtomo/teamtomo/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-fit-in-map.svg?color=green)](https://pypi.org/project/torch-fit-in-map)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-fit-in-map.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml)

## Overview

`torch-fit-in-map` is a PyTorch package for rigid-body volume alignment in cryo-EM. It finds the rotation and translation that best superimposes a *mobile* volume onto a *reference* volume using normalised cross-correlation (NCC).

Two input modes are supported:

- **Map-to-map** — align one MRC density map onto another.
- **PDB-to-map** — simulate an electrostatic potential density from an atomic model and align it to an experimental map.

## Features

- Exhaustive SO(3) grid search with per-rotation FFT-based optimal translation
- Symmetry-aware search (`C1`, `C4`, `D2`, `T`, `O`, `I`, …) to restrict the search to the asymmetric unit
- Gradient-based local refinement (L-BFGS or Adam) using PyTorch autograd
- Multi-start refinement to escape local minima
- Optional soft masking
- Multi-GPU support for both exhaustive search and gradient refinement
- Atomic-model output: transform the input PDB/mmCIF coordinates into the reference frame
- Command-line tools: `torch-fit-in-map` and `torch-simulate-density`

## Installation

```bash
pip install torch-fit-in-map
```

For PDB-to-map alignment, the electrostatic potential simulator must also be installed:

```bash
pip install git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git
```

## Basic Usage

### Map-to-map alignment

```python
import torch
from torch_fit_in_map import align_volumes, apply_alignment

# (d, h, w) float tensors at the same pixel size
reference = torch.load("reference.pt")
mobile    = torch.load("mobile.pt")

result = align_volumes(reference, mobile, pixel_size_angstroms=1.5)

print(f"NCC score:         {result.score:.4f}")
print(f"Rotation (zyx):\n{result.rotation_matrix}")
print(f"Translation (px):  {result.translation_pixels}")
print(f"Translation (Å):   {result.translation_angstroms}")

aligned = apply_alignment(mobile, result)
```

### From MRC files

```python
from torch_fit_in_map import align_volumes_from_files, apply_alignment

result = align_volumes_from_files("reference.mrc", "mobile.mrc")
```

Voxel sizes are read from the MRC headers; if they differ the mobile map is
automatically Fourier-rescaled to match the reference.

### PDB-to-map alignment

```python
from torch_fit_in_map import align_map_to_pdb_from_files

result = align_map_to_pdb_from_files(
    map_path="experimental.mrc",
    pdb_path="model.pdb",
    desired_resolution_angstroms=8.0,  # low-pass filter simulated density
)
```

### Tuning the search

```python
from torch_fit_in_map import (
    align_volumes,
    ExhaustiveSearchConfig,
    GradientRefinementConfig,
)

exhaustive_cfg = ExhaustiveSearchConfig(
    angular_step_degrees=7.5,   # finer orientation sampling
    symmetry="C4",              # restrict search to C4 asymmetric unit
    n_start=5,                  # refine top-5 poses independently
    rotation_batch_size=32,     # increase for faster throughput on large VRAM
)

gradient_cfg = GradientRefinementConfig(
    optimizer="lbfgs",
    n_iterations=200,
    loss="ncc",
)

result = align_volumes(
    reference,
    mobile,
    exhaustive_config=exhaustive_cfg,
    gradient_config=gradient_cfg,
    pixel_size_angstroms=1.5,
)
```

Pass `gradient_config=None` to return the exhaustive-search result without
further refinement.

### Multi-GPU

```python
exhaustive_cfg = ExhaustiveSearchConfig(
    angular_step_degrees=7.5,
    devices=["cuda:0", "cuda:1"],
)
gradient_cfg = GradientRefinementConfig(
    n_start=4,
    devices=["cuda:0", "cuda:1"],
)
result = align_volumes(reference, mobile,
                       exhaustive_config=exhaustive_cfg,
                       gradient_config=gradient_cfg)
```

## AlignmentResult

All alignment functions return an `AlignmentResult`:

| Field | Type | Description |
|---|---|---|
| `rotation_matrix` | `(3, 3)` tensor | Rotation in zyx convention |
| `translation_pixels` | `(3,)` tensor | Translation in zyx pixels |
| `score` | `float` | Peak NCC score (higher is better, max 1.0) |
| `translation_angstroms` | `(3,)` tensor or `None` | Translation in Å (when pixel size is provided) |

Use `apply_alignment(mobile, result)` to produce the aligned volume.

## Command-line tools

### Align two volumes or a PDB to a map

```bash
# Map-to-map
torch-fit-in-map reference.mrc mobile.mrc --output aligned.mrc

# PDB-to-map (auto-detected from extension)
torch-fit-in-map experimental.mrc model.pdb \
    --desired-resolution 8.0 \
    --output fitted.pdb \
    --save-simulated simulated.mrc
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--angular-step` | `15.0` | Angular search step in degrees |
| `--symmetry` | `C1` | Point-group symmetry of the reference |
| `--n-start` | `1` | Top poses to refine independently |
| `--n-iter` | `100` | Gradient refinement iterations (0 to skip) |
| `--optimizer` | `lbfgs` | `lbfgs` or `adam` |
| `--mask` | — | Optional soft-mask MRC |
| `--output-json` | — | Write result (rotation, translation, score) as JSON |
| `--device` | `auto` | `cpu`, `cuda`, `all`, or `0,1` for multi-GPU |
| `--quiet` | — | Suppress progress bars (requires `--output` or `--output-json`) |

### Simulate a density from an atomic model

```bash
torch-simulate-density model.pdb \
    --output simulated.mrc \
    --pixel-size 1.5 \
    --box-size 128 \
    --desired-resolution 6.0
```

## License

BSD 3-Clause — see the [LICENSE](https://github.com/teamtomo/teamtomo/blob/main/LICENSE) file for details.
