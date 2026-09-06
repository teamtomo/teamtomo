# torch-calculate-electrostatic-potential

Differentiable 2D projected and 3D electrostatic potentials from Peng 1996
electron-scattering factors.

The high-level API consumes
`torch_structure_manipulation.AtomicStructure`. The tensor-only
`calculate_scattering_potential_2d` and `calculate_scattering_potential_3d`
kernels remain public and support arbitrary leading batch dimensions.

Coordinates and spacing are in Angstroms. Axis order is ZYX in 3D and YX in 2D.

## Units and normalization

`peng1996_element_params.npy` contains Peng et al. (1996) **elastic electron
scattering factors**, not X-ray form factors:

```text
f_e(s) = sum_i a_i exp(-b_i s²),  s = sin(theta) / wavelength
```

The amplitudes `a_i` and `f_e` are in Angstroms and `b_i` is in Angstroms
squared. The X-ray-to-electron Mott-Bethe conversion
`f_e(s) = 0.023934 (Z - f_X(s)) / s²` is therefore already incorporated in the
tabulated coefficients and must not be applied again.

An electron scattering factor is not itself a real-space potential in volts.
The package converts it to the Fourier transform of the electrostatic potential
using

```text
V_tilde(g) = C f_e(g / 2),  g = 2s
C = 2 pi hbar² / (m_e e) = 47.877647... V Angstrom²
```

The inverse transform returned by `calculate_scattering_potential_3d` and
`potential_from_structure_3d` is therefore in **volts**. The 2D functions
analytically integrate the 3D potential over the omitted spatial axis and
return a projected potential in **volt-Angstroms**.

The bonded coefficients come from
[Shtyrov et al. (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13167779/)
and use the equivalent convention `f_e(g) = sum_i a_i exp(-b_i g² / 4)`.
Protein and RNA currently share the same coefficient table because RNA-specific
factors have not yet been measured.

## Installation

```sh
# From PyPI (after first release)
pip install torch-calculate-electrostatic-potential
```

```sh
# Development install from the monorepo
pip install -e packages/primitives/torch-calculate-electrostatic-potential
```

With [uv](https://github.com/astral-sh/uv): `uv pip install torch-calculate-electrostatic-potential`.

## Usage

```python
from torch_calculate_electrostatic_potential import (
    GridConfig,
    potential_from_structure_2d,
    potential_from_structure_3d,
)
from torch_structure_manipulation import AtomicStructure

structure = AtomicStructure.from_dataframe(atoms, device="cuda")

grid_3d = GridConfig.from_grid_shape_and_voxel_size(
    grid_shape=(128, 128, 128),
    voxel_size=(1.0, 1.0, 1.0),
    center_zyx=(0.0, 0.0, 0.0),
    sublattice_radius=5.0,
)
volume = potential_from_structure_3d(
    structure,
    grid_3d,
    scattering_factors="peng_bonded",
    bonded_fallback="elemental",
)

grid_2d = GridConfig.from_grid_shape_and_voxel_size(
    grid_shape=(128, 128),
    voxel_size=(1.0, 1.0),
    center_yx=(0.0, 0.0),
)
projected = potential_from_structure_2d(structure, grid_2d)
```

`scattering_factors="peng_elemental"` is the default and ignores bonding
metadata. `"peng_bonded"` must be selected explicitly and requires
`bonded_environments` and per-atom `molecule_types`. Unsupported `other`
molecules and absent keys either emit one warning and use elemental values
(`bonded_fallback="elemental"`) or raise (`bonded_fallback="error"`).

The molecule type is the scattering-factor provider key, not merely descriptive
metadata. Custom providers can therefore supply different tables for protein,
RNA, or any additional molecule type:

## Batched structures and bonded factors

`AtomicStructure` may carry broadcast-compatible batch dimensions on positions
and other numerical fields. The tensor kernels and elemental Peng lookup support
that directly.

Bonded factors are different:

- `bonded_environments` and `molecule_types` are **flat tuples** (one string per
  atom), shared across the whole batch.
- `resolve_scattering_parameters(..., scattering_factors="peng_bonded")` requires
  **one-dimensional** `atomic_numbers` with shape `(n_atoms,)`.

Practical guidance:

| Use case | Elemental | Bonded |
|----------|-----------|--------|
| Single structure | yes | yes |
| Multiple poses, same chemistry (`positions` batched, `atomic_numbers` `(n,)`) | yes | yes |
| Batched `atomic_numbers` with shape `(batch, n_atoms)` | yes | no — raises |
| Different chemistry per batch member | N/A | no — not representable |

For different structures, call `potential_from_structure_3d` once per
`AtomicStructure` (or loop over batch indices).

```python
from torch_calculate_electrostatic_potential import BondedScatteringFactorTable

custom_factors = {
    "protein": BondedScatteringFactorTable(
        parameters_a=protein_parameters_a,
        parameters_b=protein_parameters_b,
    ),
    "rna": BondedScatteringFactorTable(
        parameters_a=rna_parameters_a,
        parameters_b=rna_parameters_b,
    ),
}
volume = potential_from_structure_3d(
    structure,
    grid_3d,
    scattering_factors=custom_factors,
    bonded_fallback="error",
)
```

Each parameter mapping is keyed by the structure's `bonded_environments`
strings. The low-level tensor API remains available for callers that have
already resolved arbitrary per-atom `a` and `b` tensors.

The lower-level route exposes parameter tensors directly:

```python
from torch_calculate_electrostatic_potential import (
    calculate_scattering_potential_3d,
    get_peng_scattering_parameters,
)

atom_params_a, atom_params_b = get_peng_scattering_parameters(atomic_numbers)
potential_volume = calculate_scattering_potential_3d(
    atom_pos_zyx,
    atom_bfactors,
    atom_params_a,
    atom_params_b,
    grid_3d,
    atom_occupancies=occupancies,
)
```

Positions, B-factors, occupancies, and explicit parameter tensors remain
differentiable. `sublattice_radius` controls the finite local stencil; increase
it for broad Gaussians.

## Continuum solvent (ice)

Solvent is modeled as an additive potential in **volts** on the same 3D grid as
the dry ESP. Geometry and potential layers are composable:

| Piece | Role |
|-------|------|
| `distance_to_surface` | Min surface distance `\|x−atom\| − VdW` |
| `vdw_probe_occupancy` | Binary solvent mask `dist >= probe_radius` |
| `constant_solvent_potential` | Bulk ice MIP × occupancy |
| `shang_sigworth_density` / `shang_sigworth_solvent_potential` | Shang & Sigworth (2012) continuum hydration layer |

### Quick start: ESP with ice

```python
import pandas as pd
from torch_structure_manipulation import AtomicStructure
from torch_calculate_electrostatic_potential import (
    GridConfig,
    solvated_potential_from_structure_3d,
)

# Build structure (or AtomicStructure.from_dataframe after mmdf.read(...))
atoms = pd.DataFrame(
    {
        "x": [0.0],
        "y": [0.0],
        "z": [0.0],
        "element": ["C"],
        "atom": ["C"],
        "b_isotropic": [20.0],
        "occupancy": [1.0],
    }
)
structure = AtomicStructure.from_dataframe(atoms)

grid = GridConfig.from_grid_shape_and_voxel_size(
    grid_shape=(64, 64, 64),
    voxel_size=(1.0, 1.0, 1.0),
    center_zyx=(0.0, 0.0, 0.0),
)

# Shang & Sigworth continuum hydration (default when water is on).
volume = solvated_potential_from_structure_3d(
    structure,
    grid,
    model_water_potential=True,  # defaults to solvent_model="shang_sigworth"
    ice_potential_V=3.6,
    probe_radius=1.4,
)

# Flat bulk ice instead:
# volume = solvated_potential_from_structure_3d(
#     structure, grid,
#     model_water_potential=True,
#     solvent_model="constant",
#     ice_potential_V=3.6,
# )

# Dry atoms only (same as potential_from_structure_3d):
# volume = solvated_potential_from_structure_3d(structure, grid)
# # or explicitly: model_water_potential=False

# Feed volts into multislice:
# from torch_scattering import multislice
# exit_wave = multislice(volume, pixel_size=1.0, voltage=300)
```

### Composable pieces

```python
from torch_calculate_electrostatic_potential import (
    distance_to_surface,
    vdw_probe_occupancy,
    constant_solvent_potential,
    shang_sigworth_solvent_potential,
    potential_from_structure_3d,
    solvent_potential_from_structure_3d,
)

atomic = potential_from_structure_3d(structure, grid)

# Option A: constant ice from a VdW+probe mask
dist, nearest_z = distance_to_surface(
    structure.positions_zyx, structure.atomic_numbers, grid
)
occupancy = vdw_probe_occupancy(dist, probe_radius=1.4)
solvent = constant_solvent_potential(occupancy, ice_potential_V=3.6)
volume = atomic + solvent

# Option B: Shang–Sigworth continuum solvent only
solvent = solvent_potential_from_structure_3d(
    structure, grid, model="shang_sigworth", ice_potential_V=3.6
)
volume = atomic + solvent

# Option C: build Shang–Sigworth potential from the distance field yourself
solvent = shang_sigworth_solvent_potential(
    dist, nearest_z, ice_potential_V=3.6, probe_radius=1.4
)
volume = atomic + solvent
```

Solvent helpers currently require an unbatched structure
(`positions_zyx` shape `(n_atoms, 3)`). Enclosed cavities are not flood-filled.

## Testing

Install the package together with test dependencies:

```sh
pip install "torch-calculate-electrostatic-potential[test]" @ git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git
pytest
```

With coverage: `pytest --cov=torch_calculate_electrostatic_potential --cov-report=html`.

## Requirements

- Python >= 3.11
- PyTorch >= 2.0
- torch-structure-manipulation, numpy, einops

## License

BSD 3-Clause License
