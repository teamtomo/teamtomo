"""Test-only integration from atomic metadata to multislice exit waves."""

import pandas as pd
import torch
from torch_calculate_electrostatic_potential import (
    GridConfig,
    calculate_scattering_potential_3d,
    get_peng_scattering_parameters,
    potential_from_structure_3d,
)
from torch_structure_manipulation import (
    AtomicStructure,
    annotate_bonding_environments,
)

from torch_scattering import multislice


def test_structure_potentials_feed_multislice_as_real_tensors():
    atoms = pd.DataFrame(
        [
            ("A", 1, "ALA", "C", "C", 0.0, 0.0, 0.0),
            ("A", 1, "ALA", "O", "O", 1.2, 0.0, 0.0),
            ("A", 1, "ALA", "CA", "C", -1.2, 0.0, 0.0),
            ("A", 2, "GLY", "N", "N", 2.4, 0.0, 0.0),
        ],
        columns=[
            "chain",
            "residue_id",
            "residue",
            "atom",
            "element",
            "x",
            "y",
            "z",
        ],
    )
    atoms["b_isotropic"] = 10.0
    atoms["occupancy"] = 1.0
    annotated = annotate_bonding_environments(atoms, include_hydrogens=False)
    structure = AtomicStructure.from_dataframe(annotated.iloc[[0]])
    grid = GridConfig.from_grid_shape_and_voxel_size(
        (9, 9, 9),
        (1.0, 1.0, 1.0),
        center_zyx=(0.0, 0.0, 0.0),
        sublattice_radius=4.0,
    )

    elemental = potential_from_structure_3d(structure, grid)
    bonded = potential_from_structure_3d(
        structure,
        grid,
        scattering_factors="peng_bonded",
        bonded_fallback="error",
    )

    parameters_a, parameters_b = get_peng_scattering_parameters(
        structure.atomic_numbers, device=grid.device, dtype=grid.dtype
    )
    direct_elemental = calculate_scattering_potential_3d(
        structure.positions_zyx,
        structure.b_factors,
        parameters_a,
        parameters_b,
        grid,
        atom_occupancies=structure.occupancies,
    )
    assert torch.equal(elemental, direct_elemental)

    for potential in (elemental, bonded):
        assert potential.dtype == torch.float32
        wave = multislice(potential, pixel_size=1.0, voltage=300.0)
        assert wave.dtype == torch.complex64
        assert wave.shape == (9, 9)
        assert torch.isfinite(wave).all()
