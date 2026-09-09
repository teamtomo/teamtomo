import pandas as pd
import pytest
import torch
from torch_structure_manipulation import (
    AtomicStructure,
    annotate_bonding_environments,
)

from torch_calculate_electrostatic_potential import (
    BondedScatteringFactorTable,
    GridConfig,
    calculate_scattering_potential_3d,
    get_peng_scattering_parameters,
    potential_from_structure_2d,
    potential_from_structure_3d,
    resolve_scattering_parameters,
)
from torch_calculate_electrostatic_potential.utils import peng_model


def _grid(ndim):
    center = {"center_zyx": (0.0, 0.0, 0.0)} if ndim == 3 else {"center_yx": (0.0, 0.0)}
    return GridConfig.from_grid_shape_and_voxel_size(
        (11,) * ndim,
        (1.0,) * ndim,
        **center,
        sublattice_radius=5.0,
    )


def _frame(**metadata):
    data = {
        "x": [0.0, 1.2],
        "y": [0.0, -0.4],
        "z": [0.0, 0.7],
        "element": ["C", "O"],
        "atom": ["CA", "O"],
        "b_isotropic": [10.0, 12.0],
        "occupancy": [1.0, 0.8],
    }
    data.update(metadata)
    return pd.DataFrame(data)


def test_dataframe_structure_elemental_matches_direct_tensor_path():
    structure = AtomicStructure.from_dataframe(_frame())
    grid = _grid(3)
    actual = potential_from_structure_3d(structure, grid)
    a, b = get_peng_scattering_parameters(
        structure.atomic_numbers, device=grid.device, dtype=grid.dtype
    )
    expected = calculate_scattering_potential_3d(
        structure.positions_zyx,
        structure.b_factors,
        a,
        b,
        grid,
        atom_occupancies=structure.occupancies,
    )
    assert torch.equal(actual, expected)


def test_atomic_structure_batches_broadcast_into_potential_volumes():
    structure = AtomicStructure.from_dataframe(_frame())
    structure = structure.with_positions(
        torch.stack(
            (
                structure.positions_zyx,
                structure.positions_zyx + torch.tensor([0.0, 0.0, 1.0]),
            )
        )
    )

    potential = potential_from_structure_3d(structure, _grid(3))

    assert structure.batch_shape == (2,)
    assert potential.shape == (2, 11, 11, 11)
    assert not torch.equal(potential[0], potential[1])


def test_bonded_model_is_explicit_and_changes_parameters(monkeypatch):
    structure = AtomicStructure.from_dataframe(
        _frame(
            bonded_environments=["C(HHCC)", "O(C, amide)"],
            molecule_type=["protein", "protein"],
        )
    )
    grid = _grid(3)
    elemental = potential_from_structure_3d(structure, grid)
    bonded = potential_from_structure_3d(
        structure, grid, scattering_factors="peng_bonded", bonded_fallback="error"
    )
    assert not torch.allclose(elemental, bonded)

    monkeypatch.setattr(
        peng_model,
        "_load_bonded_providers",
        lambda: (_ for _ in ()).throw(AssertionError("metadata was inspected")),
    )
    potential_from_structure_3d(structure, grid, scattering_factors="peng_elemental")


def test_bonded_model_requires_metadata():
    structure = AtomicStructure.from_dataframe(_frame())
    with pytest.raises(ValueError, match="requires bonded_environments"):
        potential_from_structure_3d(
            structure, _grid(3), scattering_factors="peng_bonded"
        )


def test_bonded_fallback_warns_once_or_errors():
    structure = AtomicStructure.from_dataframe(
        _frame(
            bonded_environments=["missing-one", "missing-two"],
            molecule_type=["other", "protein"],
        )
    )
    with pytest.warns(UserWarning, match="using elemental fallback") as records:
        fallback = potential_from_structure_3d(
            structure,
            _grid(3),
            scattering_factors="peng_bonded",
            bonded_fallback="elemental",
        )
    assert len(records) == 1
    assert torch.isfinite(fallback).all()
    with pytest.raises(ValueError, match="unsupported bonded"):
        potential_from_structure_3d(
            structure,
            _grid(3),
            scattering_factors="peng_bonded",
            bonded_fallback="error",
        )


def test_custom_bonded_provider_is_selected_per_atom():
    providers = {
        "protein": BondedScatteringFactorTable(
            parameters_a={"X": [1.0] * 5},
            parameters_b={"X": [3.0] * 5},
        ),
        "rna": BondedScatteringFactorTable(
            parameters_a={"X": [2.0] * 5},
            parameters_b={"X": [4.0] * 5},
        ),
    }
    a, b = resolve_scattering_parameters(
        torch.tensor([6, 6]),
        scattering_factors=providers,
        bonded_environments=("X", "X"),
        molecule_types=("protein", "rna"),
        bonded_fallback="error",
        dtype=torch.float64,
    )
    assert a.dtype == torch.float64
    assert a[:, 0].tolist() == [1.0, 2.0]
    assert b[:, 0].tolist() == [3.0, 4.0]


def test_high_level_api_accepts_custom_protein_and_rna_factors():
    structure = AtomicStructure.from_dataframe(
        _frame(
            bonded_environments=["X", "X"],
            molecule_type=["protein", "rna"],
        )
    )
    providers = {
        "protein": BondedScatteringFactorTable(
            parameters_a={"X": [1.0] * 5},
            parameters_b={"X": [3.0] * 5},
        ),
        "rna": BondedScatteringFactorTable(
            parameters_a={"X": [2.0] * 5},
            parameters_b={"X": [4.0] * 5},
        ),
    }

    potential = potential_from_structure_3d(
        structure,
        _grid(3),
        scattering_factors=providers,
        bonded_fallback="error",
    )

    assert potential.shape == (11, 11, 11)
    assert torch.isfinite(potential).all()


def test_bonding_canonical_keys_match_packaged_parameters():
    protein_atoms = pd.DataFrame(
        [
            ("A", 1, "ALA", "C", "C", 0.0),
            ("A", 1, "ALA", "O", "O", 1.0),
            ("A", 1, "ALA", "CA", "C", 2.0),
            ("A", 2, "GLY", "CA", "C", 3.0),
            ("A", 2, "GLY", "N", "N", 4.0),
        ],
        columns=["chain", "residue_id", "residue", "atom", "element", "x"],
    )
    rna_atoms = pd.DataFrame(
        [
            ("R", 1, "A", "C3'", "C", 0.0),
            ("R", 1, "A", "O3'", "O", 1.0),
            ("R", 2, "U", "P", "P", 2.0),
            ("R", 2, "U", "O5'", "O", 3.0),
        ],
        columns=["chain", "residue_id", "residue", "atom", "element", "x"],
    )
    for atoms, row, expected in (
        (protein_atoms, 0, "C(CNO)"),
        (rna_atoms, 1, "O(CP)"),
    ):
        atoms["y"] = 0.0
        atoms["z"] = 0.0
        annotated = annotate_bonding_environments(atoms, include_hydrogens=False)
        assert annotated.loc[row, "bonded_environments"] == expected
        structure = AtomicStructure.from_dataframe(annotated.iloc[[row]])
        potential = potential_from_structure_3d(
            structure,
            _grid(3),
            scattering_factors="peng_bonded",
            bonded_fallback="error",
        )
        assert torch.isfinite(potential).all()


def test_structure_2d_projects_z_and_preserves_gradients():
    structure = AtomicStructure.from_dataframe(_frame())
    positions = structure.positions_zyx.clone().requires_grad_(True)
    differentiable = structure.with_positions(positions)
    image = potential_from_structure_2d(differentiable, _grid(2))
    assert image.shape == (11, 11)
    image.sum().backward()
    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()
    assert positions.grad[:, 0].count_nonzero() == 0
    assert positions.grad[:, 1:].abs().sum() > 0


def test_bonded_scattering_factor_table_reports_gaussian_term_count():
    table = BondedScatteringFactorTable(
        parameters_a={"X": [1.0, 2.0, 3.0, 4.0, 5.0]},
        parameters_b={"X": [6.0, 7.0, 8.0, 9.0, 10.0]},
    )
    assert table.n_gaussian_terms == 5


def test_bonded_scattering_factor_table_validates_sequence_lengths():
    with pytest.raises(ValueError, match="same length"):
        BondedScatteringFactorTable(
            parameters_a={"X": [1.0, 2.0]},
            parameters_b={"X": [3.0]},
        )
    with pytest.raises(ValueError, match="same number of Gaussian terms"):
        BondedScatteringFactorTable(
            parameters_a={"X": [1.0, 2.0], "Y": [3.0, 4.0, 5.0]},
            parameters_b={"X": [1.0, 2.0], "Y": [3.0, 4.0, 5.0]},
        )


def test_resolve_scattering_parameters_rejects_mismatched_gaussian_term_count():
    providers = {
        "protein": BondedScatteringFactorTable(
            parameters_a={"X": [1.0, 2.0, 3.0]},
            parameters_b={"X": [4.0, 5.0, 6.0]},
        )
    }
    with pytest.raises(ValueError, match="currently requires 5"):
        resolve_scattering_parameters(
            torch.tensor([6]),
            scattering_factors=providers,
            bonded_environments=("X",),
            molecule_types=("protein",),
            bonded_fallback="error",
        )
