import pandas as pd
import pytest
import torch

from torch_structure_manipulation import AtomicStructure


def test_from_dataframe_maps_xyz_to_zyx_and_defaults():
    df = pd.DataFrame(
        {
            "x": [1.0, 4.0],
            "y": [2.0, 5.0],
            "z": [3.0, 6.0],
            "element": ["C", "O"],
            "atom": ["CA", "O"],
        }
    )

    structure = AtomicStructure.from_dataframe(df, dtype=torch.float64)

    assert structure.positions_zyx.dtype == torch.float64
    assert torch.equal(
        structure.positions_zyx, torch.tensor([[3, 2, 1], [6, 5, 4]]).double()
    )
    assert structure.atomic_numbers.tolist() == [6, 8]
    assert structure.atom_names == ("CA", "O")
    assert structure.b_factors.tolist() == [0.0, 0.0]
    assert structure.occupancies.tolist() == [1.0, 1.0]


def test_from_dataframe_uses_metadata_and_device():
    df = pd.DataFrame(
        {
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            "element": ["N"],
            "b_isotropic": [12.5],
            "occupancy": [0.75],
            "bonded_environments": ["N(CH)"],
            "molecule_type": ["protein"],
        }
    )
    structure = AtomicStructure.from_dataframe(df, device="cpu")
    assert structure.positions_zyx.device.type == "cpu"
    assert structure.b_factors.item() == 12.5
    assert structure.occupancies.item() == 0.75
    assert structure.bonded_environments == ("N(CH)",)
    assert structure.molecule_types == ("protein",)


def test_with_positions_is_non_mutating_and_supports_new_batch_shape():
    original = AtomicStructure.from_dataframe(
        pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0], "element": ["H"]})
    )
    replacement = torch.zeros(4, 1, 3)
    moved = original.with_positions(replacement)
    assert moved is not original
    assert moved.positions_zyx is replacement
    assert moved.batch_shape == (4,)
    assert original.positions_zyx.tolist() == [[3.0, 2.0, 1.0]]
    with pytest.raises(ValueError, match="same number of atoms"):
        original.with_positions(torch.zeros(2, 3))


def test_tensor_fields_support_batching_and_broadcasting():
    structure = AtomicStructure(
        positions_zyx=torch.zeros(2, 1, 5, 3),
        atomic_numbers=torch.tensor([1, 6, 7, 8, 16]),
        elements=("H", "C", "N", "O", "S"),
        atom_names=("",) * 5,
        b_factors=torch.zeros(1, 3, 5),
        occupancies=torch.tensor(1.0),
    )

    assert structure.num_atoms == 5
    assert structure.batch_shape == (2, 3)


def test_tensor_fields_reject_incompatible_batches_and_atom_dimensions():
    common = {
        "elements": ("H", "C"),
        "atom_names": ("", ""),
        "b_factors": torch.zeros(2),
        "occupancies": torch.ones(2),
    }
    with pytest.raises(ValueError, match="atomic_numbers"):
        AtomicStructure(
            positions_zyx=torch.zeros(2, 3),
            atomic_numbers=torch.tensor([1]),
            **common,
        )
    with pytest.raises(ValueError, match="incompatible batch"):
        AtomicStructure(
            positions_zyx=torch.zeros(2, 2, 3),
            atomic_numbers=torch.ones(3, 2, dtype=torch.int64),
            **common,
        )


def test_to_preserves_integer_dtype_and_moves_floating_dtype():
    structure = AtomicStructure.from_dataframe(
        pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0], "element": ["H"]})
    )

    converted = structure.to(dtype=torch.float64)

    assert converted.positions_zyx.dtype == torch.float64
    assert converted.b_factors.dtype == torch.float64
    assert converted.occupancies.dtype == torch.float64
    assert converted.atomic_numbers.dtype == torch.int64


def test_from_dataframe_validates_columns_and_elements():
    with pytest.raises(ValueError, match="missing required"):
        AtomicStructure.from_dataframe(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="unknown element"):
        AtomicStructure.from_dataframe(
            pd.DataFrame({"x": [1], "y": [2], "z": [3], "element": ["unobtainium"]})
        )


def test_from_annotated_dataframe_adds_bonding_metadata():
    df = pd.DataFrame(
        [
            ("A", 1, "ALA", "C", "C", 0.0, 0.0, 0.0),
            ("A", 1, "ALA", "O", "O", 1.0, 0.0, 0.0),
            ("A", 1, "ALA", "CA", "C", 2.0, 0.0, 0.0),
            ("A", 2, "GLY", "N", "N", 3.0, 0.0, 0.0),
        ],
        columns=["chain", "residue_id", "residue", "atom", "element", "x", "y", "z"],
    )

    structure = AtomicStructure.from_annotated_dataframe(df, include_hydrogens=False)

    assert structure.bonded_environments is not None
    assert structure.molecule_types is not None
    assert structure.bonded_environments[0] == "C(CNO)"
    assert all(molecule_type == "protein" for molecule_type in structure.molecule_types)


def test_structure_transforms_are_reexported_from_root():
    import torch_structure_manipulation as tsm

    for name in (
        "center_structure",
        "apply_rotation_to_coords",
        "df_to_atomzyx",
    ):
        assert name in tsm.__all__
        assert callable(getattr(tsm, name))
