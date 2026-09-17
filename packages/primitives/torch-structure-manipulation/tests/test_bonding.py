import pandas as pd

from torch_structure_manipulation import (
    annotate_bonding_environments,
    classify_structure_composition,
    get_scattering_provider_keys,
)


def _atoms(rows):
    return pd.DataFrame(
        rows, columns=["chain", "residue_id", "residue", "atom", "element"]
    )


def test_mixed_and_unknown_residues_are_classified_per_atom():
    df = _atoms(
        [
            ("A", 1, "ALA", "CA", "C"),
            ("B", 1, "U", "C2", "C"),
            ("C", 1, "LIG", "C1", "C"),
        ]
    )
    result = annotate_bonding_environments(df, include_hydrogens=False)
    assert result["molecule_type"].tolist() == ["protein", "rna", "other"]
    assert result.loc[2, "bonded_environments"] == "C()"
    assert "bonded_environments" not in df


def test_peptide_bonds_are_independent_of_atom_row_order():
    df = _atoms(
        [
            ("A", 1, "ALA", "C", "C"),
            ("A", 1, "ALA", "O", "O"),
            ("A", 1, "ALA", "CA", "C"),
            ("A", 2, "GLY", "CA", "C"),
            ("A", 2, "GLY", "N", "N"),
        ]
    )
    result = annotate_bonding_environments(df, include_hydrogens=False)
    assert result.loc[0, "bonded_environments"] == "C(CNO)"
    assert result.loc[1, "bonded_environments"] == "O(C, amide)"
    assert result.loc[4, "bonded_environments"] == "N(CC)"


def test_rna_inter_residue_and_terminal_bonds():
    df = _atoms(
        [
            ("R", 1, "A", "C3'", "C"),
            ("R", 1, "A", "O3'", "O"),
            ("R", 2, "U", "P", "P"),
            ("R", 2, "U", "O5'", "O"),
            ("R", 2, "U", "C3'", "C"),
            ("R", 2, "U", "O3'", "O"),
        ]
    )
    result = annotate_bonding_environments(df)
    assert result.loc[1, "bonded_environments"] == "O(CP)"
    assert result.loc[2, "bonded_environments"] == "P(OO)"
    assert result.loc[5, "bonded_environments"] == "O(CH)"


def test_hydrogen_inclusion_is_deterministic():
    df = _atoms(
        [
            ("A", 1, "ALA", "N", "N"),
            ("A", 1, "ALA", "CA", "C"),
            ("A", 1, "ALA", "H", "H"),
        ]
    )
    with_h = annotate_bonding_environments(df)
    without_h = annotate_bonding_environments(df, include_hydrogens=False)
    assert with_h.loc[0, "bonded_environments"] == "N(CH)"
    assert without_h.loc[0, "bonded_environments"] == "N(C)"
    assert with_h.loc[2, "bonded_environments"] == "H(N)"


def test_missing_columns_are_reported():
    try:
        annotate_bonding_environments(pd.DataFrame({"residue": ["ALA"]}))
    except ValueError as error:
        assert "missing required structure columns" in str(error)
    else:
        raise AssertionError("expected validation failure")


def test_public_scattering_provider_apis():
    df = _atoms(
        [
            ("A", 1, "ALA", "CA", "C"),
            ("B", 1, "U", "C2", "C"),
            ("C", 1, "LIG", "C1", "C"),
        ]
    )

    assert get_scattering_provider_keys(df) == ["protein", "rna", "other"]
    assert classify_structure_composition(df) == "rna+protein"
    assert classify_structure_composition(df.iloc[[2]]) == "other"
