"""Annotate atomic bonding environments from packaged residue templates."""

from __future__ import annotations

import json
from importlib.resources import files
from itertools import pairwise
from typing import Any

import pandas as pd

from .structure_transforms import (
    get_nucleic_acid_residues,
    get_protein_residues,
)

_REQUIRED_COLUMNS = {"atom", "chain", "element", "residue", "residue_id"}
NUCLEIC_ACID_RESIDUES = get_nucleic_acid_residues()
PROTEIN_RESIDUES = get_protein_residues()


def _load_templates() -> tuple[
    dict[str, dict[str, list[str]]], dict[str, dict[str, list[str]]]
]:
    resource = files(__package__).joinpath("bonding_data.json")
    data: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return data["protein"], data["rna"]


_PROTEIN_BONDING, _RNA_BONDING = _load_templates()


def annotate_bonding_environments(
    df: pd.DataFrame, include_hydrogens: bool = True
) -> pd.DataFrame:
    """Return a copy with ``bonded_environments`` and per-atom ``molecule_type``.

    The input follows mmdf naming conventions but is not tied to mmdf itself.
    Bonding templates are keyed by residue and atom names; adjacent numeric
    residue IDs are used to add peptide and phosphodiester bonds.
    """
    missing = sorted(_REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"missing required structure columns: {missing}")

    result = df.copy()
    if result.empty:
        result["bonded_environments"] = pd.Series(dtype="object", index=result.index)
        result["molecule_type"] = pd.Series(dtype="object", index=result.index)
        return result

    residues = [str(value).strip().upper() for value in result["residue"]]
    atoms = [str(value).strip().upper() for value in result["atom"]]
    elements = [str(value).strip().upper() for value in result["element"]]
    residue_ids = [str(value) for value in result["residue_id"]]
    chains = [str(value) for value in result["chain"]]

    residue_lookup: dict[tuple[str, str], dict[str, str]] = {}
    residue_names: dict[tuple[str, str], str] = {}
    for residue, atom, element, residue_id, chain in zip(
        residues, atoms, elements, residue_ids, chains, strict=True
    ):
        key = (chain, residue_id)
        residue_lookup.setdefault(key, {})[atom] = element
        residue_names.setdefault(key, residue)

    next_residue, previous_residue = _build_residue_order(residue_names)
    molecule_types = [_molecule_type(residue) for residue in residues]
    environments = [
        _environment_for_atom(
            residue=residue,
            atom=atom,
            element=element,
            key=(chain, residue_id),
            residue_lookup=residue_lookup,
            next_residue=next_residue,
            previous_residue=previous_residue,
            include_hydrogens=include_hydrogens,
        )
        for residue, atom, element, residue_id, chain in zip(
            residues, atoms, elements, residue_ids, chains, strict=True
        )
    ]
    result["bonded_environments"] = environments
    result["molecule_type"] = molecule_types
    return result


def classify_structure_composition(df: pd.DataFrame) -> str:
    """Summarize residue composition for the whole table.

    Returns an aggregate label such as ``"protein"``, ``"rna"``,
    ``"rna+protein"``, or ``"other"``. This is descriptive metadata only and
    is not a valid per-atom scattering-provider key.
    """
    residues = {str(value).strip().upper() for value in df["residue"]}
    has_protein = bool(residues & PROTEIN_RESIDUES)
    has_rna = bool(residues & NUCLEIC_ACID_RESIDUES)
    if has_protein and has_rna:
        return "rna+protein"
    if has_protein:
        return "protein"
    if has_rna:
        return "rna"
    return "other"


def get_scattering_provider_keys(df: pd.DataFrame) -> list[str]:
    """Return the Peng scattering-provider key for each atom.

    Each value is ``"protein"``, ``"rna"``, or ``"other"`` and matches the
    ``molecule_type`` column written by :func:`annotate_bonding_environments`.
    """
    return [_molecule_type(str(residue).strip().upper()) for residue in df["residue"]]


def _environment_for_atom(
    *,
    residue: str,
    atom: str,
    element: str,
    key: tuple[str, str],
    residue_lookup: dict[tuple[str, str], dict[str, str]],
    next_residue: dict[tuple[str, str], tuple[str, str]],
    previous_residue: dict[tuple[str, str], tuple[str, str]],
    include_hydrogens: bool,
) -> str:
    template = _template_for(residue)
    bonded_names = list(template.get(atom, ()))
    if residue in _RNA_BONDING and atom == "O3'" and key in next_residue:
        bonded_names = [name for name in bonded_names if name != "HO3'"]

    bonded_elements: list[str] = []
    residue_atoms = residue_lookup.get(key, {})
    for bonded_name in bonded_names:
        bonded_element = _find_element(bonded_name, residue_atoms)
        if bonded_element:
            if include_hydrogens or bonded_element != "H":
                bonded_elements.append(bonded_element)
        elif include_hydrogens and _is_hydrogen_name(bonded_name):
            bonded_elements.append("H")

    if residue in _PROTEIN_BONDING and atom == "C" and key in next_residue:
        _append_atom_element(
            bonded_elements, "N", residue_lookup[next_residue[key]], include_hydrogens
        )
    elif residue in _PROTEIN_BONDING and atom == "N" and key in previous_residue:
        _append_atom_element(
            bonded_elements,
            "C",
            residue_lookup[previous_residue[key]],
            include_hydrogens,
        )
    elif residue in _RNA_BONDING and atom == "O3'" and key in next_residue:
        _append_atom_element(
            bonded_elements, "P", residue_lookup[next_residue[key]], include_hydrogens
        )
    elif residue in _RNA_BONDING and atom == "P" and key in previous_residue:
        _append_atom_element(
            bonded_elements,
            "O3'",
            residue_lookup[previous_residue[key]],
            include_hydrogens,
        )

    bonded_key = "".join(sorted(bonded_elements))
    category = _oxygen_carbon_category(
        residue, atom, element, bonded_key, key, residue_lookup, next_residue
    )
    suffix = f", {category}" if category is not None else ""
    return f"{element}({bonded_key}{suffix})"


def _oxygen_carbon_category(
    residue: str,
    atom: str,
    element: str,
    bonded_key: str,
    key: tuple[str, str],
    residue_lookup: dict[tuple[str, str], dict[str, str]],
    next_residue: dict[tuple[str, str], tuple[str, str]],
) -> str | None:
    if element != "O" or bonded_key != "C":
        return None
    if atom == "OXT" or (residue == "ASP" and atom in {"OD1", "OD2"}):
        return "carboxyl"
    if residue == "GLU" and atom in {"OE1", "OE2"}:
        return "carboxyl"
    if (
        residue in _PROTEIN_BONDING
        and atom == "O"
        and key in next_residue
        and _find_element("N", residue_lookup[next_residue[key]])
    ):
        return "amide"
    return None


def _build_residue_order(
    residue_names: dict[tuple[str, str], str],
) -> tuple[
    dict[tuple[str, str], tuple[str, str]],
    dict[tuple[str, str], tuple[str, str]],
]:
    by_chain: dict[str, list[tuple[float, tuple[str, str]]]] = {}
    for key in residue_names:
        try:
            numeric_id = float(key[1])
        except ValueError:
            continue
        by_chain.setdefault(key[0], []).append((numeric_id, key))

    next_residue: dict[tuple[str, str], tuple[str, str]] = {}
    previous_residue: dict[tuple[str, str], tuple[str, str]] = {}
    for residues in by_chain.values():
        residues.sort(key=lambda item: item[0])
        for (left_id, left_key), (right_id, right_key) in pairwise(residues):
            if abs(right_id - left_id - 1.0) < 1e-6:
                next_residue[left_key] = right_key
                previous_residue[right_key] = left_key
    return next_residue, previous_residue


def _template_for(residue: str) -> dict[str, list[str]]:
    if residue in _PROTEIN_BONDING:
        return _PROTEIN_BONDING[residue]
    return _RNA_BONDING.get(residue, {})


def _molecule_type(residue: str) -> str:
    if residue in PROTEIN_RESIDUES:
        return "protein"
    if residue in NUCLEIC_ACID_RESIDUES:
        return "rna"
    return "other"


def _append_atom_element(
    elements: list[str],
    atom_name: str,
    residue_atoms: dict[str, str],
    include_hydrogens: bool,
) -> None:
    element = _find_element(atom_name, residue_atoms)
    if element and (include_hydrogens or element != "H"):
        elements.append(element)


def _find_element(atom_name: str, residue_atoms: dict[str, str]) -> str:
    if atom_name in residue_atoms:
        return residue_atoms[atom_name]
    normalized = atom_name.replace("'", "").replace("*", "")
    for candidate, element in residue_atoms.items():
        if candidate.replace("'", "").replace("*", "") == normalized:
            return element
    return ""


def _is_hydrogen_name(atom_name: str) -> bool:
    return atom_name.upper().startswith("H")
