"""Utility functions for structure transformations."""

import pandas as pd
import torch


def df_to_atomzyx(df: pd.DataFrame) -> torch.Tensor:
    """Extract atom coordinates from DataFrame as torch tensor.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with z, y, x columns

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_atoms, 3) containing z, y, x coordinates
    """
    return torch.tensor(df[["z", "y", "x"]].values, dtype=torch.float32)


def df_to_atomxyz(df: pd.DataFrame) -> torch.Tensor:
    """Extract atom coordinates from DataFrame as torch tensor.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with x, y, z columns

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_atoms, 3) containing x, y, z coordinates
    """
    return torch.tensor(df[["x", "y", "z"]].values, dtype=torch.float32)


def get_protein_residues() -> set[str]:
    """Get set of standard and non-standard amino acid residue names.

    Returns
    -------
    set[str]
        Set of amino acid residue names (uppercase).
    """
    return {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        # Non-standard amino acids
        "SEC",
        "PYL",
        "MSE",
        "HYP",
        "NLE",
    }


def get_nucleic_acid_residues() -> set[str]:
    """Get set of nucleic acid (RNA/DNA) residue names.

    Returns
    -------
    set[str]
        Set of nucleic acid residue names (uppercase).
    """
    return {
        "A",
        "T",
        "G",
        "C",
        "U",  # Standard bases
        "DA",
        "DT",
        "DG",
        "DC",  # DNA
        "ADE",
        "THY",
        "GUA",
        "CYT",
        "URA",  # Full names
        # Modified bases
        "PSU",
        "I",
        "M7G",
        "M2G",
        "M22G",
        "YYG",
        "H2U",
        "OMC",
        "OMG",
    }
