"""Functions for selecting and filtering atoms."""

import warnings

import pandas as pd
import torch

from .utils import (
    df_to_atomxyz,
    df_to_atomzyx,
    get_nucleic_acid_residues,
    get_protein_residues,
)


def find_atoms_in_ball(
    df: pd.DataFrame,
    center: tuple[float, float, float],
    radius: float,
    zyx: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find atoms inside and outside a ball (sphere) of specified radius.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with coordinate columns (z, y, x if zyx=True, or
        x, y, z if zyx=False)
    center : tuple[float, float, float]
        Center point for ball query. Order depends on zyx parameter:
        - If zyx=True: (z, y, x) order
        - If zyx=False: (x, y, z) order
    radius : float
        Radius in Angstroms
    zyx : bool, default=True
        If True, coordinates are in (z, y, x) order. If False, in (x, y, z) order.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (atoms_inside, atoms_outside) DataFrames
    """
    if len(df) == 0:
        empty_df = df.copy()
        return empty_df, empty_df

    inside_mask = ball_query_atoms(df, center, radius, zyx=zyx)
    outside_mask = ~inside_mask

    atoms_inside = df[inside_mask.cpu().numpy()].copy()
    atoms_outside = df[outside_mask.cpu().numpy()].copy()

    return atoms_inside, atoms_outside


def ball_query_atoms(
    coordinates: torch.Tensor | pd.DataFrame,
    center: tuple[float, float, float],
    radius: float,
    zyx: bool = True,
) -> torch.Tensor:
    """Query atoms within a ball (sphere) and return a boolean mask.

    Parameters
    ----------
    coordinates : torch.Tensor | pd.DataFrame
        Either a tensor of shape (n_atoms, 3) with coordinates, or a DataFrame
        with coordinate columns. If DataFrame, columns depend on zyx parameter:
        - If zyx=True: DataFrame must have z, y, x columns
        - If zyx=False: DataFrame must have x, y, z columns
    center : tuple[float, float, float]
        Center point for ball query. Order depends on zyx parameter:
        - If zyx=True: (z, y, x) order
        - If zyx=False: (x, y, z) order
    radius : float
        Radius in Angstroms
    zyx : bool, default=True
        If True, coordinates are in (z, y, x) order. If False, in (x, y, z) order.
        Coordinates are used as-is without reordering.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape (n_atoms,) indicating which atoms are inside the ball
    """
    # Extract coordinates from DataFrame if needed
    if isinstance(coordinates, pd.DataFrame):
        if zyx:
            # DataFrame has z, y, x columns - use as-is
            coords = df_to_atomzyx(coordinates)
        else:
            # DataFrame has x, y, z columns - use as-is
            coords = df_to_atomxyz(coordinates)
    else:
        # Tensor is provided - assume it's already in the correct order based on zyx
        coords = coordinates

    # Ensure coordinates are on the same device
    device = coords.device
    center_tensor = torch.tensor(center, dtype=torch.float32, device=device)

    # Calculate distances
    distances = torch.norm(coords - center_tensor, dim=1)
    inside_mask = distances <= radius

    return torch.as_tensor(inside_mask, dtype=torch.bool, device=coords.device)


def remove_sidechains(
    df: pd.DataFrame, keep_backbone_atoms: list[str] | None = None
) -> pd.DataFrame:
    """Remove sidechain atoms, keeping only backbone atoms.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with 'atom' column containing atom names
    keep_backbone_atoms : List[str] | None
        List of atom names to keep. If None, uses standard protein backbone atoms.

    Returns
    -------
    pd.DataFrame
        DataFrame with only backbone atoms
    """
    if keep_backbone_atoms is None:
        # Standard protein backbone atoms
        keep_backbone_atoms = ["N", "CA", "C", "O", "H", "HA", "OXT"]
        # Add nucleic acid backbone atoms
        keep_backbone_atoms.extend(
            ["P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
        )

    # Filter atoms
    backbone_mask = df["atom"].isin(keep_backbone_atoms)

    return df[backbone_mask].copy()


def separate_protein_rna(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate protein and RNA/DNA components.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with 'residue' column

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (protein_df, nucleic_acid_df)
    """
    amino_acids = get_protein_residues()
    nucleic_acids = get_nucleic_acid_residues()

    # Separate based on residue names
    protein_mask = df["residue"].isin(amino_acids)
    nucleic_mask = df["residue"].isin(nucleic_acids)

    protein_df = df[protein_mask].copy()
    nucleic_df = df[nucleic_mask].copy()

    # Handle remaining residues (warn if significant amount)
    remaining = df[~(protein_mask | nucleic_mask)]
    if len(remaining) > 0:
        warnings.warn(
            f"Found {len(remaining)} atoms in {remaining['residue'].nunique()} "
            f"unrecognized residue types: {set(remaining['residue'].unique())}",
            stacklevel=2,
        )

    return protein_df, nucleic_df
