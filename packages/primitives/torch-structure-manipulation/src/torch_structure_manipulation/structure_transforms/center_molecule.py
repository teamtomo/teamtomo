"""Functions for centering molecular structures."""

import warnings

import pandas as pd
import torch

from .utils import df_to_atomxyz, df_to_atomzyx


def center_structure(
    df: pd.DataFrame,
    center_point: tuple[float, float, float] | None = None,
    use_center_of_mass: bool = False,
    atom_selection: pd.Series | None = None,
    zyx: bool = True,
) -> pd.DataFrame:
    """Center DataFrame at specified point or origin.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with coordinate columns. Columns depend on zyx parameter:
        - If zyx=True: DataFrame must have z, y, x columns
        - If zyx=False: DataFrame must have x, y, z columns
    center_point : tuple[float, float, float] | None
        Target center point. Order depends on zyx parameter:
        - If zyx=True: (z, y, x) order
        - If zyx=False: (x, y, z) order
        If None, centers at origin.
    use_center_of_mass : bool
        If True, uses center of mass. If False, uses geometric center.
    atom_selection : pd.Series | None
        Boolean series to select atoms for center calculation
    zyx : bool, default=True
        If True, coordinates are in (z, y, x) order. If False, in (x, y, z) order.
        Coordinates are used as-is without reordering.

    Returns
    -------
    pd.DataFrame
        Centered DataFrame in the same format as input
    """
    if len(df) == 0:
        return df.copy()

    if zyx:
        coords = df_to_atomzyx(df)
        centered_coords = center_structure_from_coords(
            coords, center_point, use_center_of_mass, atom_selection, df, zyx=True
        )
        df_result = df.copy()
        df_result[["z", "y", "x"]] = centered_coords.numpy()
    else:
        coords = df_to_atomxyz(df)
        centered_coords = center_structure_from_coords(
            coords, center_point, use_center_of_mass, atom_selection, df, zyx=False
        )
        df_result = df.copy()
        df_result[["x", "y", "z"]] = centered_coords.numpy()
    return df_result


def center_structure_from_coords(
    coordinates: torch.Tensor,
    center_point: tuple[float, float, float] | None = None,
    use_center_of_mass: bool = False,
    atom_selection: pd.Series | None = None,
    df: pd.DataFrame | None = None,
    zyx: bool = True,
) -> torch.Tensor:
    """Center structure from coordinate tensor to specified center point.

    Parameters
    ----------
    coordinates : torch.Tensor
        Tensor of shape (n_atoms, 3) containing coordinates in any order
        (e.g., z, y, x or x, y, z)
    center_point : tuple[float, float, float] | None
        Target center point. Order depends on zyx parameter:
        - If zyx=True: (z, y, x) order
        - If zyx=False: (x, y, z) order
        If None, centers at origin.
    use_center_of_mass : bool
        If True, uses center of mass. If False, uses geometric center.
    atom_selection : pd.Series | None
        Boolean series to select atoms for center calculation
    df : pd.DataFrame | None
        DataFrame to extract masses from if needed for center of mass.
        Required if use_center_of_mass is True.
    zyx : bool, default=True
        If True, coordinates are in (z, y, x) order. If False, in (x, y, z) order.

    Returns
    -------
    torch.Tensor
        Centered coordinate tensor in the same order as input
    """
    # Extract selection mask if provided
    selection_mask = None
    if atom_selection is not None:
        selection_mask = torch.tensor(atom_selection.values, dtype=torch.bool)

    # Extract masses from DataFrame if needed for center of mass
    masses = None
    if use_center_of_mass and df is not None:
        if "atomic_weight" in df.columns:
            masses = torch.tensor(df["atomic_weight"].values, dtype=torch.float32)
        elif "atomic_number" in df.columns:
            masses = torch.tensor(df["atomic_number"].values, dtype=torch.float32)

    if selection_mask is not None:
        selected_coords = coordinates[selection_mask]
        selected_masses = masses[selection_mask] if masses is not None else None
    else:
        selected_coords = coordinates
        selected_masses = masses

    if center_point is None:
        target_center = torch.zeros(3, dtype=torch.float32, device=coordinates.device)
    else:
        target_center = torch.tensor(
            center_point, dtype=torch.float32, device=coordinates.device
        )

    current_center = calculate_center_from_tensors(
        selected_coords, use_center_of_mass, selected_masses
    )

    translation = target_center - current_center
    centered_coords = coordinates + translation

    return centered_coords


def calculate_center_from_tensors(
    coords: torch.Tensor, use_center_of_mass: bool, masses: torch.Tensor | None = None
) -> torch.Tensor:
    """Calculate center from coordinates and masses tensors.

    Parameters
    ----------
    coords : torch.Tensor
        Tensor of shape (n_atoms, 3) containing coordinates
    use_center_of_mass : bool
        If True, uses center of mass. If False, uses geometric center.
    masses : torch.Tensor | None
        Tensor of atomic masses. Required if use_center_of_mass is True.

    Returns
    -------
    torch.Tensor
        Center point as tensor of shape (3,)
    """
    if not use_center_of_mass:
        return torch.mean(coords, dim=0)

    if masses is None:
        # Fallback to geometric center if no masses provided
        warnings.warn(
            "use_center_of_mass is True but masses is None. "
            "Falling back to geometric center.",
            stacklevel=2,
        )
        return torch.mean(coords, dim=0)

    # Ensure masses are on same device as coords
    masses = masses.to(coords.device)

    total_mass = torch.sum(masses)
    if total_mass == 0:
        warnings.warn(
            "use_center_of_mass is True but total_mass is 0. "
            "Falling back to geometric center.",
            stacklevel=2,
        )
        return torch.mean(coords, dim=0)

    center_of_mass = torch.sum(coords * masses.unsqueeze(1), dim=0) / total_mass
    return center_of_mass
