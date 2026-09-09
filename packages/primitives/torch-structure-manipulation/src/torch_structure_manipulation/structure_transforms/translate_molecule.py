"""Functions for translating molecular structures."""

import numpy as np
import pandas as pd
import torch

from .utils import df_to_atomxyz, df_to_atomzyx


def apply_translation(
    df: pd.DataFrame,
    translation_vector: tuple[float, float, float] | np.ndarray | torch.Tensor,
    zyx: bool = True,
) -> pd.DataFrame:
    """Apply a translation to the structure.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with coordinate columns. Columns depend on zyx parameter:
        - If zyx=True: DataFrame must have z, y, x columns
        - If zyx=False: DataFrame must have x, y, z columns
    translation_vector : tuple[float, float, float] | np.ndarray | torch.Tensor
        Translation vector. Order depends on zyx parameter:
        - If zyx=True: (dz, dy, dx) order
        - If zyx=False: (dx, dy, dz) order
    zyx : bool, default=True
        If True, coordinates are in (z, y, x) order. If False, in (x, y, z) order.
        Coordinates are used as-is without reordering.

    Returns
    -------
    pd.DataFrame
        DataFrame with translated coordinates in the same format as input
    """
    if zyx:
        coords = df_to_atomzyx(df)
        translated_coords = apply_translation_to_coords(coords, translation_vector)
        df = df.copy()
        df[["z", "y", "x"]] = translated_coords.cpu().numpy()
    else:
        coords = df_to_atomxyz(df)
        translated_coords = apply_translation_to_coords(coords, translation_vector)
        df = df.copy()
        df[["x", "y", "z"]] = translated_coords.cpu().numpy()
    return df


def apply_translation_to_coords(
    coordinates: torch.Tensor,
    translation_vector: tuple[float, float, float] | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Apply a translation to coordinate tensor.

    Parameters
    ----------
    coordinates : torch.Tensor
        Tensor of shape (n_atoms, 3) containing coordinates in any order
        (e.g., z, y, x or x, y, z)
    translation_vector : tuple[float, float, float] | np.ndarray | torch.Tensor
        Translation vector. Must match the coordinate order of the input tensor.

    Returns
    -------
    torch.Tensor
        Translated coordinate tensor in the same order as input
    """
    if isinstance(translation_vector, (tuple, list)):
        translation = torch.tensor(
            translation_vector, dtype=torch.float32, device=coordinates.device
        )
    elif isinstance(translation_vector, np.ndarray):
        translation = (
            torch.from_numpy(translation_vector).float().to(coordinates.device)
        )
    else:
        translation = translation_vector.to(coordinates.device)

    return coordinates + translation
