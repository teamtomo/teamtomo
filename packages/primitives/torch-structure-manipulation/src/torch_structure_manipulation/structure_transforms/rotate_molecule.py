"""Functions for rotating molecular structures."""

import numpy as np
import pandas as pd
import roma
import torch

from .utils import df_to_atomxyz, df_to_atomzyx


def apply_rotation(
    df: pd.DataFrame,
    rotation_matrix: np.ndarray | torch.Tensor,
    center_point: tuple[float, float, float] | None = None,
    zyx: bool = True,
) -> pd.DataFrame:
    """Apply a rotation matrix to the structure.

    Parameters
    ----------
    df : pd.DataFrame
        Structure DataFrame with coordinate columns. Columns depend on zyx parameter:
        - If zyx=True: DataFrame must have z, y, x columns
        - If zyx=False: DataFrame must have x, y, z columns
    rotation_matrix : np.ndarray | torch.Tensor
        3x3 rotation matrix designed for (x, y, z) coordinates
    center_point : tuple[float, float, float] | None
        Point to rotate around. Order depends on zyx parameter:
        - If zyx=True: (z, y, x) order
        - If zyx=False: (x, y, z) order
        If None, rotates around origin
    zyx : bool, default=True
        If True, coordinates are in (z, y, x) order. If False, in (x, y, z) order.
        Coordinates are used as-is without reordering.

    Returns
    -------
    pd.DataFrame
        DataFrame with rotated coordinates in the same format as input
    """
    if zyx:
        coords = df_to_atomzyx(df)
        rotated_coords = apply_rotation_to_coords(
            coords, rotation_matrix, center_point, zyx=True
        )
        df = df.copy()
        df[["z", "y", "x"]] = rotated_coords.cpu().numpy()
    else:
        coords = df_to_atomxyz(df)
        rotated_coords = apply_rotation_to_coords(
            coords, rotation_matrix, center_point, zyx=False
        )
        df = df.copy()
        df[["x", "y", "z"]] = rotated_coords.cpu().numpy()
    return df


def apply_rotation_to_coords(
    coordinates: torch.Tensor,
    rotation_matrix: np.ndarray | torch.Tensor,
    center_point: tuple[float, float, float] | None = None,
    zyx: bool = True,
) -> torch.Tensor:
    """Apply a rotation matrix to coordinate tensor.

    Parameters
    ----------
    coordinates : torch.Tensor
        Tensor of shape (n_atoms, 3) containing coordinates in any order
        (e.g., z, y, x or x, y, z)
    rotation_matrix : np.ndarray | torch.Tensor
        3x3 rotation matrix designed for (x, y, z) coordinates
    center_point : tuple[float, float, float] | None
        Point to rotate around. Order depends on zyx parameter:
        - If zyx=True: (z, y, x) order
        - If zyx=False: (x, y, z) order
        If None, rotates around origin
    zyx : bool, default=True
        If True, coordinates are in (z, y, x) order. If False, in (x, y, z) order.

    Returns
    -------
    torch.Tensor
        Rotated coordinate tensor in the same order as input
    """
    # Convert to torch tensor if needed
    if isinstance(rotation_matrix, np.ndarray):
        rotation_matrix = torch.from_numpy(rotation_matrix).float()

    # Ensure rotation_matrix is on same device as coordinates
    rotation_matrix = rotation_matrix.to(coordinates.device)

    # Center coordinates if specified
    if center_point is not None:
        center = torch.tensor(
            center_point, dtype=torch.float32, device=coordinates.device
        )
        coordinates = coordinates - center

    if zyx:
        # Convert zyx coordinates to xyz for rotation
        # zyx: [z, y, x] -> xyz: [x, y, z]
        coords_xyz = coordinates[:, [2, 1, 0]]

        # Apply rotation (rotation_matrix is designed for xyz)
        rotated_xyz = torch.matmul(coords_xyz, rotation_matrix.T)

        # Convert back from xyz to zyx
        # xyz: [x, y, z] -> zyx: [z, y, x]
        rotated_coords = rotated_xyz[:, [2, 1, 0]]
    else:
        # Coordinates are already in xyz, apply rotation directly
        rotated_coords = torch.matmul(coordinates, rotation_matrix.T)

    # Translate back if centered
    if center_point is not None:
        rotated_coords = rotated_coords + center

    return rotated_coords


def create_rotation_matrix_from_euler(
    angles: torch.Tensor,
    order: str = "ZYZ",
    degrees: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create rotation matrix from Euler angles using roma.

    Parameters
    ----------
    angles : torch.Tensor
        Euler angles as a tensor.
        Shape can be (3,) for single rotation or (..., 3) for batch.
        The last dimension must be 3, corresponding to (alpha, beta, gamma).
    order : str
        Rotation order convention (e.g., 'xyz', 'zyx', 'ZYZ', 'zyz').
        Uppercase letters indicate intrinsic rotations.
        Lowercase indicate extrinsic rotations.
    degrees : bool
        If True, input angles are in degrees. If False, angles are in radians.
        Default is True (degrees).
    device : torch.device | None
        Device on which to perform computation.
        If None, uses the device of the input tensor.

    Returns
    -------
    torch.Tensor
        Rotation matrix.
        Shape is (3, 3) for single rotation or (..., 3, 3) for batch.
    """
    # Move to device if specified
    if device is not None:
        angles = angles.to(device)
    else:
        device = angles.device

    # Use roma to construct rotation matrix
    # roma uses uppercase for intrinsic, lowercase for extrinsic rotations
    rot_mat = roma.euler_to_rotmat(order, angles, degrees=degrees, device=device)

    return torch.as_tensor(rot_mat, dtype=angles.dtype, device=device)
