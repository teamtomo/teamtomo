"""Coordinate helpers for affine transforms."""

import torch
import torch.nn.functional as F


def homogenise_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """Convert coordinates to homogeneous coordinates by appending a 1.

    Parameters
    ----------
    coords : torch.Tensor
        `(..., d)` array of d-dimensional coordinates.

    Returns
    -------
    torch.Tensor
        `(..., d+1)` array of homogeneous coordinates, with 1 in the last
        position.
    """
    return F.pad(torch.as_tensor(coords), pad=(0, 1), mode="constant", value=1)
