import torch
import torch.nn.functional as F


def homogenise_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """Homogenous coordinates to normal coordinates.

    Parameters
    ----------
    coords: torch.Tensor
        `(..., d)` array of d-dimensional coordinates

    Returns
    -------
    output: torch.Tensor
        `(..., d+1)` array of homogenous coordinates
    """
    return F.pad(torch.as_tensor(coords), pad=(0, 1), mode='constant', value=1)
