import einops
import torch
import torch.nn.functional as F
from typing import Any, Union, Literal
import numpy as np 
import pandas as pd


def _find_peaks_2d_torch(
        image: torch.Tensor,
        min_distance: int = 1,
        threshold_abs: float = 0.0,
        exclude_border: int = 0,
) -> torch.Tensor:
    """
    Internal function to find local peaks in a 2D tensor.
    """
    image = einops.rearrange(image, "h w -> 1 1 h w")
    mask = F.max_pool2d(
        image,
        kernel_size=(min_distance * 2 + 1, min_distance * 2 + 1),
        stride=1,
        padding=min_distance,
    )
    mask = einops.rearrange(mask, "1 1 h w -> h w")
    image = einops.rearrange(image, "1 1 h w -> h w")
    mask = (image == mask) & (image > threshold_abs)
    if exclude_border > 0:
        mask[:exclude_border, :] = False
        mask[-exclude_border:, :] = False
        mask[:, :exclude_border] = False
        mask[:, -exclude_border:] = False

    coords = torch.nonzero(mask, as_tuple=False)
    heights = image[mask]
    return coords, heights


def find_peaks_2d(
        image: Union[torch.Tensor, np.ndarray], 
        min_distance: int = 1,
        threshold_abs: float = 0.0,
        exclude_border: int = 0,
        return_as: Literal["torch","numpy","dataframe"] = "torch",
) -> torch.Tensor:
    """
    Find local peaks in a 2D image.

    Accepts various input types (torch.Tensor, numpy.ndarray) and attempts
    to convert them to torch.Tensor before processing.

    Parameters
    ----------
    image : Any
        A 2D tensor-like object (e.g., torch.Tensor, numpy.ndarray)
        representing the input image.
    min_distance : int, optional
        Minimum distance between peaks. Default is 1.
    threshold_abs : float, optional
        Minimum intensity value for a peak to be considered. Default is 0.0.
    exclude_border : int, optional
        Width of the border to exclude from peak detection. Default is 0.
    return_as : str, optional
        The format of the output. Default is "torch".
        Other options are "numpy" and "dataframe".

    Returns
    -------
    torch.Tensor
        A tensor of shape (N, 2), where N is the number of peaks, and each row
        contains the (Y, X) coordinates of a peak.

    Raises
    ------
    TypeError
        If the input image cannot be converted to a torch.Tensor.
    ValueError
        If the input image is not 2-dimensional after conversion.
    """
    if isinstance(image, torch.Tensor):
        image_tensor = image
    elif isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image)
    # Add checks for pandas/polars DataFrames/Series here if needed
    # elif pd and isinstance(image, pd.DataFrame):
    #     image_tensor = torch.from_numpy(image.values)
    # elif pl and isinstance(image, pl.DataFrame):
    #     image_tensor = torch.from_numpy(image.to_numpy())
    else:
        try:
            # Attempt a general conversion for other array-like objects
            image_tensor = torch.as_tensor(image)
        except Exception as e:
            raise TypeError(
                f"Input type {type(image)} not supported or conversion failed: {e}"
            )

    if image_tensor.ndim != 2:
        raise ValueError(
            f"Input image must be 2-dimensional, but got {image_tensor.ndim} dimensions."
        )

    found_peaks, heights = _find_peaks_2d_torch(
        image=image_tensor,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=exclude_border,
    )

    if return_as == "torch":
        return found_peaks, heights
    elif return_as == "numpy":
        return found_peaks.numpy(), heights.numpy()
    elif return_as == "dataframe":
        # Use einops.pack to properly handle tensors with different dimensions
        # First tensor has shape [N, 2], second has shape [N]
        # We're packing them along the second dimension (dim=1)
        packed, _ = einops.pack([found_peaks, heights], 'n *')
        return pd.DataFrame(packed.cpu(), columns=["y", "x", "height"])
    else:
        raise ValueError(f"Invalid return_as value: {return_as}")

def _find_peaks_3d_torch(
        volume: torch.Tensor,
        min_distance: int = 1,
        threshold_abs: float = 0.0,
        exclude_border: int = 0,
) -> torch.Tensor:
    """
    Internal function to find local peaks in a 3D tensor.
    """
    volume = einops.rearrange(volume, "d h w -> 1 1 d h w")
    mask = F.max_pool3d(
        volume,
        kernel_size=(min_distance * 2 + 1, min_distance * 2 + 1, min_distance * 2 + 1),
        stride=1,
        padding=min_distance,
    )
    mask = einops.rearrange(mask, "1 1 d h w -> d h w")
    volume = einops.rearrange(volume, "1 1 d h w -> d h w")
    mask = (volume == mask) & (volume > threshold_abs)
    if exclude_border > 0:
        mask[:exclude_border, :, :] = False
        mask[-exclude_border:, :, :] = False
        mask[:, :exclude_border, :] = False
        mask[:, -exclude_border:, :] = False
        mask[:, :, :exclude_border] = False
        mask[:, :, -exclude_border:] = False

    coords = torch.nonzero(mask, as_tuple=False)
    heights = volume[mask]
    return coords, heights  


def find_peaks_3d(
        volume: Union[torch.Tensor, np.ndarray], 
        min_distance: int = 1,
        threshold_abs: float = 0.0,
        exclude_border: int = 0,
        return_as: Literal["torch","numpy","dataframe"] = "torch",
) -> torch.Tensor:
    """
    Find local peaks in a 3D volume.

    Accepts various input types (torch.Tensor, numpy.ndarray) and attempts
    to convert them to torch.Tensor before processing.

    Parameters
    ----------
    volume : Any
        A 3D tensor-like object (e.g., torch.Tensor, numpy.ndarray)
        representing the input volume.
    min_distance : int, optional
        Minimum distance between peaks. Default is 1.
    threshold_abs : float, optional
        Minimum intensity value for a peak to be considered. Default is 0.0.
    exclude_border : int, optional
        Width of the border to exclude from peak detection. Default is 0.
    return_as : str, optional
        The format of the output. Default is "torch".
        Other options are "numpy" and "dataframe".

    Returns
    -------
    torch.Tensor
        A tensor of shape (N, 3), where N is the number of peaks, and each row
        contains the (Z, Y, X) coordinates of a peak.

    Raises
    ------
    TypeError
        If the input volume cannot be converted to a torch.Tensor.
    ValueError
        If the input volume is not 3-dimensional after conversion.
    """
    if isinstance(volume, torch.Tensor):
        volume_tensor = volume
    elif isinstance(volume, np.ndarray):
        volume_tensor = torch.from_numpy(volume)
    # Add checks for pandas/polars DataFrames/Series here if needed
    else:
        try:
            # Attempt a general conversion for other array-like objects
            volume_tensor = torch.as_tensor(volume)
        except Exception as e:
            raise TypeError(
                f"Input type {type(volume)} not supported or conversion failed: {e}"
            )

    if volume_tensor.ndim != 3:
        raise ValueError(
            f"Input volume must be 3-dimensional, but got {volume_tensor.ndim} dimensions."
        )

    found_peaks, heights = _find_peaks_3d_torch(
        volume=volume_tensor,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=exclude_border,
    )

    if return_as == "torch":
        return found_peaks, heights
    elif return_as == "numpy":
        return found_peaks.cpu().numpy(), heights.cpu().numpy()
    elif return_as == "dataframe":
        # Use einops.pack to properly handle tensors with different dimensions
        # First tensor has shape [N, 3], second has shape [N]
        # We're packing them along the second dimension (dim=1)
        packed, _ = einops.pack([found_peaks, heights], 'n *')
        return pd.DataFrame(packed.cpu(), columns=["z", "y", "x", "height"])
    else:
        raise ValueError(f"Invalid return_as value: {return_as}")
