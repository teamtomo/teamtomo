from typing import Tuple

import numpy as np
import torch
import einops

from torch_cryoeraser.utils import estimate_background_std
from torch_cryoeraser.sparse_local_mean import estimate_local_mean


def erase_region_2d(
    image: torch.Tensor,
    mask: torch.Tensor,
    background_intensity_model_resolution: Tuple[int, int] = (5, 5),
    background_intensity_model_samples: int = 20000,
) -> torch.Tensor:
    """Inpaint image(s) with gaussian noise matching local image mean and std.


    Parameters
    ----------
    image: torch.Tensor
        `(..., h, w)` or `(h, w)` array containing image data for erase.
    mask: torch.Tensor
        `(..., h, w)` or `(h, w)` binary mask.
        Foreground pixels (1) will be erased.
    background_intensity_model_resolution: Tuple[int, int]
        Number of points in each image dimension for the background mean model.
        Minimum of two points in each dimension.
    background_intensity_model_samples: int
        Number of sample points used to determine the model of the background mean.

    Returns
    -------
    erased_image: torch.Tensor
        `(..., h, w)` or `(h, w)` array containing image data inpainted in the foreground pixels of the mask
        with gaussian noise matching the local mean and global standard deviation of the image
        for background pixels.
    """
    # coerce to tensor
    image = torch.as_tensor(image)
    mask = torch.as_tensor(mask, dtype=torch.bool)

    # shape check
    if image.shape != mask.shape:
        raise ValueError("image shape must match mask shape.")

    # pack into (b, h, w)
    image, ps = einops.pack([image], "* h w")
    mask, _ = einops.pack([mask], "* h w")

    # allocate for output
    erased_image = torch.empty_like(image)

    # process images, one at a time
    for idx, _image in enumerate(image):
        erased_image[idx] = _erase_single_image(
            image=_image,
            mask=mask[idx],
            background_model_resolution=background_intensity_model_resolution,
            n_background_samples=background_intensity_model_samples,
        )

    # unpack to original shape
    [erased_image] = einops.unpack(erased_image, packed_shapes=ps, pattern="* h w")

    return torch.as_tensor(erased_image, dtype=torch.float32)


def _erase_single_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    background_model_resolution: Tuple[int, int] = (5, 5),
    n_background_samples: int = 20000,
) -> np.ndarray:
    """Erase masked regions of an image with gaussian noise.

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` array containing image data for erase.
    mask: torch.Tensor
        `(h, w)` binary mask separating foreground from background pixels.
        Foreground pixels (value == 1) will be erased.
    background_model_resolution: Tuple[int, int]
        Number of points in each image dimension for the background mean model.
        Minimum of two points in each dimension.
    n_background_samples: int
        Number of sampling points for background mean estimation.

    Returns
    -------
    inpainted_image: torch.Tensor
        `(h, w)` array containing image data inpainted in the foreground pixels of the mask
        with gaussian noise matching the local mean and global standard deviation of the image
        for background pixels.
    """
    inpainted_image = torch.clone(torch.as_tensor(image))
    local_mean = estimate_local_mean(
        image=image,
        mask=torch.logical_not(mask),
        resolution=background_model_resolution,
        n_samples_for_fit=n_background_samples,
    )

    # fill foreground pixels with local mean
    idx_foreground = torch.argwhere(mask.bool())  # (b, 2)
    n_foreground_pixels = len(idx_foreground)
    idx_h, idx_w = idx_foreground[:, -2], idx_foreground[:, -1]
    inpainted_image[idx_h, idx_w] = local_mean[idx_h, idx_w]

    # add noise with mean=0 std=background std estimate
    background_std = estimate_background_std(image, mask)
    noise = np.random.normal(loc=0, scale=background_std, size=n_foreground_pixels)
    inpainted_image[idx_h, idx_w] += torch.as_tensor(noise)
    return inpainted_image
