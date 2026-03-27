import warnings
from typing import Tuple, Optional

import numpy as np
import torch
from scipy.interpolate import LSQBivariateSpline


def estimate_local_mean(
    image: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    resolution: Tuple[int, int] = (5, 5),
    n_samples_for_fit: int = 20000,
):
    """Estimate local mean of an image with a bivariate cubic spline.

    A mask can be provided to

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` array containing image data.
    mask: Optional[torch.Tensor]
        `(h, w)` array containing a binary mask specifying foreground
        and background pixels for the estimation.
    resolution: Tuple[int, int]
        Resolution of the local mean estimate in dimensions `h` and `w`.
    n_samples_for_fit: int
        Number of samples taken from foreground pixels for background mean estimation.
        The number of background pixels will be used if this number is greater than the
        number of background pixels.

    Returns
    -------
    local_mean: torch.Tensor
        `(h, w)` array containing a local estimate of the local mean.
    """
    input_dtype = image.dtype
    image = image.numpy()
    mask = np.ones_like(image) if mask is None else mask.numpy()

    # get a random set of foreground pixels for the background fit
    foreground_sample_idx = np.argwhere(mask == 1)

    n_samples_for_fit = min(n_samples_for_fit, len(foreground_sample_idx))
    selection = np.random.choice(
        foreground_sample_idx.shape[0], size=n_samples_for_fit, replace=False
    )
    foreground_sample_idx = foreground_sample_idx[selection]
    y, x = foreground_sample_idx[:, 0], foreground_sample_idx[:, 1]
    z = image[(y, x)]

    # fit a bivariate spline to the data with the specified background model resolution
    ty = np.linspace(0, image.shape[0], num=resolution[0])
    tx = np.linspace(0, image.shape[1], num=resolution[1])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        background_model = LSQBivariateSpline(y, x, z, tx, ty)

    # evaluate the model over a grid covering the whole image
    y = np.arange(image.shape[-2])
    x = np.arange(image.shape[-1])
    local_mean = background_model(y, x, grid=True)
    return torch.tensor(local_mean, dtype=input_dtype)
