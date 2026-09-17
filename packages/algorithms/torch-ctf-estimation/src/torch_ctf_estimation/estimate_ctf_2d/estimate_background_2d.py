"""Estimate background in 2D from a power spectrum."""

from __future__ import annotations

from typing import TYPE_CHECKING

import einops
import torchvision.transforms.functional as TF
from torch_fourier_filter.dft_utils import rotational_average_dft_2d

if TYPE_CHECKING:
    import torch


def estimate_background_2d(
    power_spectrum: torch.Tensor,
    image_sidelength: int | tuple[int, int],
) -> torch.Tensor:
    """Estimate background in 2D from a power spectrum.

    Parameters
    ----------
    power_spectrum: torch.Tensor
        Power spectrum of the image, shape (H, W) for rfft (W = width//2+1).
    image_sidelength: int or (int, int)
        If int: sidelength of a square image (image_shape = (L, L)).
        If (height, width): image shape so output matches power_spectrum.

    Returns
    -------
    bg_estimate_2d: torch.Tensor
        Background estimate in 2D, same shape as power_spectrum.
    """
    device = power_spectrum.device
    if isinstance(image_sidelength, tuple):
        image_shape = image_sidelength
    else:
        image_shape = (image_sidelength, image_sidelength)

    raps_2d, _ = rotational_average_dft_2d(
        dft=power_spectrum.cpu(),
        image_shape=image_shape,
        rfft=True,
        fftshifted=False,
        return_1d_average=False,
    )
    raps_2d = raps_2d.to(device)
    raps_2d[0, 0] = 0
    raps_2d = einops.rearrange(raps_2d, "h w -> 1 1 h w")

    # Scale kernel size and sigma with patch size (use min dimension for non-square)
    reference_size = 256
    ref_dim = min(image_shape)
    kernel_size = int(25 * ref_dim / reference_size)
    kernel_size = (
        kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    )  # Must be odd
    sigma = 10.0 * ref_dim / reference_size

    bg_estimate_2d = TF.gaussian_blur(raps_2d, kernel_size=kernel_size, sigma=sigma)
    bg_estimate_2d = einops.rearrange(bg_estimate_2d, "1 1 h w -> h w")
    return bg_estimate_2d
