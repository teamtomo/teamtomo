"""Detector quantum efficiency via MTF convolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_fourier_filter.mtf import make_mtf_grid

from torch_simulate_image._validate import image_shape_from_tensor, validate_real_image
from torch_simulate_image.detector.mtf import resolve_mtf_tensors

if TYPE_CHECKING:
    from torch_simulate_image.config import DqeConfig

_EPS = 1e-12


def apply_dqe(
    image: torch.Tensor,
    config: DqeConfig,
) -> torch.Tensor:
    """Apply detector MTF in Fourier space, preserving the spatial mean.

    Parameters
    ----------
    image : torch.Tensor
        Real image or counts with shape ``(..., H, W)``.
    config : DqeConfig
        DQE options. When ``config.apply`` is ``False``, returns ``image``
        unchanged.

    Returns
    -------
    torch.Tensor
        MTF-blurred image with the same shape as ``image``.
    """
    validate_real_image(image)
    if not config.apply:
        return image

    image_shape = image_shape_from_tensor(image)
    frequencies, amplitudes = resolve_mtf_tensors(config)
    frequencies = frequencies.to(device=image.device, dtype=image.dtype)
    amplitudes = amplitudes.to(device=image.device, dtype=image.dtype)

    mtf = make_mtf_grid(
        image_shape=image_shape,
        mtf_frequencies=frequencies,
        mtf_amplitudes=amplitudes,
        rfft=True,
        fftshift=False,
        device=image.device,
    )
    # Preserve mean intensity by normalizing the DC component to unity.
    dc_index = (0, 0)
    mtf = mtf / mtf[dc_index].clamp(min=_EPS)

    mean_before = image.mean(dim=(-2, -1), keepdim=True)
    image_dft = torch.fft.rfft2(image, dim=(-2, -1))
    blurred_dft = image_dft * mtf
    blurred = torch.fft.irfft2(
        blurred_dft,
        s=image_shape,
        dim=(-2, -1),
        norm="backward",
    )
    mean_after = blurred.mean(dim=(-2, -1), keepdim=True).clamp(min=_EPS)
    scaled: torch.Tensor = blurred * (mean_before / mean_after)
    return scaled
