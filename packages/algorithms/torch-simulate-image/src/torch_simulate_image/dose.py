"""Dose weighting wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_fourier_filter.dose_weight import dose_weight_2d

from torch_simulate_image._validate import image_shape_from_tensor, validate_real_image

if TYPE_CHECKING:
    from torch_simulate_image.config import DoseWeightConfig


def apply_dose_weight(
    image: torch.Tensor,
    config: DoseWeightConfig,
    *,
    pixel_size: float,
    voltage_kv: float,
) -> torch.Tensor:
    """Apply Grant & Grigorieff dose weighting to a real-space image.

    Parameters
    ----------
    image : torch.Tensor
        Real intensity with shape ``(..., H, W)``.
    config : DoseWeightConfig
        Dose weighting options.
    pixel_size : float
        Pixel size in Angstroms.
    voltage_kv : float
        Acceleration voltage in kilovolts.

    Returns
    -------
    torch.Tensor
        Dose-weighted image in real space.
    """
    validate_real_image(image)
    if not config.apply:
        return image

    image_shape = image_shape_from_tensor(image)
    dose = 0.5 * (config.dose_start + config.dose_end)
    image_dft = torch.fft.rfft2(image, dim=(-2, -1))
    weighted_dft = dose_weight_2d(
        image_dft=image_dft,
        image_shape=image_shape,
        pixel_size=pixel_size,
        dose=dose,
        voltage=voltage_kv,
        crit_exposure_bfactor=config.crit_exposure_bfactor,
        rfft=True,
        fftshift=False,
        device=image.device,
    )
    weighted: torch.Tensor = torch.fft.irfft2(
        weighted_dft,
        s=image_shape,
        dim=(-2, -1),
        norm="backward",
    )
    return weighted
