"""Dose weighting wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_fourier_filter.dose_weight import cumulative_dose_filter_3d

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
    """Apply Grant & Grigorieff radiation damage to a real-space image.

    Each frequency is scaled by the mean of the exposure filter
    ``exp(-N / 2 Ne)`` over the cumulative exposures ``N`` from
    ``config.dose_start`` to ``config.dose_end``
    (:func:`torch_fourier_filter.dose_weight.cumulative_dose_filter_3d`).

    Parameters
    ----------
    image : torch.Tensor
        Real intensity with shape ``(..., H, W)``.
    config : DoseWeightConfig
        Dose weighting options.
    pixel_size : float
        Pixel size in Angstroms.
    voltage_kv : float
        Acceleration voltage in kilovolts. At 200 and 100 kV the exposures are
        divided by 0.8 and 0.64, as in RELION.

    Returns
    -------
    torch.Tensor
        Dose-weighted image in real space.
    """
    validate_real_image(image)
    if not config.apply:
        return image
    if config.dose_end is None:  # the config was changed after validation
        msg = "When dose_weight.apply is True, dose_end must be set."
        raise ValueError(msg)

    image_shape = image_shape_from_tensor(image)
    dose_filter = cumulative_dose_filter_3d(
        volume_shape=image_shape,
        pixel_size=pixel_size,
        start_exposure=config.dose_start,
        end_exposure=config.dose_end,
        voltage=voltage_kv,
        crit_exposure_bfactor=config.crit_exposure_bfactor,
        rfft=True,
        fftshift=False,
        device=image.device,
    )
    image_dft = torch.fft.rfft2(image, dim=(-2, -1))
    weighted: torch.Tensor = torch.fft.irfft2(
        image_dft * dose_filter,
        s=image_shape,
        dim=(-2, -1),
        norm="backward",
    )
    return weighted
