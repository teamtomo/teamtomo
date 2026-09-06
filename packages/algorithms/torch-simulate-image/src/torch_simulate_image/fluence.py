"""Fluence scaling from relative intensity to expected counts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch_simulate_image._validate import validate_real_image

if TYPE_CHECKING:
    import torch

    from torch_simulate_image.config import FluenceConfig

_EPS = 1e-12


def scale_to_expected_counts(
    intensity: torch.Tensor,
    config: FluenceConfig,
    *,
    pixel_size: float,
) -> torch.Tensor:
    """Scale relative intensity to expected electron counts per pixel.

    Intensity is normalized by its spatial mean, then converted with::

        λ = (I / mean(I)) * dose_e_per_A2 * pixel_size² * coincidence_loss

    so a uniform wave averages to
    ``dose_e_per_A2 * pixel_size² * coincidence_loss`` counts per pixel.

    Parameters
    ----------
    intensity : torch.Tensor
        Non-negative real intensity with shape ``(..., H, W)``.
    config : FluenceConfig
        Fluence and coincidence-loss options.
    pixel_size : float
        Pixel size in Angstroms.

    Returns
    -------
    torch.Tensor
        Expected counts ``λ`` with the same shape as ``intensity``.
    """
    validate_real_image(intensity)
    pixel_area = pixel_size * pixel_size
    mean_intensity = intensity.mean(dim=(-2, -1), keepdim=True).clamp(min=_EPS)
    intensity_norm = intensity / mean_intensity
    return intensity_norm * config.dose_e_per_A2 * pixel_area * config.coincidence_loss
