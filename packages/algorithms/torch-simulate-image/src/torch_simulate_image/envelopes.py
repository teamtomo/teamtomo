"""Envelope functions applied to intensity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_fourier_filter.envelopes import (
    Cc_envelope,
    Cs_envelope,
    b_envelope,
    dose_envelope,
)

from torch_simulate_image._validate import image_shape_from_tensor, validate_real_image

if TYPE_CHECKING:
    from torch_simulate_image.config import CtfConfig, EnvelopeConfig, FluenceConfig


def apply_envelopes(
    intensity: torch.Tensor,
    config: EnvelopeConfig,
    *,
    pixel_size: float,
    fluence: FluenceConfig,
    ctf: CtfConfig,
) -> torch.Tensor:
    """Apply optional envelopes in Fourier space.

    Supports B-factor, dose, Cs (spatial coherence), and Cc (temporal
    coherence) envelopes from ``torch_fourier_filter.envelopes``. Cs / Cc
    use defocus, Cs, and voltage from ``ctf``.

    Parameters
    ----------
    intensity : torch.Tensor
        Real intensity with shape ``(..., H, W)``.
    config : EnvelopeConfig
        Envelope options.
    pixel_size : float
        Pixel size in Angstroms.
    fluence : FluenceConfig
        Used for dose envelope fluence when enabled.
    ctf : CtfConfig
        Supplies voltage, defocus, and spherical aberration for Cs / Cc.

    Returns
    -------
    torch.Tensor
        Envelope-filtered intensity in real space.
    """
    validate_real_image(intensity)
    if not config.apply:
        return intensity
    any_enabled = (
        config.b_factor != 0.0
        or config.dose_envelope
        or config.cs_envelope
        or config.cc_envelope
    )
    if not any_enabled:
        return intensity

    image_shape = image_shape_from_tensor(intensity)
    device = intensity.device
    intensity_dft = torch.fft.rfft2(intensity, dim=(-2, -1))
    envelope = torch.ones(
        intensity_dft.shape[-2:],
        dtype=intensity.dtype,
        device=device,
    )

    if config.b_factor != 0.0:
        envelope = envelope * b_envelope(
            B=config.b_factor,
            image_shape=image_shape,
            pixel_size=pixel_size,
            rfft=True,
            fftshift=False,
            device=device,
        )

    if config.dose_envelope:
        envelope = envelope * dose_envelope(
            fluence=fluence.dose_e_per_A2,
            image_shape=image_shape,
            pixel_size=pixel_size,
            rfft=True,
            fftshift=False,
            device=device,
        )

    if config.cs_envelope:
        envelope = envelope * Cs_envelope(
            spherical_aberration=ctf.spherical_aberration_mm,
            defocus=ctf.defocus_um,
            image_shape=image_shape,
            pixel_size=pixel_size,
            rfft=True,
            fftshift=False,
            device=device,
            voltage=ctf.voltage_kv,
            alpha=config.illumination_semiangle_mrad,
        )

    if config.cc_envelope:
        envelope = envelope * Cc_envelope(
            chromatic_aberration=config.chromatic_aberration_mm,
            image_shape=image_shape,
            pixel_size=pixel_size,
            rfft=True,
            fftshift=False,
            device=device,
            voltage=ctf.voltage_kv,
            energy_spread=config.energy_spread_ev,
            deltaV_V=config.delta_v_over_v,
            deltaI_I=config.delta_i_over_i,
        )

    filtered_dft = intensity_dft * envelope
    filtered: torch.Tensor = torch.fft.irfft2(
        filtered_dft,
        s=image_shape,
        dim=(-2, -1),
        norm="backward",
    )
    return filtered
