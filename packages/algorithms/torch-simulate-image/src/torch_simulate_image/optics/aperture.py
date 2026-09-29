"""Objective aperture (pupil) applied to complex exit waves."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_ctf import calculate_relativistic_electron_wavelength
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from torch_simulate_image._validate import image_shape_from_tensor, validate_exit_wave

if TYPE_CHECKING:
    from torch_simulate_image.config import ObjectiveApertureConfig


def resolve_aperture_cutoff_frequency(
    config: ObjectiveApertureConfig,
    *,
    voltage_kv: float,
) -> float:
    """Return the aperture cutoff spatial frequency in Å⁻¹.

    Parameters
    ----------
    config : ObjectiveApertureConfig
        Aperture options with exactly one cutoff source set.
    voltage_kv : float
        Acceleration voltage in kV (used when converting semi-angle).

    Returns
    -------
    float
        Cutoff frequency ``q_max`` in Å⁻¹.
    """
    if config.cutoff_frequency_inv_A is not None:
        return float(config.cutoff_frequency_inv_A)
    if config.outer_semiangle_mrad is None:
        msg = "Objective aperture cutoff is not specified."
        raise ValueError(msg)

    wavelength_m = calculate_relativistic_electron_wavelength(voltage_kv * 1e3)
    wavelength_A = float(wavelength_m) * 1e10
    alpha_rad = config.outer_semiangle_mrad * 1e-3
    return alpha_rad / wavelength_A


def make_objective_aperture_mask(
    image_shape: tuple[int, int],
    *,
    pixel_size: float,
    q_max: float,
    soft_edge_half_width_inv_A: float = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a circular pupil mask for a full (complex) FFT.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Real-space ``(H, W)``.
    pixel_size : float
        Pixel size in Angstroms.
    q_max : float
        Cutoff spatial frequency in Å⁻¹.
    soft_edge_half_width_inv_A : float
        Cosine soft-edge half-width in Å⁻¹. Roll-off spans
        ``[q_max - w, q_max + w]``. ``0`` yields a hard mask.
    device : torch.device or None
        Device for the mask.
    dtype : torch.dtype
        Real dtype of the mask.

    Returns
    -------
    torch.Tensor
        Real mask of shape ``image_shape``, values in ``[0, 1]``.
    """
    freq = fftfreq_grid(
        image_shape=image_shape,
        rfft=False,
        fftshift=False,
        spacing=pixel_size,
        norm=True,
        device=device,
    ).to(dtype=dtype)

    if soft_edge_half_width_inv_A <= 0.0:
        mask = torch.zeros_like(freq)
        mask[freq <= q_max] = 1.0
        return mask

    # Cosine roll-off from 1 → 0 over [q_max - w, q_max + w].
    half_width = soft_edge_half_width_inv_A
    inner = q_max - half_width
    outer = q_max + half_width
    mask = torch.ones_like(freq)
    transition = (freq > inner) & (freq < outer)
    # Map freq in [inner, outer] to [0, 1]; cos(pi t) goes 1 → -1 → rescale to 1 → 0.
    t = (freq[transition] - inner) / (2.0 * half_width)
    mask[transition] = 0.5 * (1.0 + torch.cos(torch.pi * t))
    mask[freq >= outer] = 0.0
    return mask


def apply_objective_aperture(
    exit_wave: torch.Tensor,
    config: ObjectiveApertureConfig,
    *,
    pixel_size: float,
    voltage_kv: float,
) -> torch.Tensor:
    """Apply a circular objective aperture to a complex exit wave.

    Multiplies ``FFT(ψ)`` by a circular pupil ``A(q)`` (hard or soft edge),
    removing electrons scattered beyond the cutoff before image formation.

    Parameters
    ----------
    exit_wave : torch.Tensor
        Complex tensor with shape ``(..., H, W)``.
    config : ObjectiveApertureConfig
        Aperture options. When ``config.apply`` is ``False``, returns
        ``exit_wave`` unchanged.
    pixel_size : float
        Pixel size in Angstroms.
    voltage_kv : float
        Acceleration voltage in kV (for semi-angle → frequency conversion).

    Returns
    -------
    torch.Tensor
        Aperture-filtered complex exit wave.
    """
    validate_exit_wave(exit_wave)
    if not config.apply:
        return exit_wave

    image_shape = image_shape_from_tensor(exit_wave)
    q_max = resolve_aperture_cutoff_frequency(config, voltage_kv=voltage_kv)
    real_dtype = torch.float32 if exit_wave.dtype == torch.complex64 else torch.float64
    mask = make_objective_aperture_mask(
        image_shape,
        pixel_size=pixel_size,
        q_max=q_max,
        soft_edge_half_width_inv_A=config.soft_edge_half_width_inv_A,
        device=exit_wave.device,
        dtype=real_dtype,
    )
    exit_dft = torch.fft.fft2(exit_wave, dim=(-2, -1))
    filtered_dft = exit_dft * mask.to(dtype=exit_wave.dtype)
    filtered: torch.Tensor = torch.fft.ifft2(
        filtered_dft, dim=(-2, -1), norm="backward"
    )
    return filtered.to(dtype=exit_wave.dtype)
