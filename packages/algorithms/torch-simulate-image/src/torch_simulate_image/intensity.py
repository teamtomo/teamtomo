"""Intensity formation from complex exit waves."""

from __future__ import annotations

import torch

from torch_simulate_image._validate import validate_exit_wave


def exit_wave_to_intensity(exit_wave: torch.Tensor) -> torch.Tensor:
    """Convert a complex exit wave to real intensity.

    Parameters
    ----------
    exit_wave : torch.Tensor
        Complex tensor with shape ``(..., H, W)``.

    Returns
    -------
    torch.Tensor
        Real intensity ``|ψ|²`` with the same batch shape as ``exit_wave``.
    """
    validate_exit_wave(exit_wave)
    return (exit_wave.real.square() + exit_wave.imag.square()).to(
        dtype=torch.float32 if exit_wave.dtype == torch.complex64 else torch.float64
    )
