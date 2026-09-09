"""Pipeline orchestration for micrograph simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch_simulate_image._validate import validate_exit_wave, validate_real_image
from torch_simulate_image.detector.dqe import apply_dqe
from torch_simulate_image.dose import apply_dose_weight
from torch_simulate_image.envelopes import apply_envelopes
from torch_simulate_image.fluence import scale_to_expected_counts
from torch_simulate_image.intensity import exit_wave_to_intensity
from torch_simulate_image.noise.poisson import poisson_sample
from torch_simulate_image.optics.aperture import apply_objective_aperture
from torch_simulate_image.optics.ctf import apply_ctf_to_exit_wave

if TYPE_CHECKING:
    import torch

    from torch_simulate_image.config import MicrographSimulationConfig


def _intensity_to_micrograph(
    intensity: torch.Tensor,
    config: MicrographSimulationConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Shared intensity → counts path (envelopes, dose, fluence, Poisson, DQE)."""
    image = intensity

    if config.envelope.apply:
        image = apply_envelopes(
            image,
            config.envelope,
            pixel_size=config.pixel_size,
            fluence=config.fluence,
            ctf=config.ctf,
        )

    image = apply_dose_weight(
        image,
        config.dose_weight,
        pixel_size=config.pixel_size,
        voltage_kv=config.ctf.voltage_kv,
    )

    expected = scale_to_expected_counts(
        image,
        config.fluence,
        pixel_size=config.pixel_size,
    )

    if config.dqe.apply and config.dqe.apply_before_noise:
        expected = apply_dqe(expected, config.dqe)

    if config.return_expected_counts or not config.poisson.apply:
        counts = expected
    else:
        counts = poisson_sample(expected, config.poisson, generator=generator)

    if config.dqe.apply and not config.dqe.apply_before_noise:
        counts = apply_dqe(counts, config.dqe)

    return counts


def simulate_micrograph(
    exit_wave: torch.Tensor,
    config: MicrographSimulationConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Form a 2D micrograph from a complex exit wave.

    Parameters
    ----------
    exit_wave : torch.Tensor
        Complex tensor with shape ``(..., H, W)``.
    config : MicrographSimulationConfig
        Optics, fluence, noise, and detector options.
    generator : torch.Generator or None
        Optional RNG passed to Poisson sampling.

    Returns
    -------
    torch.Tensor
        Micrograph counts (float) with shape ``(..., H, W)``.
    """
    validate_exit_wave(exit_wave)
    wave = apply_objective_aperture(
        exit_wave,
        config.objective_aperture,
        pixel_size=config.pixel_size,
        voltage_kv=config.ctf.voltage_kv,
    )
    wave = apply_ctf_to_exit_wave(
        wave,
        config.ctf,
        pixel_size=config.pixel_size,
    )
    intensity = exit_wave_to_intensity(wave)
    return _intensity_to_micrograph(intensity, config, generator=generator)


def simulate_micrograph_from_intensity(
    intensity: torch.Tensor,
    config: MicrographSimulationConfig,
    *,
    skip_ctf: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Form a micrograph when intensity is already available.

    Parameters
    ----------
    intensity : torch.Tensor
        Real intensity with shape ``(..., H, W)``.
    config : MicrographSimulationConfig
        Simulation options. CTF is skipped by default.
    skip_ctf : bool
        When ``True`` (default), CTF settings are ignored.
    generator : torch.Generator or None
        Optional RNG for Poisson sampling.

    Returns
    -------
    torch.Tensor
        Simulated micrograph counts.
    """
    validate_real_image(intensity)

    if not skip_ctf and config.ctf.apply:
        msg = (
            "CTF cannot be applied from intensity alone; provide an exit wave or "
            "keep skip_ctf=True."
        )
        raise ValueError(msg)

    return _intensity_to_micrograph(intensity, config, generator=generator)
