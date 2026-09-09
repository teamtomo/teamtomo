"""Poisson shot-noise sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torch_simulate_image._validate import validate_real_image

if TYPE_CHECKING:
    from torch_simulate_image.config import PoissonConfig


def poisson_sample(
    expected_counts: torch.Tensor,
    config: PoissonConfig,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample electron counts from a Poisson distribution.

    Parameters
    ----------
    expected_counts : torch.Tensor
        Non-negative expected counts with shape ``(..., H, W)``.
    config : PoissonConfig
        Sampling options. When ``config.apply`` is ``False``, returns
        ``expected_counts`` unchanged.
    generator : torch.Generator or None
        Optional RNG, takes precedence over ``config``. When ``None`` and
        ``config.deterministic`` is ``True``, a generator is seeded from
        ``config.seed`` on ``expected_counts`` device. Otherwise sampling
        draws fresh randomness from the global RNG on every call.

    Returns
    -------
    torch.Tensor
        Sampled counts (float dtype) or expected counts when sampling is off.
    """
    validate_real_image(expected_counts)
    if not config.apply:
        return expected_counts

    if generator is None and config.deterministic:
        assert config.seed is not None  # enforced by PoissonConfig validator
        generator = torch.Generator(device=expected_counts.device)
        generator.manual_seed(config.seed)

    safe_counts = expected_counts.clamp(min=0.0)
    return torch.poisson(safe_counts, generator=generator)
