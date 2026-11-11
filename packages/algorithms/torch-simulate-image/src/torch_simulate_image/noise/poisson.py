"""Poisson shot-noise sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torch_simulate_image._validate import validate_real_image

if TYPE_CHECKING:
    from torch_simulate_image.config import PoissonConfig

_MASK64 = (1 << 64) - 1


def poisson_sample(
    expected_counts: torch.Tensor,
    config: PoissonConfig,
    *,
    draw_index: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample electron counts from a Poisson distribution.

    Each call samples with its own freshly seeded generator. Reusing one
    generator is not enough for independent calls: on CUDA, PyTorch advances
    the generator too little per ``torch.poisson`` call, so consecutive draws
    correlate at some count levels (~0.6 at 48 counts/pixel in torch 2.9).

    Parameters
    ----------
    expected_counts : torch.Tensor
        Non-negative expected counts with shape ``(..., H, W)``.
    config : PoissonConfig
        Sampling options. When ``config.apply`` is ``False``, returns
        ``expected_counts`` unchanged. When ``config.seed`` is set, the
        generator is seeded from a hash of ``(config.seed, draw_index)``;
        otherwise from torch's global RNG.
    draw_index : int or None
        Which draw this is (e.g. the frame or tilt number). Required when
        ``config.seed`` is set: calls with the same seed and index return
        identical noise. Ignored otherwise.
    generator : torch.Generator or None
        Optional RNG, takes precedence over ``config`` and ``draw_index``. On
        CUDA, reusing one generator across calls gives correlated draws (see
        above).

    Returns
    -------
    torch.Tensor
        Sampled counts (float dtype) or expected counts when sampling is off.
    """
    validate_real_image(expected_counts)
    if not config.apply:
        return expected_counts

    if generator is None:
        if config.seed is None:
            seed = int(torch.randint(2**62, ()))
        elif draw_index is None:
            msg = (
                "poisson.seed is set, so pass a draw_index (e.g. the frame or "
                "tilt number) to give each draw its own noise."
            )
            raise ValueError(msg)
        else:
            seed = _draw_seed(config.seed, draw_index)
        generator = torch.Generator(device=expected_counts.device)
        generator.manual_seed(seed)

    safe_counts = expected_counts.clamp(min=0.0)
    return torch.poisson(safe_counts, generator=generator)


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _draw_seed(seed: int, draw_index: int) -> int:
    """Hash a base seed and a draw index into one 64-bit seed.

    Hashed rather than added, so that seed 0's draw 1 is not seed 1's draw 0.
    """
    return _splitmix64(_splitmix64(seed & _MASK64) ^ (draw_index & _MASK64))
