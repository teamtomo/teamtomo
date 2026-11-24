"""Poisson shot-noise sampling."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import torch

from torch_simulate_image._validate import validate_real_image

if TYPE_CHECKING:
    from torch_simulate_image.config import PoissonConfig


def poisson_sample(
    expected_counts: torch.Tensor,
    config: PoissonConfig,
    *,
    draw_index: int | None = None,
) -> torch.Tensor:
    """Sample electron counts from a Poisson distribution.

    Parameters
    ----------
    expected_counts : torch.Tensor
        Non-negative expected counts with shape ``(..., H, W)``.
    config : PoissonConfig
        Sampling options. When ``config.apply`` is ``False``, returns
        ``expected_counts`` unchanged.
    draw_index : int or None
        Which draw this is (e.g. the frame or tilt number). Required when
        ``config.seed`` is set: the same seed and index always give the same
        noise. Ignored otherwise.

    Returns
    -------
    torch.Tensor
        Sampled counts (float dtype) or expected counts when sampling is off.

    Notes
    -----
    Every call uses a freshly seeded generator. Do not reuse one generator
    across calls: on CUDA, ``torch.poisson`` advances the Philox offset too
    little per call, so consecutive draws can be correlated.
    """
    validate_real_image(expected_counts)
    if not config.apply:
        return expected_counts

    generator = torch.Generator(device=expected_counts.device)
    generator.manual_seed(_resolve_seed(config.seed, draw_index))
    return torch.poisson(expected_counts.clamp(min=0.0), generator=generator)


def _resolve_seed(seed: int | None, draw_index: int | None) -> int:
    """Return a 64-bit seed: random if ``seed`` is None, else hash(seed, draw_index).

    Hashed rather than added, so that seed 0's draw 1 is not seed 1's draw 0.
    """
    if seed is None:
        return int(torch.randint(2**62, ()))  # respects torch.manual_seed

    if draw_index is None:
        msg = (
            "poisson.seed is set, so pass a draw_index (e.g. the frame or tilt "
            "number) to give each draw its own noise."
        )
        raise ValueError(msg)

    digest = hashlib.blake2b(f"{seed}:{draw_index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")
