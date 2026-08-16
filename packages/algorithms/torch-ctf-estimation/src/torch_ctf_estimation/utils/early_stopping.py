"""Plateau-style early stopping for Adam optimisation loops."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def make_early_stopper(
    patience: int = 5,
    window_size: int = 3,
    tolerance: float = 1e-5,
) -> Callable[[float], bool]:
    """Return a stateful callable that signals when optimisation should stop.

    Early stopping is triggered (returns True) when an average of the loss
    history (of ``window_size``) has a relative change (in absolute terms)
    less than ``tolerance`` for ``patience`` consecutive checks. Otherwise,
    continue optimisation (returns False).
    """
    loss_history: list[float] = []
    wait = 0

    def update(loss: float) -> bool:
        nonlocal loss_history, wait

        loss_history.append(loss)
        if len(loss_history) < window_size:
            return False

        smoothed_this = sum(loss_history[-window_size:]) / window_size
        smoothed_prev = sum(loss_history[-window_size - 1 : -1]) / window_size
        relative_diff = (smoothed_this - smoothed_prev) / (abs(smoothed_prev) + 1e-12)

        if abs(relative_diff) > tolerance:
            wait = 0
        else:
            wait += 1

        return wait >= patience

    return update
