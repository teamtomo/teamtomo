"""Cross-correlation helpers for CTF fit reliability (final scalars)."""

from __future__ import annotations

import torch


def l2_normalized_cross_correlation(
    y: torch.Tensor,
    m: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    """
    Cosine similarity between two 1D vectors: (y @ m) / (||y|| ||m||).

    Parameters
    ----------
    y, m
        Same shape, real-valued (e.g. fit-band background-subtracted power and
        CTF^2 times envelope).
    eps
        Floor for norms to avoid division by zero.

    Returns
    -------
    float
        Typically in [-1, 1] for unconstrained data; often [0, 1] for nonnegative
        spectra.
    """
    y_flat = y.reshape(-1)
    m_flat = m.reshape(-1)
    ny = torch.linalg.norm(y_flat)
    nm = torch.linalg.norm(m_flat)
    if ny < eps or nm < eps:
        return float("nan")
    return float((torch.dot(y_flat, m_flat) / (ny * nm)).detach().cpu().item())


def pearson_r_flat(
    y_flat: torch.Tensor,
    m_flat: torch.Tensor,
) -> float:
    """
    Pearson correlation coefficient r between two flattened tensors of equal length.

    Uses ``torch.corrcoef`` on stacked [y, m]. Returns NaN if undefined
    (e.g. zero variance).
    """
    y_flat = y_flat.reshape(-1).float()
    m_flat = m_flat.reshape(-1).float()
    if y_flat.numel() < 2:
        return float("nan")
    stacked = torch.stack([y_flat, m_flat], dim=0)
    c = torch.corrcoef(stacked)
    return float(c[0, 1].detach().cpu().item())
