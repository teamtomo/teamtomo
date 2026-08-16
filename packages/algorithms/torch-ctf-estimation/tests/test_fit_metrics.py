"""Tests for fit correlation helpers."""

import math

import pytest
import torch

from torch_ctf_estimation.metrics.fit_metrics import (
    l2_normalized_cross_correlation,
    pearson_r_flat,
)


def test_l2_ncc_proportional_vectors_is_one():
    y = torch.tensor([1.0, 2.0, 3.0])
    m = 2.0 * y
    cc = l2_normalized_cross_correlation(y, m)
    assert cc == pytest.approx(1.0)


def test_l2_ncc_orthogonal_is_zero():
    y = torch.tensor([1.0, 0.0, 0.0])
    m = torch.tensor([0.0, 1.0, 0.0])
    cc = l2_normalized_cross_correlation(y, m)
    assert cc == pytest.approx(0.0)


def test_pearson_r_flat_perfect_linear():
    y = torch.randn(100)
    m = 3.0 * y + 5.0
    r = pearson_r_flat(y, m)
    assert r == pytest.approx(1.0, abs=1e-5)


def test_pearson_r_flat_uncorrelated_noise():
    torch.manual_seed(0)
    y = torch.randn(5000)
    m = torch.randn(5000)
    r = pearson_r_flat(y, m)
    assert abs(r) < 0.05


def test_l2_ncc_zero_norm_returns_nan():
    y = torch.zeros(3)
    m = torch.tensor([1.0, 0.0, 0.0])
    assert math.isnan(l2_normalized_cross_correlation(y, m))
