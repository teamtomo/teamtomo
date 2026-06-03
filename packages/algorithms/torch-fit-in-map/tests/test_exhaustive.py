"""Tests for the exhaustive SO(3) search."""

import torch
import pytest

from torch_fit_in_map import ExhaustiveSearchConfig, exhaustive_search
from torch_fit_in_map._exhaustive import (
    _euler_zyz_to_4x4_zyx,
    _argmax_to_shift,
    _parse_symmetry,
)


def test_align_identity_no_transform():
    """Aligning a volume with itself should recover near-identity rotation and zero shift."""
    ref = torch.rand(24, 24, 24)
    result = exhaustive_search(
        ref, ref, config=ExhaustiveSearchConfig(angular_step_degrees=15.0), verbose=False
    )
    assert result.rotation_matrix.shape == (3, 3)
    assert result.translation_pixels.shape == (3,)
    # Self-alignment: rotation should be close to identity
    assert torch.allclose(result.rotation_matrix, torch.eye(3), atol=0.1)
    # Translation should be within one voxel
    assert result.translation_pixels.abs().max().item() < 2.0


def test_euler_zyz_identity():
    """ZYZ (0, 0, 0) should give identity matrix."""
    angles = torch.tensor([[0.0, 0.0, 0.0]])
    M = _euler_zyz_to_4x4_zyx(angles)
    assert M.shape == (1, 4, 4)
    expected = torch.eye(4).unsqueeze(0)
    assert torch.allclose(M, expected, atol=1e-5)


def test_argmax_to_shift_positive():
    """Index below half-size should give a positive shift."""
    shape = (16, 16, 16)
    t = _argmax_to_shift(torch.tensor(2 * 16 * 16 + 3 * 16 + 4), shape)
    assert t[0] == pytest.approx(2.0)
    assert t[1] == pytest.approx(3.0)
    assert t[2] == pytest.approx(4.0)


def test_argmax_to_shift_wrapped():
    """Index above half-size should give a negative (wrapped) shift."""
    shape = (16, 16, 16)
    # Index 14 in a size-16 dim → 14 - 16 = -2
    flat = 14 * 16 * 16
    t = _argmax_to_shift(torch.tensor(flat), shape)
    assert t[0] == pytest.approx(-2.0)


def test_config_pixel_size_angstroms():
    """pixel_size_angstroms should populate translation_angstroms."""
    ref = torch.rand(20, 20, 20)
    cfg = ExhaustiveSearchConfig(angular_step_degrees=20.0, pixel_size_angstroms=1.5)
    result = exhaustive_search(ref, ref, config=cfg)
    assert result.translation_angstroms is not None
    assert result.translation_angstroms.shape == (3,)


def test_parse_symmetry_cyclic():
    assert _parse_symmetry("C1") == ("C", 1)
    assert _parse_symmetry("C4") == ("C", 4)
    assert _parse_symmetry("c6") == ("C", 6)


def test_parse_symmetry_dihedral():
    assert _parse_symmetry("D2") == ("D", 2)


def test_parse_symmetry_special():
    assert _parse_symmetry("T") == ("T", 1)
    assert _parse_symmetry("O") == ("O", 1)
    assert _parse_symmetry("I") == ("I", 1)


def test_parse_symmetry_invalid():
    with pytest.raises(ValueError):
        _parse_symmetry("X3")
    with pytest.raises(ValueError):
        _parse_symmetry("C0")


def test_symmetry_reduces_search_space():
    """C4 symmetry should sample fewer orientations than C1."""
    from torch_so3 import get_symmetry_ranges, get_uniform_euler_angles

    step = 15.0
    r_c1 = get_symmetry_ranges("C", 1)
    r_c4 = get_symmetry_ranges("C", 4)
    n_c1 = get_uniform_euler_angles(psi_step=step, theta_step=step, **r_c1._asdict()).shape[0]
    n_c4 = get_uniform_euler_angles(psi_step=step, theta_step=step, **r_c4._asdict()).shape[0]
    assert n_c4 < n_c1


def test_exhaustive_search_with_symmetry():
    """Search with C1 symmetry should recover near-identity rotation for self-alignment."""
    ref = torch.rand(20, 20, 20)
    cfg = ExhaustiveSearchConfig(angular_step_degrees=20.0, symmetry="C1")
    result = exhaustive_search(ref, ref, config=cfg, verbose=False)
    assert torch.allclose(result.rotation_matrix, torch.eye(3), atol=0.15)


def test_exhaustive_topk_returns_k_results():
    """_exhaustive_topk with n_start=3 should return 3 results sorted best-first."""
    from torch_fit_in_map._exhaustive import _exhaustive_topk

    ref = torch.rand(20, 20, 20)
    cfg = ExhaustiveSearchConfig(angular_step_degrees=30.0, n_start=3)
    results = _exhaustive_topk(ref, ref, config=cfg, mask=None, verbose=False)
    assert len(results) == 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_align_volumes_multistart():
    """align_volumes with n_start=3 should return the best NCC-scored refined result."""
    from torch_fit_in_map import align_volumes, GradientRefinementConfig

    ref = torch.rand(20, 20, 20)
    cfg = ExhaustiveSearchConfig(angular_step_degrees=30.0, n_start=3)
    result = align_volumes(
        ref, ref,
        exhaustive_config=cfg,
        gradient_config=GradientRefinementConfig(n_iterations=5),
        verbose=False,
    )
    # After gradient refinement, score is NCC ∈ [-1, 1]; self-alignment should be positive
    assert result.score > 0.0
