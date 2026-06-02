"""Tests for gradient-based refinement."""

import torch

from torch_align_volumes import GradientRefinementConfig, gradient_refine
from torch_align_volumes._gradient import (
    _axis_angle_to_rotation_matrix_xyz,
    _rotation_matrix_xyz_to_axis_angle,
    _flip_3x3,
)


def test_axis_angle_identity():
    """Zero axis-angle vector should give identity rotation."""
    v = torch.zeros(3)
    R = _axis_angle_to_rotation_matrix_xyz(v)
    assert torch.allclose(R, torch.eye(3), atol=1e-5)


def test_axis_angle_roundtrip():
    """Axis-angle → matrix → axis-angle should approximately recover the original."""
    v_orig = torch.tensor([0.3, -0.2, 0.5])
    R = _axis_angle_to_rotation_matrix_xyz(v_orig)
    v_rec = _rotation_matrix_xyz_to_axis_angle(R)
    # Signs may differ (axis and -axis with ±theta are equivalent)
    assert torch.allclose(R, _axis_angle_to_rotation_matrix_xyz(v_rec), atol=1e-5)


def test_flip_3x3_roundtrip():
    """Flipping twice should return the original matrix."""
    M = torch.rand(3, 3)
    assert torch.allclose(_flip_3x3(_flip_3x3(M)), M)


def test_gradient_refine_identity():
    """Refining a perfect alignment (identity init) should yield positive NCC."""
    ref = torch.rand(20, 20, 20)
    R_init = torch.eye(3)
    t_init = torch.zeros(3)
    cfg = GradientRefinementConfig(n_iterations=5, loss="ncc", optimizer="adam")
    result = gradient_refine(ref, ref, R_init, t_init, config=cfg, verbose=False)
    # Score is NCC ∈ [-1, 1]; self-alignment at identity init should be positive
    assert result.score > 0.0
    assert result.rotation_matrix.shape == (3, 3)


def test_gradient_refine_lbfgs():
    """Refining with L-BFGS should also yield positive NCC."""
    ref = torch.rand(20, 20, 20)
    R_init = torch.eye(3)
    t_init = torch.zeros(3)
    cfg = GradientRefinementConfig(n_iterations=5, loss="ncc", optimizer="lbfgs")
    result = gradient_refine(ref, ref, R_init, t_init, config=cfg, verbose=False)
    assert result.score > 0.0
    assert result.rotation_matrix.shape == (3, 3)
