"""Tests for projection-based alignment."""

import torch

from torch_fit_in_map import ProjectionAlignmentConfig, projection_align


def test_projection_align_identity():
    """Aligning a cubic volume with itself should give score > 0."""
    vol = torch.rand(24, 24, 24)
    cfg = ProjectionAlignmentConfig(angular_step_degrees=30.0)
    result = projection_align(vol, vol, config=cfg)
    assert result.score > 0.0
    assert result.rotation_matrix.shape == (3, 3)
    assert result.translation_pixels.shape == (3,)
    # Depth shift should be zero for projection-based method
    assert result.translation_pixels[0].item() == 0.0
