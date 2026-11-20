"""Tests for projection-based alignment."""

import math

import numpy as np
import torch
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

from torch_fit_in_map import ProjectionAlignmentConfig, projection_align


def _asymmetric_volume(d: int) -> torch.Tensor:
    """Gaussian blobs of unequal size and weight with no rotational symmetry."""
    centre = (d - 1) / 2.0
    grid = np.stack(np.meshgrid(*[np.arange(d)] * 3, indexing="ij"), axis=-1)
    blobs = [  # (zyx offset from centre, sigma, weight)
        ((0.0, 0.0, 0.0), 3.0, 1.0),
        ((6.0, 0.0, 0.0), 2.0, 0.8),
        ((0.0, 8.0, 0.0), 1.5, 0.6),
        ((-2.0, -3.0, 6.0), 2.5, 0.4),
    ]
    vol = np.zeros((d, d, d))
    for offset, sigma, weight in blobs:
        r2 = ((grid - (centre + np.array(offset))) ** 2).sum(axis=-1)
        vol += weight * np.exp(-r2 / (2 * sigma**2))
    return torch.from_numpy(vol).float()


def _rotation_angle_degrees(a: torch.Tensor, b: torch.Tensor) -> float:
    """Geodesic distance between two rotation matrices."""
    cos = ((a.T @ b).trace().item() - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def test_projection_align_identity():
    """Aligning a cubic volume with itself should give score > 0."""
    vol = torch.rand(24, 24, 24)
    cfg = ProjectionAlignmentConfig(angular_step_degrees=30.0)
    result = projection_align(vol, vol, config=cfg)
    assert result.score > 0.0
    assert result.rotation_matrix.shape == (3, 3)
    assert result.translation_pixels.shape == (3,)
    # Self-alignment needs no shift
    assert torch.equal(result.translation_pixels, torch.zeros(3))


def test_projection_align_recovers_known_rotation():
    """The returned rotation is the one that maps the mobile onto the reference.

    The mobile is the reference actively rotated by ``Q`` (zyx) about the box
    centre and then shifted, built with scipy so the test does not share
    conventions with the code under test.  ``AlignmentResult`` defines
    ``rotation_matrix`` as the pull matrix that brings the mobile back onto the
    reference, which for this mobile is ``Q`` itself.
    """
    d = 32
    step = 20.0
    reference = _asymmetric_volume(d)

    q = Rotation.from_rotvec(np.deg2rad(70.0) * np.array([1.0, 2.0, -3.0]) / 14**0.5)
    q_zyx = q.as_matrix()
    t_zyx = np.array([1.0, 2.0, -3.0])
    centre = np.full(3, (d - 1) / 2.0)
    # mobile[o] = reference[Q^T (o - c) + c + t]: reference rotated by Q, and
    # translated so that rotating back then shifting by t restores it.
    mobile = affine_transform(
        reference.numpy(),
        q_zyx.T,
        offset=centre + t_zyx - q_zyx.T @ centre,
        order=3,
    )
    mobile = torch.from_numpy(mobile).float()

    cfg = ProjectionAlignmentConfig(angular_step_degrees=step)
    result = projection_align(reference, mobile, config=cfg)

    q_torch = torch.from_numpy(q_zyx).float()
    error = _rotation_angle_degrees(result.rotation_matrix, q_torch)
    assert error < step, f"rotation is {error:.1f} deg from Q"
    torch.testing.assert_close(
        result.translation_pixels,
        torch.tensor(t_zyx, dtype=torch.float32),
        atol=1.5,
        rtol=0.0,
    )
