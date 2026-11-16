"""Tests for the specimen tilt forward model."""

import numpy as np
import pytest
import torch
from torch_affine_utils.transforms_3d import Ry, Rz
from torch_transform_image import rotate_then_shift_image_3d

from torch_simulate_image import tilt_volume


@pytest.fixture
def volume() -> torch.Tensor:
    """An asymmetric specimen, so in-plane rotations are detectable."""
    n = 32
    c = (n - 1) / 2
    zz, yy, xx = torch.meshgrid(
        *[torch.arange(n, dtype=torch.float32) - c] * 3, indexing="ij"
    )
    vol = torch.zeros(n, n, n)
    for r, a, zc in [(9, 10, 1), (7, 100, -2), (5, 210, 2), (10, 300, 0)]:
        x0, y0 = r * np.cos(np.deg2rad(a)), r * np.sin(np.deg2rad(a))
        vol += torch.exp(
            -((xx - x0) ** 2 + (yy - y0) ** 2 + (zz - zc) ** 2) / (2 * 1.6**2)
        )
    return vol


def test_zero_tilt_is_identity(volume):
    out = tilt_volume(volume, tilt_deg=0.0)
    assert torch.allclose(out, volume, atol=1e-5)


def test_tilt_matches_y_rotation(volume):
    """With no detector rotation this is a plain rotation about y."""
    out = tilt_volume(volume, tilt_deg=30.0)
    expected = rotate_then_shift_image_3d(volume, rotate_zyx=(0.0, 30.0, 0.0))
    assert torch.allclose(out, expected, atol=1e-5)


def test_detector_rotation_at_zero_tilt_is_in_plane_rotation(volume):
    """At zero tilt, the detector rotation is a pure in-plane rotation."""
    out = tilt_volume(volume, tilt_deg=0.0, detector_rotation_deg=20.0)
    expected = rotate_then_shift_image_3d(volume, rotate_zyx=(20.0, 0.0, 0.0))
    assert torch.allclose(out, expected, atol=1e-5)


def test_composition_is_rz_ry():
    """The matrix really is Rz(detector) @ Ry(tilt), not a conjugation."""
    for tilt in (-45.0, 12.0, 60.0):
        for det in (-30.0, 0.0, 5.0):
            expected = (Rz(det) @ Ry(tilt))[:3, :3]
            # recover the applied rotation by tilting a basis-vector delta volume
            n = 16
            c = n // 2
            got = []
            for axis in range(3):
                vol = torch.zeros(n, n, n)
                # a point 4 voxels along one axis, in zyx indexing
                idx = [c, c, c]
                idx[2 - axis] += 4
                vol[tuple(idx)] = 1.0
                out = tilt_volume(vol, tilt_deg=tilt, detector_rotation_deg=det)
                flat = out.flatten().argmax().item()
                z, y, x = np.unravel_index(flat, (n, n, n))
                got.append([x - c, y - c, z - c])
            got = np.array(got).T / 4.0
            assert np.abs(got - expected.numpy()).max() < 0.35


def test_fill_value_pads_with_solvent(volume):
    """Out-of-bounds samples become `fill_value`, not zero."""
    ice = 3.6
    out = tilt_volume(volume + ice, tilt_deg=60.0, fill_value=ice)
    # the corners are pulled from outside the box at high tilt
    corners = torch.tensor([out[0, 0, 0], out[0, 0, -1], out[0, -1, 0], out[-1, 0, 0]])
    assert torch.allclose(corners, torch.full_like(corners, ice), atol=1e-4)
    # and with no fill value they are zero instead
    out0 = tilt_volume(volume + ice, tilt_deg=60.0)
    assert out0[0, 0, 0].abs() < 1e-4


def test_tilt_axis_stays_on_y(volume):
    """Points on y are unmoved by the tilt itself (detector rotation aside)."""
    n = 32
    c = n // 2
    vol = torch.zeros(n, n, n)
    vol[c, c + 8, c] = 1.0  # a point on +y
    for tilt in (-60.0, -20.0, 25.0, 60.0):
        out = tilt_volume(vol, tilt_deg=tilt)
        z, y, x = np.unravel_index(out.flatten().argmax().item(), (n, n, n))
        assert (z, y, x) == (c, c + 8, c)


def test_multichannel_input(volume):
    """`affine_transform_image_3d` treats a leading dim as channels, and moves
    them last -- (c, d, h, w) -> (d, h, w, c). Documented, not preferred: pass
    one volume at a time."""
    stack = torch.stack([volume, volume * 2])
    out = tilt_volume(stack, tilt_deg=15.0)
    assert out.shape == (*volume.shape, 2)
    assert torch.allclose(out[..., 1], out[..., 0] * 2, atol=1e-4)
