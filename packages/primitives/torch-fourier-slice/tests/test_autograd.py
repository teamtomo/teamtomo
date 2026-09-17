"""Tests for differentiability of, and absence of side effects in, slice extraction."""

import pytest
import torch

from torch_fourier_slice import project_3d_to_2d
from torch_fourier_slice.slice_extraction import (
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multichannel,
)


def _rfft_volume(n: int = 16, n_channels: int | None = None) -> torch.Tensor:
    """A volume in the fftshifted rfft layout the extraction functions expect."""
    shape = (n, n, n) if n_channels is None else (n_channels, n, n, n)
    volume = torch.rand(shape)
    dft = torch.fft.rfftn(
        torch.fft.ifftshift(volume, dim=(-3, -2, -1)), dim=(-3, -2, -1)
    )
    return torch.fft.fftshift(dft, dim=(-3, -2))


@pytest.mark.parametrize(
    "extract_fn, n_channels",
    [
        (extract_central_slices_rfft_3d, None),
        (extract_central_slices_rfft_3d_multichannel, 2),
    ],
)
def test_extraction_does_not_modify_rotation_matrices(extract_fn, n_channels):
    """Extraction must not write into the caller's rotation matrices.

    `rot_tolerance` snaps near-zero matrix elements to zero. Doing that in place
    silently zeroed elements of the caller's own array, because the preceding
    `.to(torch.float32)` is a no-op that returns the same tensor when the input
    is already float32.
    """
    rotation_matrices = torch.eye(3).expand(4, 3, 3).contiguous()
    # a legitimately tiny, non-zero element, below the default rot_tolerance
    rotation_matrices[0, 0, 1] = 1e-12
    expected = rotation_matrices.clone()

    extract_fn(
        volume_rfft=_rfft_volume(n_channels=n_channels),
        rotation_matrices=rotation_matrices,
    )

    assert torch.equal(rotation_matrices, expected)


@pytest.mark.parametrize(
    "extract_fn, n_channels",
    [
        (extract_central_slices_rfft_3d, None),
        (extract_central_slices_rfft_3d_multichannel, 2),
    ],
)
def test_extraction_accepts_rotation_matrices_requiring_grad(extract_fn, n_channels):
    """Rotation matrices may be optimised, so they may be autograd leaves.

    The in-place `rot_tolerance` write raised
    "a leaf Variable that requires grad is being used in an in-place operation".
    """
    rotation_matrices = torch.eye(3).expand(4, 3, 3).contiguous().clone()
    rotation_matrices.requires_grad_(True)

    slices = extract_fn(
        volume_rfft=_rfft_volume(n_channels=n_channels),
        rotation_matrices=rotation_matrices,
    )
    slices.abs().sum().backward()

    assert rotation_matrices.grad is not None
    assert torch.all(torch.isfinite(rotation_matrices.grad))


def test_project_3d_to_2d_is_differentiable_wrt_rotation_matrices():
    """Gradients reach the rotation matrices through the whole projection."""
    volume = torch.rand(16, 16, 16)
    rotation_matrices = torch.eye(3).expand(3, 3, 3).contiguous().clone()
    rotation_matrices.requires_grad_(True)

    project_3d_to_2d(volume, rotation_matrices).sum().backward()

    grad = rotation_matrices.grad
    assert grad is not None
    assert torch.all(torch.isfinite(grad))
    assert torch.any(grad != 0)


def test_project_3d_to_2d_is_differentiable_wrt_volume():
    """Gradients reach the volume, so it can be optimised by reprojection."""
    volume = torch.rand(16, 16, 16, requires_grad=True)
    rotation_matrices = torch.eye(3).expand(3, 3, 3).contiguous()

    project_3d_to_2d(volume, rotation_matrices).sum().backward()

    assert volume.grad is not None
    assert torch.all(torch.isfinite(volume.grad))
    assert torch.any(volume.grad != 0)
