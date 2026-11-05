"""Pin the direction of rotation of every layer against a real-space ground truth.

Real-space functions take an *active* rotation: the projection with ``R`` is of the
volume rotated by ``R``. Fourier-space functions take the *passive* rotation (image
frame -> volume frame), so the slice extracted with ``R`` is of the volume rotated
by ``R.T``. See teamtomo/teamtomo#145.

Round-trip tests pass under either convention, so these compare against
``scipy.ndimage.affine_transform`` on an asymmetric volume instead. Projecting with
the wrong convention gives a correlation of ~0.6 here, the right one ~1.0.
"""

import numpy as np
import pytest
import torch
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

from torch_fourier_slice import (
    backproject_2d_to_3d,
    backproject_2d_to_3d_multichannel,
    experimental,
    extract_central_slices_rfft_3d,
    project_2d_to_1d,
    project_3d_to_2d,
    project_3d_to_2d_multichannel,
)

SIDELENGTH = 32
CENTER = SIDELENGTH // 2

requires_mojo = pytest.mark.skipif(
    not experimental.mojo_kernels_available(),
    reason="mojo package not installed / kernels failed to compile",
)


def _gaussian_blobs(ndim: int, n_blobs: int, seed: int = 0) -> np.ndarray:
    """Randomly placed Gaussian blobs, an image or volume with no symmetry."""
    rng = np.random.default_rng(seed)
    grid = np.meshgrid(*[np.arange(SIDELENGTH) - CENTER] * ndim, indexing="ij")
    out = np.zeros((SIDELENGTH,) * ndim, dtype=np.float32)
    for _ in range(n_blobs):
        position = rng.uniform(-7, 7, ndim)
        d2 = sum((g - p) ** 2 for g, p in zip(grid, position, strict=True))
        out += np.exp(-d2 / (2 * 1.5**2))
    return out


def _rotate(array: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Actively rotate an image/volume about its center (DC position in the DFT).

    `rotation_matrix` acts on xy(z) column vectors; arrays are indexed (z)yx, and
    `affine_transform` pulls samples, hence the reversal and the transpose.
    """
    matrix = rotation_matrix[::-1, ::-1].T
    center = np.full(array.ndim, CENTER)
    return affine_transform(array, matrix, offset=center - matrix @ center, order=1)


def _cc(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(array), dtype=torch.float32)


def _xyz_to_zyx(rotation_matrices: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rotation_matrices[..., ::-1, ::-1])


@pytest.fixture
def volume() -> np.ndarray:
    return _gaussian_blobs(ndim=3, n_blobs=12)


@pytest.fixture
def rotations() -> np.ndarray:
    return Rotation.random(8, random_state=1).as_matrix()


@pytest.mark.parametrize("zyx_matrices", [False, True])
def test_project_3d_to_2d_rotates_volume_actively(volume, rotations, zyx_matrices):
    matrices = _xyz_to_zyx(rotations) if zyx_matrices else rotations
    projections = project_3d_to_2d(
        _tensor(volume), _tensor(matrices), zyx_matrices=zyx_matrices
    )
    for projection, rotation in zip(projections, rotations, strict=True):
        assert _cc(projection, _rotate(volume, rotation).sum(axis=0)) > 0.99


def test_project_3d_to_2d_multichannel_rotates_volume_actively(volume, rotations):
    channels = np.stack([volume, volume[::-1]])
    projections = project_3d_to_2d_multichannel(_tensor(channels), _tensor(rotations))
    for projection, rotation in zip(projections, rotations, strict=True):
        for channel, channel_volume in zip(projection, channels, strict=True):
            expected = _rotate(channel_volume, rotation).sum(axis=0)
            assert _cc(channel, expected) > 0.99


def test_project_2d_to_1d_rotates_image_actively():
    image = _gaussian_blobs(ndim=2, n_blobs=8)
    angles = (25, 70, 130, 200)
    rotations = np.stack(
        [Rotation.from_euler("z", a, degrees=True).as_matrix()[:2, :2] for a in angles]
    )
    lines = project_2d_to_1d(_tensor(image), _tensor(rotations))
    for line, rotation in zip(lines, rotations, strict=True):
        assert _cc(line, _rotate(image, rotation).sum(axis=0)) > 0.99


@pytest.mark.parametrize("multichannel", [False, True])
def test_backproject_2d_to_3d_takes_the_rotation_of_the_volume(volume, multichannel):
    # images of the volume actively rotated by each matrix, made in real space
    rotations = Rotation.random(50, random_state=2).as_matrix()
    images = _tensor(np.stack([_rotate(volume, r).sum(axis=0) for r in rotations]))
    if multichannel:
        reconstruction = backproject_2d_to_3d_multichannel(
            images[:, None], _tensor(rotations)
        )[0]
    else:
        reconstruction = backproject_2d_to_3d(images, _tensor(rotations))
    assert _cc(reconstruction, volume) > 0.98


def test_extract_central_slices_rfft_3d_takes_the_passive_rotation(volume, rotations):
    dft = torch.fft.rfftn(torch.fft.ifftshift(_tensor(volume)), dim=(-3, -2, -1))
    dft = torch.fft.fftshift(dft, dim=(-3, -2))
    slices = extract_central_slices_rfft_3d(dft, _tensor(rotations))
    slices = torch.fft.ifftshift(slices, dim=(-2,))
    images = torch.fft.irfftn(slices, dim=(-2, -1), s=(SIDELENGTH, SIDELENGTH))
    images = torch.fft.fftshift(images, dim=(-2, -1))
    for image, rotation in zip(images, rotations, strict=True):
        assert _cc(image, _rotate(volume, rotation.T).sum(axis=0)) > 0.98


@requires_mojo
def test_experimental_project_3d_to_2d_rotates_volume_actively(volume, rotations):
    projections = experimental.project_3d_to_2d(
        _tensor(volume), _tensor(_xyz_to_zyx(rotations))
    )
    for projection, rotation in zip(projections, rotations, strict=True):
        assert _cc(projection, _rotate(volume, rotation).sum(axis=0)) > 0.99


@requires_mojo
def test_experimental_project_3d_to_2d_matches_torch_backend(volume, rotations):
    mojo = experimental.project_3d_to_2d(
        _tensor(volume), _tensor(_xyz_to_zyx(rotations))
    )
    torch_backend = project_3d_to_2d(_tensor(volume), _tensor(rotations))
    for a, b in zip(mojo, torch_backend, strict=True):
        assert _cc(a, b) > 0.9999


@requires_mojo
def test_experimental_backproject_2d_to_3d_takes_the_rotation_of_the_volume(volume):
    rotations = Rotation.random(50, random_state=2).as_matrix()
    images = _tensor(np.stack([_rotate(volume, r).sum(axis=0) for r in rotations]))
    reconstruction = experimental.backproject_2d_to_3d(
        images, _tensor(_xyz_to_zyx(rotations))
    )
    assert _cc(reconstruction, volume) > 0.98


@requires_mojo
def test_experimental_extract_takes_the_passive_rotation(volume, rotations):
    dft = torch.fft.rfftn(torch.fft.fftshift(_tensor(volume)), dim=(-3, -2, -1))
    slices = experimental.extract_central_slices_rfft_3d(
        dft.contiguous(), _tensor(_xyz_to_zyx(rotations))
    )
    images = torch.fft.irfftn(slices, dim=(-2, -1), s=(SIDELENGTH, SIDELENGTH))
    images = torch.fft.fftshift(images, dim=(-2, -1))
    for image, rotation in zip(images, rotations, strict=True):
        assert _cc(image, _rotate(volume, rotation.T).sum(axis=0)) > 0.98
