"""Compare projections and reconstructions against a real-space ground truth.

Round trips and the position of a point's peak still pass when the DFT is sampled
half a voxel off, so these compare against ``scipy.ndimage.affine_transform`` of an
asymmetric volume about voxel ``n // 2`` instead. Both even and odd box sizes are
checked: odd sizes put DC at ``n // 2 = (n - 1) / 2`` in a fftshifted DFT, which the
even case can't distinguish from ``n / 2``.
"""

import numpy as np
import pytest
import torch
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

from torch_fourier_slice import (
    backproject_2d_to_3d,
    backproject_2d_to_3d_multichannel,
    project_2d_to_1d,
    project_3d_to_2d,
    project_3d_to_2d_multichannel,
)

SIDELENGTHS = [32, 33]


def _gaussian_blobs(
    sidelength: int, ndim: int, n_blobs: int, seed: int = 0
) -> np.ndarray:
    """Randomly placed Gaussian blobs, an image or volume with no symmetry."""
    rng = np.random.default_rng(seed)
    center = sidelength // 2
    grid = np.meshgrid(*[np.arange(sidelength) - center] * ndim, indexing="ij")
    out = np.zeros((sidelength,) * ndim, dtype=np.float32)
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
    center = np.full(array.ndim, array.shape[0] // 2)
    return affine_transform(array, matrix, offset=center - matrix @ center, order=1)


def _project(array: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Real-space projection along z (or y) with the rotation the package takes.

    The projection functions take the passive rotation (image frame -> volume
    frame), so the projection with `R` is of the volume rotated by `R.T`.
    """
    return _rotate(array, rotation_matrix.T).sum(axis=0)


def _cc(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(array), dtype=torch.float32)


def _rotations_3d() -> np.ndarray:
    """The identity, which needs no interpolation in real space, and some others."""
    random = Rotation.random(4, random_state=1).as_matrix()
    return np.concatenate([np.eye(3)[None], random])


@pytest.mark.parametrize("pad_factor, min_cc", [(2.0, 0.998), (1.0, 0.99)])
@pytest.mark.parametrize("sidelength", SIDELENGTHS)
def test_project_3d_to_2d_matches_real_space(sidelength, pad_factor, min_cc):
    volume = _gaussian_blobs(sidelength, ndim=3, n_blobs=12)
    rotations = _rotations_3d()
    projections = project_3d_to_2d(
        _tensor(volume), _tensor(rotations), pad_factor=pad_factor
    )
    for projection, rotation in zip(projections, rotations, strict=True):
        assert _cc(projection, _project(volume, rotation)) > min_cc


@pytest.mark.parametrize("sidelength", SIDELENGTHS)
def test_project_3d_to_2d_multichannel_matches_real_space(sidelength):
    volume = _gaussian_blobs(sidelength, ndim=3, n_blobs=12)
    channels = np.stack([volume, volume[::-1]])
    rotations = _rotations_3d()
    projections = project_3d_to_2d_multichannel(_tensor(channels), _tensor(rotations))
    for projection, rotation in zip(projections, rotations, strict=True):
        for channel, channel_volume in zip(projection, channels, strict=True):
            assert _cc(channel, _project(channel_volume, rotation)) > 0.998


@pytest.mark.parametrize("sidelength", SIDELENGTHS)
def test_project_3d_to_2d_transform_matrix_matches_real_space(sidelength):
    volume = _gaussian_blobs(sidelength, ndim=3, n_blobs=12)
    transform_matrix = np.array([[1.1, 0.0], [0.0, 0.95]])
    projection = project_3d_to_2d(
        _tensor(volume), torch.eye(3), transform_matrix=_tensor(transform_matrix)
    )
    center = np.full(2, sidelength // 2)
    expected = affine_transform(
        volume.sum(axis=0),
        transform_matrix,
        offset=center - transform_matrix @ center,
        order=1,
    )
    assert _cc(projection, expected) > 0.998


@pytest.mark.parametrize("pad_factor, min_cc", [(2.0, 0.998), (1.0, 0.99)])
@pytest.mark.parametrize("sidelength", SIDELENGTHS)
def test_project_2d_to_1d_matches_real_space(sidelength, pad_factor, min_cc):
    image = _gaussian_blobs(sidelength, ndim=2, n_blobs=8)
    angles = (0, 25, 70, 130, 200, 290)
    rotations = np.stack(
        [Rotation.from_euler("z", a, degrees=True).as_matrix()[:2, :2] for a in angles]
    )
    lines = project_2d_to_1d(_tensor(image), _tensor(rotations), pad_factor=pad_factor)
    for line, rotation in zip(lines, rotations, strict=True):
        assert _cc(line, _project(image, rotation)) > min_cc


@pytest.mark.parametrize("pad_factor, min_cc", [(2.0, 0.995), (1.0, 0.97)])
@pytest.mark.parametrize("multichannel", [False, True])
@pytest.mark.parametrize("sidelength", SIDELENGTHS)
def test_backproject_2d_to_3d_matches_real_space(
    sidelength, multichannel, pad_factor, min_cc
):
    # images of the volume made in real space
    volume = _gaussian_blobs(sidelength, ndim=3, n_blobs=12)
    rotations = Rotation.random(200, random_state=2).as_matrix()
    images = _tensor(np.stack([_project(volume, r) for r in rotations]))
    if multichannel:
        images = torch.stack([images, images], dim=-3)
        reconstruction = backproject_2d_to_3d_multichannel(
            images, _tensor(rotations), pad_factor=pad_factor
        )[1]
    else:
        reconstruction = backproject_2d_to_3d(
            images, _tensor(rotations), pad_factor=pad_factor
        )
    assert _cc(reconstruction, volume) > min_cc


def _float32_rotation(axis: str, degrees: float) -> np.ndarray:
    """Rotation about x, y or z with its trig done in float32, as is common in torch.

    In float32, cos(90°) is -4.4e-8 rather than 0, which is above the extractor's
    default `rot_tolerance` of 1e-8, so it isn't zeroed.
    """
    theta = torch.deg2rad(torch.tensor(degrees, dtype=torch.float32))
    c, s = torch.cos(theta).item(), torch.sin(theta).item()
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    matrix = np.eye(3, dtype=np.float32)
    matrix[i, i], matrix[i, j], matrix[j, i], matrix[j, j] = c, -s, s, c
    return matrix


RIGHT_ANGLE_ROTATIONS = [
    *[[(axis, degrees)] for axis in "xyz" for degrees in (90, -90, 180, 270)],
    [("x", 90), ("y", 90)],
    [("y", 90), ("z", 90)],
]


def _rotation_id(rotations: list[tuple[str, float]]) -> str:
    return "@".join(f"R{axis}({degrees})" for axis, degrees in rotations)


@pytest.mark.parametrize("multichannel", [False, True])
@pytest.mark.parametrize("rotations", RIGHT_ANGLE_ROTATIONS, ids=_rotation_id)
def test_project_3d_to_2d_float32_right_angles_match_real_space(
    rotations, multichannel
):
    # e.g. a view along x leaves every point of the slice ~1e-8 below the x=0 plane
    # of the rfft, too close to flip into the non-redundant half, and samples taken
    # outside the array are zero
    rotation = np.eye(3, dtype=np.float32)
    for axis, degrees in rotations:
        rotation = rotation @ _float32_rotation(axis, degrees)
    volume = _gaussian_blobs(32, ndim=3, n_blobs=12)
    if multichannel:
        channels = np.stack([volume, volume[::-1]])
        projection = project_3d_to_2d_multichannel(
            _tensor(channels), _tensor(rotation)
        )[0]
    else:
        projection = project_3d_to_2d(_tensor(volume), _tensor(rotation))
    assert _cc(projection, _project(volume, rotation.astype(np.float64))) > 0.998
