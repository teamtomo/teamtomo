import pytest
import torch
from torch_image_interpolation import insert_into_image_2d
from torch_tilt_series import TiltSeries

from torch_tiltxcorr import chop_tracks, tiltxcorr_patches

SIZE = 256
TILT_ANGLES = torch.arange(-60.0, 61.0, 3.0)
AXIS = 85.0


def render(tilt_series, points_zyx, size=SIZE, sigma=1.5):
    """Images of Gaussian blobs at the projected points, one per tilt."""
    projected = tilt_series.project_points(points_zyx)  # (n, tilts, 2), A == px
    center = torch.tensor([size // 2, size // 2]).float()
    images = []
    for t in range(projected.shape[1]):
        image, _ = insert_into_image_2d(
            torch.ones(len(points_zyx)),
            coordinates=projected[:, t] + center,
            image=torch.zeros(size, size),
        )
        images.append(image)
    images = torch.stack(images)
    freq_y = torch.fft.fftfreq(size)[:, None]
    freq_x = torch.fft.rfftfreq(size)[None, :]
    gaussian = torch.exp(-2 * (torch.pi * sigma) ** 2 * (freq_y**2 + freq_x**2))
    return torch.fft.irfft2(torch.fft.rfft2(images) * gaussian, s=(size, size))


def plane_geometry(seed=0):
    g = torch.Generator().manual_seed(seed)
    shifts = torch.randn(len(TILT_ANGLES), 2, generator=g) * 8
    geometry = TiltSeries(
        tilt_angles=TILT_ANGLES,
        tilt_axis_angle=torch.tensor(AXIS),
        sample_translations=shifts,
        pixel_spacing=1.0,
    )
    n = 6000
    points = torch.zeros(n, 3)
    points[:, 1:] = (torch.rand(n, 2, generator=g) - 0.5) * 2.2 * SIZE
    return geometry, points, shifts


def test_patches_on_a_plane_follow_projected_points():
    geometry, points, true_shifts = plane_geometry()
    images = render(geometry, points)
    # coarse alignment in tiltxcorr's convention, with a few pixels of error
    g = torch.Generator().manual_seed(1)
    coarse = -true_shifts + torch.randn(true_shifts.shape, generator=g) * 2
    tracks, observed = tiltxcorr_patches(
        images, TILT_ANGLES, AXIS, shifts=coarse, patch_size=64, preprocess=False
    )
    assert observed.shape == tracks.shape[:2]
    assert observed[:, torch.argmin(TILT_ANGLES.abs())].all()
    assert observed.float().mean() > 0.5

    # the patch center in the reference tilt is a point on the plane z = 0;
    # find it, then its projection in every tilt is where the patch should be
    reference = int(torch.argmin(TILT_ANGLES.abs()))
    m = geometry.projection_matrices[reference]  # (4, 4) zyxw
    a = m[1:3, 1:3]
    b = tracks[:, reference] - m[1:3, 3]
    yx = torch.linalg.solve(a, b.T).T
    on_plane = torch.cat([torch.zeros(len(yx), 1), yx], dim=-1)
    expected = geometry.project_points(on_plane)
    error = (tracks - expected).norm(dim=-1)[observed]
    assert error.median() < 0.2
    assert error.max() < 1.0


def test_patches_leaving_the_image_are_unobserved():
    geometry, points, true_shifts = plane_geometry(seed=2)
    images = render(geometry, points)
    tracks, observed = tiltxcorr_patches(
        images, TILT_ANGLES, AXIS, shifts=-true_shifts, patch_size=64, preprocess=False
    )
    assert torch.isnan(tracks[~observed]).all()
    assert torch.isfinite(tracks[observed]).all()
    half = SIZE / 2 - 32
    assert tracks[observed].abs().max() <= half + 1e-4


def test_chop_tracks():
    n_tilts = len(TILT_ANGLES)
    tracks = torch.randn(2, n_tilts, 2)
    observed = torch.ones(2, n_tilts, dtype=torch.bool)
    observed[1, :5] = False
    tracks[1, :5] = torch.nan
    pieces, pieces_observed = chop_tracks(
        tracks, observed, TILT_ANGLES, segment_length=16, segment_overlap=4
    )
    counts = pieces_observed.sum(dim=1)
    assert torch.all(counts <= 16)
    assert torch.all(counts >= 4)
    # every observed entry is kept, and kept values are unchanged
    for k in range(2):
        covered = torch.zeros(n_tilts, dtype=torch.bool)
        for piece, seen in zip(pieces, pieces_observed):
            same = (piece[seen] == tracks[k][seen]).all()
            if same and seen.any():
                covered |= seen
        assert torch.equal(covered, observed[k])
    assert torch.isnan(pieces[~pieces_observed]).all()


def test_chop_tracks_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chop_tracks(
            torch.zeros(1, 3, 2),
            torch.ones(1, 3, dtype=torch.bool),
            torch.arange(3.0),
            segment_length=4,
            segment_overlap=4,
        )
