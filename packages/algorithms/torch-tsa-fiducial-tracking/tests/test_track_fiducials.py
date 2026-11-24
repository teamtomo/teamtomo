import pytest
import torch
from torch_image_interpolation import insert_into_image_2d
from torch_tilt_series import TiltSeries

from torch_tsa_fiducial_tracking import bead_correlation_maps, track_fiducials

SIZE = 256
SPACING = 10.0  # A per pixel
DIAMETER = 100.0  # A, i.e. 10 px
TILT_ANGLES = torch.arange(-60.0, 61.0, 3.0)
AXIS = 85.0


def render(geometry, points_zyx, amplitudes, sigma_px):
    projected = geometry.project_points(points_zyx) / SPACING
    center = torch.tensor([SIZE // 2, SIZE // 2]).float()
    images = []
    for t in range(projected.shape[1]):
        image, _ = insert_into_image_2d(
            amplitudes,
            coordinates=projected[:, t] + center,
            image=torch.zeros(SIZE, SIZE),
        )
        images.append(image)
    images = torch.stack(images)
    fy = torch.fft.fftfreq(SIZE)[:, None]
    fx = torch.fft.rfftfreq(SIZE)[None, :]
    blur = torch.exp(-2 * (torch.pi * sigma_px) ** 2 * (fy**2 + fx**2))
    return torch.fft.irfft2(torch.fft.rfft2(images) * blur, s=(SIZE, SIZE))


def make_tilt_series(n_beads=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(TILT_ANGLES)
    true_shifts = torch.randn(n, 2, generator=g) * 80  # A
    truth = TiltSeries(
        tilt_angles=TILT_ANGLES,
        tilt_axis_angle=torch.tensor(AXIS),
        sample_translations=true_shifts,
        pixel_spacing=SPACING,
    )
    beads = torch.empty(n_beads, 3)
    beads[:, 0] = (torch.rand(n_beads, generator=g) - 0.5) * 1200
    beads[:, 1:] = (torch.rand(n_beads, 2, generator=g) - 0.5) * 2000
    texture = torch.empty(20000, 3)
    texture[:, 0] = (torch.rand(20000, generator=g) - 0.5) * 1200
    texture[:, 1:] = (torch.rand(20000, 2, generator=g) - 0.5) * 2.4 * SIZE * SPACING
    images = render(truth, beads, -torch.ones(n_beads) * 40.0, sigma_px=3.0)
    images += render(truth, texture, torch.randn(20000, generator=g), sigma_px=1.5)
    images += 0.3 * torch.randn(images.shape, generator=g)
    coarse = TiltSeries(
        tilt_angles=TILT_ANGLES,
        tilt_axis_angle=torch.tensor(AXIS),
        sample_translations=true_shifts + torch.randn(n, 2, generator=g) * 20,
        pixel_spacing=SPACING,
    )
    return images, truth, coarse, beads


def rigid_free_rms(a, b):
    th = torch.deg2rad(TILT_ANGLES).double()
    basis = torch.stack([torch.ones_like(th), torch.cos(th), torch.sin(th)], -1)
    d = (a - b).double()
    coef = torch.linalg.lstsq(basis, d).solution
    return float(torch.sqrt(((d - basis @ coef) ** 2).mean()))


def test_bead_correlation_peaks_at_beads():
    images, truth, _, beads = make_tilt_series(n_beads=5)
    maps = bead_correlation_maps(images, DIAMETER / SPACING)
    projected = truth.project_points(beads) / SPACING + SIZE // 2  # (5, t, 2)
    reference = int(torch.argmin(TILT_ANGLES.abs()))
    for p in projected[:, reference]:
        y, x = torch.round(p).long().tolist()
        window = maps[reference, y - 3 : y + 4, x - 3 : x + 4]
        assert window.max() > 5
        assert window.max() == maps[reference, y - 6 : y + 7, x - 6 : x + 7].max()


def test_track_fiducials_recovers_alignment():
    images, truth, coarse, _ = make_tilt_series()
    result = track_fiducials(images, coarse, DIAMETER, search_thickness=2000.0)
    n_fiducials = result.tracks_yx.shape[0]
    assert n_fiducials >= 12  # some of the 25 synthetic beads overlap
    assert result.observed.float().mean() > 0.8
    assert torch.isfinite(result.scores[result.observed]).all()
    fitted = result.alignment.tilt_series.sample_translations
    start_error = rigid_free_rms(coarse.sample_translations, truth.sample_translations)
    error = rigid_free_rms(fitted, truth.sample_translations)
    assert start_error > 10.0
    assert error < 3.0  # A, i.e. 0.3 px
    assert result.alignment.rms_residual < 10.0  # 1 px; texture and noise


def test_track_fiducials_without_beads_raises():
    g = torch.Generator().manual_seed(0)
    images = torch.randn(len(TILT_ANGLES), 128, 128, generator=g) * 0.01
    geometry = TiltSeries(
        tilt_angles=TILT_ANGLES,
        tilt_axis_angle=torch.tensor(AXIS),
        sample_translations=torch.zeros(len(TILT_ANGLES), 2),
        pixel_spacing=SPACING,
    )
    with pytest.raises(ValueError):
        track_fiducials(images, geometry, DIAMETER, min_score=50.0)
