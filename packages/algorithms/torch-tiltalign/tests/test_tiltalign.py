import pytest
import torch
from torch_tilt_series import TiltSeries

from torch_tiltalign import fit_tilt_series, triangulate_points

TILT_ANGLES = torch.arange(-60.0, 61.0, 3.0)
AXIS = 175.6


def make_geometry(shifts=None, axis=AXIS, tilt_angles=TILT_ANGLES):
    n = len(tilt_angles)
    if shifts is None:
        shifts = torch.zeros(n, 2)
    return TiltSeries(
        tilt_angles=tilt_angles,
        tilt_axis_angle=torch.tensor(axis),
        sample_translations=shifts,
        pixel_spacing=10.0,
    )


def make_problem(n_points=30, noise=2.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(TILT_ANGLES)
    points = torch.empty(n_points, 3)
    points[:, 0] = (torch.rand(n_points, generator=g) - 0.5) * 1200  # z
    points[:, 1:] = (torch.rand(n_points, 2, generator=g) - 0.5) * 3000  # yx
    true_shifts = torch.randn(n, 2, generator=g) * 50
    truth = make_geometry(true_shifts)
    clean = truth.project_points(points)
    tracks = clean + noise * torch.randn(clean.shape, generator=g)
    observed = torch.rand(n_points, n, generator=g) > 0.1
    return truth, points, clean, tracks, observed


def rigid_free_rms(a, b, tilt_angles):
    """RMS of a - b per component after removing c + x0 cos + z0 sin."""
    th = torch.deg2rad(tilt_angles).double()
    basis = torch.stack([torch.ones_like(th), torch.cos(th), torch.sin(th)], -1)
    d = (a - b).double()
    coef = torch.linalg.lstsq(basis, d).solution
    return float(torch.sqrt(((d - basis @ coef) ** 2).mean()))


def test_triangulate_points_exact():
    truth, points, clean, _, observed = make_problem(noise=0.0)
    estimated = triangulate_points(truth, clean, observed)
    assert torch.allclose(estimated, points, atol=0.05)


def test_fit_shifts_recovers_projections():
    truth, _, clean, tracks, observed = make_problem()
    result = fit_tilt_series(make_geometry(), tracks, observed)
    fitted = result.tilt_series.project_points(result.points_zyx)
    rms = torch.sqrt(((fitted - clean) ** 2).sum(-1).mean())
    assert rms < 1.5  # below the 2 A per-axis tracking noise
    shift_rms = rigid_free_rms(
        result.tilt_series.sample_translations, truth.sample_translations, TILT_ANGLES
    )
    assert shift_rms < 1.0
    assert 1.5 < result.rms_residual < 3.5


def test_fit_keeps_starting_tomogram_position():
    truth, _, _, tracks, observed = make_problem(noise=0.0)
    result = fit_tilt_series(truth, tracks, observed)
    assert torch.allclose(
        result.tilt_series.sample_translations, truth.sample_translations, atol=0.05
    )


def test_fit_does_not_modify_input():
    _, _, _, tracks, observed = make_problem()
    start = make_geometry()
    fit_tilt_series(start, tracks, observed, refine_tilt_axis_angle=True)
    assert torch.all(start.sample_translations == 0)
    assert float(start.tilt_axis_angle) == pytest.approx(AXIS)


def test_robust_fit_rejects_outliers():
    truth, _, clean, tracks, observed = make_problem(n_points=40, seed=1)
    g = torch.Generator().manual_seed(2)
    outliers = (torch.rand(observed.shape, generator=g) < 0.03) & observed
    tracks = tracks.clone()
    tracks[outliers] += 200.0
    result = fit_tilt_series(make_geometry(), tracks, observed)
    assert torch.all(result.weights[outliers] == 0)
    fitted = result.tilt_series.project_points(result.points_zyx)
    good = observed & ~outliers
    rms = torch.sqrt(((fitted - clean)[good] ** 2).sum(-1).mean())
    assert rms < 1.5
    plain = fit_tilt_series(make_geometry(), tracks, observed, robust=False)
    fitted_plain = plain.tilt_series.project_points(plain.points_zyx)
    assert torch.sqrt(((fitted_plain - clean)[good] ** 2).sum(-1).mean()) > 2 * rms


def test_refine_tilt_axis_angle():
    g = torch.Generator().manual_seed(3)
    points = torch.empty(30, 3)
    points[:, 0] = (torch.rand(30, generator=g) - 0.5) * 1200
    points[:, 1:] = (torch.rand(30, 2, generator=g) - 0.5) * 3000
    truth = make_geometry(torch.randn(len(TILT_ANGLES), 2, generator=g) * 50, 176.0)
    tracks = truth.project_points(points)
    result = fit_tilt_series(make_geometry(), tracks, refine_tilt_axis_angle=True)
    assert float(result.tilt_series.tilt_axis_angle) == pytest.approx(176.0, abs=0.02)
    assert result.rms_residual < 0.1


def test_refine_tilt_angles():
    g = torch.Generator().manual_seed(4)
    points = torch.empty(40, 3)
    points[:, 0] = (torch.rand(40, generator=g) - 0.5) * 1500
    points[:, 1:] = (torch.rand(40, 2, generator=g) - 0.5) * 3000
    true_angles = TILT_ANGLES + 0.3 * torch.randn(len(TILT_ANGLES), generator=g)
    true_angles[torch.argmin(TILT_ANGLES.abs())] = 0.0
    truth = make_geometry(tilt_angles=true_angles)
    tracks = truth.project_points(points)
    result = fit_tilt_series(make_geometry(), tracks, refine_tilt_angles=True)
    error = (result.tilt_series.tilt_angles - true_angles).abs().max()
    assert error < 0.02
    assert result.rms_residual < 0.1


def test_unobserved_and_sparse_points():
    truth, _, clean, tracks, observed = make_problem(n_points=10)
    observed = observed.clone()
    observed[0] = False
    observed[1, 2:] = False  # only two observations: below min_observations
    tracks = tracks.clone()
    tracks[~observed] = torch.nan
    result = fit_tilt_series(make_geometry(), tracks)
    assert torch.isnan(result.residuals_yx[0]).all()
    assert torch.all(result.weights[:2] == 0)
    fitted = result.tilt_series.project_points(result.points_zyx)
    used = result.weights > 0
    assert torch.sqrt(((fitted - clean)[used] ** 2).sum(-1).mean()) < 2.0
