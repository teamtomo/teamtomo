"""Fit tilt-series geometry to tracked 2D positions of 3D points."""

from dataclasses import dataclass

import torch
from torch_tilt_series import TiltSeries

from torch_tiltalign.triangulate_points import affine_projection, triangulate_points

# Tukey biweight cutoff, in units of the robust residual scale
_TUKEY_C = 4.685
# median of the norm of a 2D isotropic Gaussian, in units of its sigma
_RAYLEIGH_MEDIAN = 1.1774


@dataclass
class TiltAlignResult:
    """Result of `fit_tilt_series`.

    Attributes
    ----------
    tilt_series : TiltSeries
        A new `TiltSeries` holding the fitted geometry.
    points_zyx : torch.Tensor
        `(n_points, 3)` fitted 3D positions, zyx, in Angstroms relative to
        the tomogram center.
    residuals_yx : torch.Tensor
        `(n_points, n_tilts, 2)` projected minus tracked positions, in
        Angstroms; NaN where the point was not observed.
    weights : torch.Tensor
        `(n_points, n_tilts)` final weight of each observation: 0 where not
        observed or rejected by the robust fit.
    """

    tilt_series: TiltSeries
    points_zyx: torch.Tensor
    residuals_yx: torch.Tensor
    weights: torch.Tensor

    @property
    def rms_residual(self) -> float:
        """Root-mean-square residual (Angstroms) over observations with weight > 0."""
        used = self.weights > 0
        r = self.residuals_yx[used]
        return float(torch.sqrt((r**2).sum(-1).mean())) if len(r) else float("nan")


def fit_tilt_series(
    tilt_series: TiltSeries,
    tracks_yx: torch.Tensor,
    observed: torch.Tensor | None = None,
    points_zyx: torch.Tensor | None = None,
    refine_shifts: bool = True,
    refine_tilt_axis_angle: bool = False,
    refine_tilt_angles: bool = False,
    robust: bool = True,
    n_robust_iterations: int = 5,
    n_weight_groups: int = 5,
    min_observations: int = 3,
    max_iterations: int = 500,
) -> TiltAlignResult:
    """Fit a tilt series' alignment to tracked 2D positions of 3D points.

    The equivalent of IMOD's tiltalign: find the 3D position of every point
    and the per-tilt alignment parameters minimizing the distance between
    `tilt_series.project_points(points)` and the tracked positions. The points
    can be gold fiducials, patch centers, or anything else tracked through the
    series.

    With only shifts refined (the default) the problem is linear least
    squares. Refining the tilt-axis angle or the tilt angles makes it
    non-linear; all cases are solved with L-BFGS from the starting geometry.

    Gauge: a rigid translation of the tomogram (moving every point and
    compensating with the per-tilt shifts) does not change the projections.
    It is fixed so that the fitted shifts stay as close as possible to the
    starting shifts, i.e. the tomogram stays where the starting geometry put
    it. When tilt angles are refined, the angle of the tilt closest to 0 is
    held fixed, since a common offset of all tilt angles is equivalent to
    rotating the points.

    Parameters
    ----------
    tilt_series : TiltSeries
        Starting geometry; not modified.
    tracks_yx : torch.Tensor
        `(n_points, n_tilts, 2)` tracked detector positions, yx, in Angstroms
        relative to the detector center (the convention of
        `TiltSeries.project_points`). Tilts are in the order of
        `tilt_series.tilt_angles`.
    observed : torch.Tensor | None
        `(n_points, n_tilts)` boolean, True where the point was tracked.
        Defaults to every finite entry of `tracks_yx`.
    points_zyx : torch.Tensor | None
        `(n_points, 3)` starting 3D positions (Angstroms, relative to the
        tomogram center). Defaults to `triangulate_points` on the starting
        geometry.
    refine_shifts : bool
        Refine the per-tilt shifts (`sample_translations`).
    refine_tilt_axis_angle : bool
        Refine one in-plane rotation common to all tilts, added to
        `tilt_series.tilt_axis_angle`.
    refine_tilt_angles : bool
        Refine each tilt angle, except the one closest to 0.
    robust : bool
        Down-weight outlying observations with a Tukey biweight on the
        residual length, re-estimated over `n_robust_iterations` fits.
    n_robust_iterations : int
        Number of reweighted fits when `robust` is True.
    n_weight_groups : int
        The residual scale for the robust weights is estimated separately in
        this many groups of consecutive tilts (by tilt angle), as in IMOD's
        tiltalign, since residuals grow with tilt.
    min_observations : int
        Points tracked in fewer tilts than this are ignored.
    max_iterations : int
        Maximum L-BFGS iterations per fit.

    Returns
    -------
    TiltAlignResult
        The fitted geometry as a new `TiltSeries`, the 3D points, residuals
        and weights.
    """
    device = tilt_series.device
    tracks = torch.as_tensor(tracks_yx, dtype=torch.float32, device=device)
    finite = torch.isfinite(tracks).all(dim=-1)
    if observed is None:
        observed = finite
    observed = torch.as_tensor(observed, device=device).bool() & finite
    observed = observed & (observed.sum(dim=1, keepdim=True) >= min_observations)
    tracks = torch.where(observed[..., None], tracks, 0.0)
    if not observed.any():
        raise ValueError("no point is observed in at least `min_observations` tilts")

    ts = _copy_tilt_series(tilt_series)
    tilt_angles0 = ts.tilt_angles.clone()
    axis0 = ts.tilt_axis_angle.clone()
    shifts0 = ts.sample_translations.clone()

    if points_zyx is None:
        points_zyx = triangulate_points(ts, tracks, observed)
    points = torch.as_tensor(points_zyx, dtype=torch.float32, device=device)
    points = points.detach().clone(memory_format=torch.contiguous_format)
    points.requires_grad_(True)
    d_shifts = torch.zeros_like(shifts0, requires_grad=refine_shifts)
    d_axis = torch.zeros((), device=device, requires_grad=refine_tilt_axis_angle)
    d_tilt = torch.zeros_like(tilt_angles0, requires_grad=refine_tilt_angles)
    tilt_free = torch.ones_like(tilt_angles0)
    tilt_free[torch.argmin(tilt_angles0.abs())] = 0.0
    params = [points] + [p for p in (d_shifts, d_axis, d_tilt) if p.requires_grad]

    def set_geometry() -> None:
        ts.sample_translations = shifts0 + d_shifts
        ts.tilt_axis_angle = axis0 + d_axis
        ts.tilt_angles = tilt_angles0 + d_tilt * tilt_free

    groups = _tilt_groups(ts.tilt_angles, n_weight_groups)
    weights = observed.float()
    n_fits = n_robust_iterations if robust else 1
    for _ in range(n_fits):
        norm = weights.sum().clamp(min=1.0)
        optimizer = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=max_iterations,
            tolerance_grad=1e-10,
            tolerance_change=1e-14,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure(w: torch.Tensor = weights, n: torch.Tensor = norm) -> torch.Tensor:
            optimizer.zero_grad()
            set_geometry()
            residuals = ts.project_points(points) - tracks
            loss = (w[..., None] * residuals**2).sum() / n
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        optimizer.step(closure)  # type: ignore[no-untyped-call]
        if robust:
            with torch.no_grad():
                set_geometry()
                lengths = torch.linalg.norm(ts.project_points(points) - tracks, dim=-1)
                weights = _tukey_weights(lengths, observed, groups)

    with torch.no_grad():
        set_geometry()
        if refine_shifts:
            _fix_translation_gauge(ts, points, shifts0)
        residuals = ts.project_points(points) - tracks
        residuals = torch.where(observed[..., None], residuals, torch.nan)
    ts.sample_translations = ts.sample_translations.detach()
    ts.tilt_axis_angle = ts.tilt_axis_angle.detach()
    ts.tilt_angles = ts.tilt_angles.detach()
    return TiltAlignResult(
        tilt_series=ts,
        points_zyx=points.detach(),
        residuals_yx=residuals.detach(),
        weights=weights.detach(),
    )


def _tilt_groups(tilt_angles: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Group index of each tilt: `n_groups` runs of consecutive tilt angles."""
    n = len(tilt_angles)
    n_groups = max(1, min(n_groups, n))
    rank = torch.empty(n, dtype=torch.long, device=tilt_angles.device)
    rank[torch.argsort(tilt_angles)] = torch.arange(n, device=tilt_angles.device)
    return rank * n_groups // n


def _tukey_weights(
    lengths: torch.Tensor, observed: torch.Tensor, groups: torch.Tensor
) -> torch.Tensor:
    """Tukey biweight of residual lengths, scaled by their robust spread.

    The spread is estimated per group of tilts from the median residual length.
    """
    cutoff = torch.ones(lengths.shape[1], device=lengths.device)
    all_scale = torch.median(lengths[observed]) / _RAYLEIGH_MEDIAN
    for g in torch.unique(groups):
        in_group = observed[:, groups == g]
        group_lengths = lengths[:, groups == g][in_group]
        scale = (
            torch.median(group_lengths) / _RAYLEIGH_MEDIAN
            if len(group_lengths) >= 10
            else all_scale
        )
        cutoff[groups == g] = _TUKEY_C * scale.clamp(min=1e-6)
    u = (lengths / cutoff).clamp(max=1.0)
    return (1 - u**2) ** 2 * observed.float()


def _fix_translation_gauge(
    ts: TiltSeries, points: torch.Tensor, shifts0: torch.Tensor
) -> None:
    """Rigidly translate the tomogram so the shifts stay near `shifts0`.

    Moving every point by `t` and every shift by `-A_i t` leaves all
    projections unchanged; `t` is chosen to minimize the change in shifts.
    """
    matrices, _ = affine_projection(ts)  # (n_tilts, 2, 3)
    change = ts.sample_translations - shifts0  # (n_tilts, 2)
    a = matrices.reshape(-1, 3)
    b = change.reshape(-1, 1)
    t = torch.linalg.lstsq(a.cpu().double(), b.cpu().double()).solution
    t = t.to(points.dtype).to(points.device)[:, 0]
    points += t
    ts.sample_translations = ts.sample_translations - matrices @ t


def _copy_tilt_series(tilt_series: TiltSeries) -> TiltSeries:
    """A new TiltSeries with cloned, detached geometry tensors."""
    return TiltSeries(
        tilt_angles=tilt_series.tilt_angles.detach().clone(),
        tilt_axis_angle=tilt_series.tilt_axis_angle.detach().clone(),
        sample_translations=tilt_series.sample_translations.detach().clone(),
        x_tilts=tilt_series.x_tilts.detach().clone(),
        sample2levelled=tilt_series.sample2levelled.detach().clone(),
        levelled2tomo=tilt_series.levelled2tomo.detach().clone(),
        local_shifts=tilt_series.local_shifts,
        local_shifts_2d=tilt_series.local_shifts_2d,
        image_path=tilt_series.image_path,
        image_indices=tilt_series.image_indices,
        pixel_spacing=tilt_series._pixel_spacing,
        device=tilt_series.device,
    )
