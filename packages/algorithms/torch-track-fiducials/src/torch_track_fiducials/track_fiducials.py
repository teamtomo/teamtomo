"""Track gold fiducials through a tilt series."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_find_peaks import find_peaks_2d
from torch_subpixel_crop import subpixel_crop_2d
from torch_tilt_series import TiltSeries
from torch_tiltalign import TiltAlignResult, affine_projection, fit_tilt_series

from torch_track_fiducials.bead_correlation import bead_correlation_maps, sample_maps

# (largest |tilt angle| tracked, search radius in bead diameters) per round
DEFAULT_SCHEDULE: tuple[tuple[float, float], ...] = (
    (40.0, 2.0),
    (90.0, 1.5),
    (90.0, 1.0),
    (90.0, 0.6),
)


@dataclass
class FiducialTracks:
    """Result of `track_fiducials`.

    Attributes
    ----------
    tracks_yx : torch.Tensor
        `(n_fiducials, n_tilts, 2)` tracked detector positions, yx, in
        Angstroms relative to the detector center (the convention of
        `TiltSeries.project_points`); NaN where not tracked.
    observed : torch.Tensor
        `(n_fiducials, n_tilts)` boolean, True where tracked.
    scores : torch.Tensor
        `(n_fiducials, n_tilts)` bead-template correlation at the tracked
        position, in units of the correlation map's robust standard
        deviation; NaN where not tracked.
    alignment : TiltAlignResult
        `torch_tiltalign.fit_tilt_series` fitted to the final tracks.
    """

    tracks_yx: torch.Tensor
    observed: torch.Tensor
    scores: torch.Tensor
    alignment: TiltAlignResult


def track_fiducials(
    images: torch.Tensor,
    tilt_series: TiltSeries,
    fiducial_diameter: float,
    max_fiducials: int = 50,
    search_thickness: float = 4000.0,
    min_score: float = 4.0,
    min_fraction_tracked: float = 0.5,
    max_tilt_offset: float = 5.0,
    schedule: Sequence[tuple[float, float]] = DEFAULT_SCHEDULE,
    refine_tilt_axis_angle: bool = False,
    refine_tilt_angles: bool = False,
    robust: bool = True,
) -> FiducialTracks:
    """Find gold fiducials and track them through a tilt series.

    1. Correlate every tilt image with a dark disc of the fiducial's size.
    2. Seed: take the strongest correlation peaks in the tilt closest to 0
       and find each one's depth by scoring points along its ray, projected
       into the tilts within +-40 degrees with the starting geometry.
    3. Track and fit, over the rounds of `schedule`: project each fiducial
       with the current geometry, correct each tilt by the offset common to
       all its fiducials (the peak of the summed correlation around their
       predicted positions), take each fiducial's correlation peak within
       the search radius, then fit the geometry and 3D positions to
       the tracks with `torch_tiltalign.fit_tilt_series`. Rounds widen the
       tilt range and shrink the search radius.

    Observations are dropped where the peak is weak, where another
    fiducial is predicted within one diameter (overlapping beads) or where
    the fiducial is near the image edge. Fiducials tracked in fewer than
    `min_fraction_tracked` of the tilts, or mostly rejected by the robust fit,
    are dropped.

    Parameters
    ----------
    images : torch.Tensor
        `(n_tilts, h, w)` tilt images, raw contrast (gold dark), in the order
        of `tilt_series.tilt_angles`.
    tilt_series : TiltSeries
        Starting geometry, e.g. from `torch_tiltxcorr.tiltxcorr`.
        `tilt_series.pixel_spacing` must be the pixel spacing of `images`.
    fiducial_diameter : float
        Gold diameter in Angstroms.
    max_fiducials : int
        Maximum number of fiducials returned (the strongest are kept).
    search_thickness : float
        Depth range (Angstroms), centered on the tomogram center, searched
        for each seed.
    min_score : float
        Minimum correlation (in robust standard deviations) for a tracked
        position to count.
    min_fraction_tracked : float
        Minimum fraction of tilts a fiducial must be tracked in.
    max_tilt_offset : float
        Before tracking in each round, the offset common to all fiducials in
        each tilt is found within this many fiducial diameters of the
        prediction, so a tilt whose starting shift is wrong by more than the
        search radius is still tracked.
    schedule : Sequence[tuple[float, float]]
        Per round: largest absolute tilt angle tracked, and the search
        radius in fiducial diameters.
    refine_tilt_axis_angle : bool
        Passed to `fit_tilt_series`.
    refine_tilt_angles : bool
        Passed to `fit_tilt_series`.
    robust : bool
        Passed to `fit_tilt_series`.

    Returns
    -------
    FiducialTracks
        Tracks, scores and the fitted alignment.
    """
    n_tilts, h, w = images.shape
    device = images.device
    spacing = tilt_series.pixel_spacing
    diameter_px = fiducial_diameter / spacing
    center = torch.tensor([h // 2, w // 2], device=device).float()
    tilt_angles = tilt_series.tilt_angles.to(device)

    maps = bead_correlation_maps(images, diameter_px)

    # 1. seeds: strongest peaks in the tilt closest to 0, then their depth
    reference = int(torch.argmin(tilt_angles.abs()))
    coords, heights = find_peaks_2d(
        maps[reference],
        min_distance=max(1, round(diameter_px)),
        threshold_abs=min_score,
        exclude_border=max(1, round(1.5 * diameter_px)),
    )
    if len(coords) == 0:
        raise ValueError("no fiducial candidates found in the tilt closest to 0")
    keep = torch.argsort(heights, descending=True)[: 3 * max_fiducials]
    seeds_yx = (coords[keep].float() - center) * spacing
    points = _seed_depths(
        maps, tilt_series, seeds_yx, reference, diameter_px, search_thickness
    )

    # 2. track and fit
    geometry = tilt_series
    tracks = torch.full((len(points), n_tilts, 2), torch.nan, device=device)
    scores = torch.full((len(points), n_tilts), torch.nan, device=device)
    observed = torch.zeros(len(points), n_tilts, dtype=torch.bool, device=device)
    fit = None
    for max_tilt, radius in schedule:
        in_range = tilt_angles.abs() <= max_tilt + 1e-3
        predicted = geometry.project_points(points) / spacing + center
        inside = _inside(predicted, h, w, diameter_px)
        # common offset of all fiducials in each tilt: corrects tilts whose
        # current shift is off by more than the search radius
        offsets = _tilt_offsets(
            maps, predicted, inside, max(radius, max_tilt_offset) * diameter_px
        )
        predicted = predicted + offsets[None]
        measured, peak = _measure(maps, predicted, radius * diameter_px)
        ok = (
            in_range[None, :]
            & (peak >= min_score)
            & _inside(predicted, h, w, diameter_px)
            & ~_crowded(predicted, diameter_px)
        )
        median_peak = torch.nanmedian(torch.where(ok, peak, torch.nan), dim=1).values
        ok &= peak >= 0.3 * median_peak[:, None]
        tracks = torch.where(ok[..., None], (measured - center) * spacing, torch.nan)
        scores = torch.where(ok, peak, torch.nan)
        observed = ok
        enough = observed.sum(dim=1) >= max(3, round(0.25 * int(in_range.sum())))
        if enough.sum() < 3:
            raise ValueError("fewer than 3 fiducials could be tracked")
        tracks, scores, observed, points = (
            tracks[enough],
            scores[enough],
            observed[enough],
            points[enough],
        )
        fit = fit_tilt_series(
            geometry,
            tracks,
            observed,
            points_zyx=points,
            refine_tilt_axis_angle=refine_tilt_axis_angle,
            refine_tilt_angles=refine_tilt_angles,
            robust=robust,
        )
        geometry, points = fit.tilt_series, fit.points_zyx

    # 3. keep fiducials tracked in enough tilts and accepted by the fit
    assert fit is not None
    used = fit.weights > 0
    fraction = used.sum(dim=1) / n_tilts
    accepted = (fraction >= min_fraction_tracked) & (
        fit.weights.sum(dim=1) >= 0.5 * observed.sum(dim=1)
    )
    strength = torch.nanmean(scores, dim=1)
    strength[~accepted] = -torch.inf
    selected = torch.argsort(strength, descending=True)[
        : min(max_fiducials, int(accepted.sum()))
    ]
    if len(selected) < 3:
        raise ValueError("fewer than 3 fiducials were tracked reliably")
    tracks, scores, observed = tracks[selected], scores[selected], observed[selected]
    fit = fit_tilt_series(
        tilt_series,
        tracks,
        observed,
        points_zyx=points[selected],
        refine_tilt_axis_angle=refine_tilt_axis_angle,
        refine_tilt_angles=refine_tilt_angles,
        robust=robust,
    )
    return FiducialTracks(
        tracks_yx=tracks, observed=observed, scores=scores, alignment=fit
    )


def _seed_depths(
    maps: torch.Tensor,
    tilt_series: TiltSeries,
    seeds_yx: torch.Tensor,
    reference: int,
    diameter_px: float,
    search_thickness: float,
    max_tilt: float = 40.0,
) -> torch.Tensor:
    """3D position of each seed: the depth along its ray that scores best."""
    n_tilts, h, w = maps.shape
    device = maps.device
    spacing = tilt_series.pixel_spacing
    center = torch.tensor([h // 2, w // 2], device=device).float()
    # tolerate coarse-alignment errors of about one bead diameter
    k = max(1, math.ceil(diameter_px))
    pooled = F.max_pool2d(maps[:, None], 2 * k + 1, stride=1, padding=k)[:, 0]

    matrices, offsets = affine_projection(tilt_series)  # (t, 2, 3), (t, 2)
    step = max(spacing, diameter_px * spacing / 2)
    n_z = max(1, int(search_thickness / step) + 1)
    z = torch.linspace(-search_thickness / 2, search_thickness / 2, n_z, device=device)
    # points on each seed's ray in the reference tilt: solve for yx at each z
    m = matrices[reference]
    rhs = seeds_yx[:, None, :] - offsets[reference] - z[None, :, None] * m[:, 0]
    yx = torch.linalg.solve(m[:, 1:], rhs.reshape(-1, 2).T).T.reshape(-1, n_z, 2)
    zyx = torch.cat([z[None, :, None].expand(len(seeds_yx), n_z, 1), yx], dim=-1)

    projected = torch.einsum("tij,snj->snti", matrices, zyx) + offsets
    projected = projected / spacing + center  # (seeds, z, tilts, 2)
    near = tilt_series.tilt_angles.to(device).abs() <= max_tilt
    values = sample_maps(pooled, projected)  # (seeds, z, tilts)
    best = (values * near).sum(dim=-1).argmax(dim=-1)
    return zyx[torch.arange(len(seeds_yx), device=device), best]


def _tilt_offsets(
    maps: torch.Tensor, predicted: torch.Tensor, valid: torch.Tensor, window: float
) -> torch.Tensor:
    """Per-tilt offset that best aligns all predicted positions with beads.

    Sums the (positive) correlation around every valid predicted position
    and takes the peak. Returns `(t, 2)` yx offsets in pixels.
    """
    n, t, _ = predicted.shape
    half = math.ceil(window)
    size = 2 * half + 1
    rounded = torch.round(torch.nan_to_num(predicted))
    crops = subpixel_crop_2d(maps, rounded, sidelength=size)  # (n, t, s, s)
    total = (crops.clamp(min=0) * valid[..., None, None]).sum(dim=0)  # (t, s, s)
    index = total.reshape(t, -1).argmax(dim=-1)
    offsets = torch.stack([index // size, index % size], dim=-1).float() - half
    return torch.where(valid.any(dim=0)[:, None], offsets, 0.0)


def _measure(
    maps: torch.Tensor, predicted: torch.Tensor, radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correlation peak within `radius` of each predicted position.

    `predicted` is `(n, t, 2)` yx array coordinates. Returns measured
    positions `(n, t, 2)` and peak values `(n, t)`.
    """
    n, t, _ = predicted.shape
    half = math.ceil(radius) + 1
    size = 2 * half + 1
    rounded = torch.round(torch.nan_to_num(predicted))
    # crop pixel (size // 2, size // 2) is the rounded prediction
    crops = subpixel_crop_2d(maps, rounded, sidelength=size)
    offset = predicted - rounded  # (n, t, 2)
    grid = torch.arange(size, device=maps.device).float() - size // 2
    dy = grid[None, None, :, None] - offset[..., 0, None, None]
    dx = grid[None, None, None, :] - offset[..., 1, None, None]
    in_radius = dy**2 + dx**2 <= radius**2
    masked = torch.where(in_radius, crops, -torch.inf)
    flat = masked.reshape(n, t, -1)
    index = flat.argmax(dim=-1)
    py, px = index // size, index % size
    peak = flat.gather(-1, index[..., None])[..., 0]

    def vertex(minus: torch.Tensor, mid: torch.Tensor, plus: torch.Tensor):
        denominator = minus - 2 * mid + plus
        safe = torch.where(denominator.abs() > 1e-12, denominator, -1.0)
        return (0.5 * (minus - plus) / safe).clamp(-0.5, 0.5)

    cy, cx = py.clamp(1, size - 2), px.clamp(1, size - 2)

    def at(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return crops.reshape(n, t, -1).gather(-1, (y * size + x)[..., None])[..., 0]

    sub_y = vertex(at(cy - 1, cx), at(cy, cx), at(cy + 1, cx))
    sub_x = vertex(at(cy, cx - 1), at(cy, cx), at(cy, cx + 1))
    measured = rounded + torch.stack(
        [cy - size // 2 + sub_y, cx - size // 2 + sub_x], dim=-1
    )
    # a maximum on the edge of the search region means no bead was found
    on_edge = (py == 0) | (px == 0) | (py == size - 1) | (px == size - 1)
    distance = (measured - predicted).norm(dim=-1)
    peak = torch.where(on_edge | (distance > radius), -torch.inf, peak)
    return measured, peak


def _inside(
    predicted: torch.Tensor, h: int, w: int, diameter_px: float
) -> torch.Tensor:
    """True where the predicted position is at least one diameter from the edge."""
    margin = diameter_px
    y, x = predicted[..., 0], predicted[..., 1]
    return (y >= margin) & (y <= h - 1 - margin) & (x >= margin) & (x <= w - 1 - margin)


def _crowded(predicted: torch.Tensor, diameter_px: float) -> torch.Tensor:
    """True where another fiducial is predicted within one diameter."""
    p = predicted.permute(1, 0, 2)  # (t, n, 2)
    distance = torch.cdist(p, p)  # (t, n, n)
    n = p.shape[1]
    distance = distance + torch.eye(n, device=p.device) * 1e9
    return (distance.min(dim=-1).values < diameter_px).permute(1, 0)
