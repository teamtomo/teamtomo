"""Marker-free tilt-series alignment by projection matching."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_tilt_series import TiltSeries, preprocess_tilt_series_images

from torch_projection_matching.backprojection import (
    affine_projection,
    backproject,
    splat,
    voxel_positions,
)


@dataclass
class ProjectionMatchingResult:
    """Result of `align_by_projection_matching`.

    Attributes
    ----------
    tilt_series : TiltSeries
        A new `TiltSeries` with the refined per-tilt shifts.
    slab : tuple[float, float]
        `(z_center, thickness)` in Angstroms of the slab reconstructed for
        matching.
    shift_updates : torch.Tensor
        `(n_iterations, n_tilts, 2)` yx shift applied at each iteration, in
        Angstroms.
    correlation : torch.Tensor
        `(n_tilts, )` normalized cross-correlation of each image with the
        reprojection of the lower tilts' reconstruction, at the final sweep
        (1 for the reference tilt closest to 0).
    """

    tilt_series: TiltSeries
    slab: tuple[float, float]
    shift_updates: torch.Tensor
    correlation: torch.Tensor


def align_by_projection_matching(
    images: torch.Tensor,
    tilt_series: TiltSeries,
    thickness: float,
    lowpass_angstroms: float | None = None,
    highpass_angstroms: float = 800.0,
    n_iterations: int = 5,
    max_shift: float | None = None,
    tolerance: float = 0.05,
    cover_all_tilts: bool = True,
    fit_slab: bool = True,
    preprocess: bool = True,
) -> ProjectionMatchingResult:
    """Refine per-tilt shifts by matching each tilt to a reprojection.

    Projection matching, the fine-alignment approach of AreTomo. Each sweep
    builds a tomogram outwards from the tilt closest to 0: every other tilt,
    in order of increasing tilt angle, is cross-correlated with the
    projection, at its own geometry, of the tomogram reconstructed from the
    lower tilts; its shift is updated by the correlation peak and it is
    then added to the tomogram. Sweeps repeat until the updates are below
    `tolerance` pixels.

    Matching against the lower tilts only anchors the alignment at 0
    degrees. Matching each tilt against all the other tilts instead lets the
    alignment drift by tens of Angstroms over successive sweeps on real data.

    The tomogram is a weighted backprojection in real space (ramp filter
    perpendicular to the tilt axis). A rigid translation of the tomogram does
    not change the projections; the part of each sweep's updates that only
    moves the tomogram is removed, so the tomogram stays where the starting
    geometry put it.

    Parameters
    ----------
    images : torch.Tensor
        `(n_tilts, h, w)` tilt images in the order of
        `tilt_series.tilt_angles`. Work at low resolution: about 20 A pixels.
    tilt_series : TiltSeries
        Starting geometry, e.g. from `torch_tiltxcorr.tiltxcorr`.
        `tilt_series.pixel_spacing` must be the pixel spacing of `images`;
        the tomogram uses the same spacing.
    thickness : float
        Thickness in Angstroms of the slab searched for the specimen (with
        `fit_slab`), or of the slab reconstructed for matching.
    lowpass_angstroms : float | None
        Low-pass filter for reconstruction and correlation, in Angstroms.
        Defaults to 4 pixels.
    highpass_angstroms : float
        High-pass filter, in Angstroms.
    n_iterations : int
        Maximum number of sweeps.
    max_shift : float | None
        Largest shift update per sweep, in Angstroms. Defaults to a quarter
        of the shorter image side.
    tolerance : float
        Stop when the largest update is below this many pixels.
    cover_all_tilts : bool
        Make the tomogram wide enough, perpendicular to the tilt axis, that
        every ray through every image crosses the whole slab (the image at
        the largest tilt sees a region 1 / cos(tilt) wider than at 0
        degrees; capped at 3x). Otherwise the reprojection is empty near the
        image edges at high tilt.
    fit_slab : bool
        At every sweep, reconstruct `thickness` with the current geometry,
        find the specimen in it (where the band-passed variance along z rises
        above half-way from background to peak, plus a 10 nm margin) and
        match within that slab only. A slab much thicker than the specimen
        weakens the 3D constraint; one thinner than the specimen biases it.
    preprocess : bool
        Subtract a background plane, band-pass filter and normalize the
        images first (`torch_tilt_series.preprocess_tilt_series_images`).

    Returns
    -------
    ProjectionMatchingResult
        The refined geometry, the shift updates and the final correlations.
    """
    n_tilts, h, w = images.shape
    device = images.device
    spacing = tilt_series.pixel_spacing
    if lowpass_angstroms is None:
        lowpass_angstroms = 4 * spacing
    if max_shift is None:
        max_shift = 0.25 * min(h, w) * spacing
    lowpass = min(0.5, spacing / lowpass_angstroms)
    if preprocess:
        images = preprocess_tilt_series_images(
            images,
            low=min(0.25, spacing / highpass_angstroms),
            high=lowpass,
            falloff=0.02,
        )
    mask = _taper(h, w, device)
    weighted = _ramp_filter(images * mask, tilt_series, lowpass)
    search_shape = _volume_shape(tilt_series, thickness, (h, w), cover_all_tilts)
    z_center, volume_shape = 0.0, search_shape

    geometry = tilt_series.copy()
    updates = []
    correlation = torch.ones(n_tilts, device=device)
    for _ in range(n_iterations):
        if fit_slab:  # re-found with the current geometry at every sweep
            z_center, slab_thickness = _fit_slab(
                weighted, geometry, search_shape, (h, w)
            )
            volume_shape = (
                max(1, round(slab_thickness / spacing)),
                search_shape[1],
                search_shape[2],
            )
        matrices, offsets = affine_projection(geometry)
        order = torch.argsort(geometry.tilt_angles.abs()).tolist()
        shifts = torch.zeros(n_tilts, 2, device=device)
        volume = torch.zeros(volume_shape, device=device)
        for k, t in enumerate(order):
            p = voxel_positions(
                matrices[t], offsets[t], volume_shape, (h, w), spacing, z_center
            )
            if k > 0:  # the tilt closest to 0 is the reference
                reprojection = splat(volume, p, (h, w))
                shift_px, correlation[t] = _match(
                    images[t] * mask, reprojection * mask, lowpass, max_shift / spacing
                )
                shifts[t] = shift_px * spacing
                p = p + shift_px
            volume += backproject(weighted[t], p)
        shifts = _remove_tomogram_translation(shifts, matrices)
        geometry.sample_translations = geometry.sample_translations + shifts
        updates.append(shifts)
        if shifts.abs().max() < tolerance * spacing:
            break
    return ProjectionMatchingResult(
        tilt_series=geometry,
        slab=(z_center, volume_shape[0] * spacing),
        shift_updates=torch.stack(updates),
        correlation=correlation,
    )


def _fit_slab(
    weighted: torch.Tensor,
    tilt_series: TiltSeries,
    volume_shape: tuple[int, int, int],
    image_shape: tuple[int, int],
    margin: float = 100.0,
) -> tuple[float, float]:
    """Z center and thickness (Angstroms) of the specimen in a reconstruction.

    The slab is where the band-passed standard deviation of each z slice,
    over the central region and in three strips across the tilt axis,
    rises above background + 50% of (peak - background); background is the
    median of the top and bottom 10% of slices. The union of the strips plus
    `margin` on each side is returned.
    """
    spacing = tilt_series.pixel_spacing
    matrices, offsets = affine_projection(tilt_series)
    volume = torch.zeros(volume_shape, device=weighted.device)
    for t in range(len(weighted)):
        p = voxel_positions(matrices[t], offsets[t], volume_shape, image_shape, spacing)
        volume += backproject(weighted[t], p)
    d, hv, wv = volume_shape
    across = min(wv, max(image_shape))
    x0 = wv // 2 - int(0.3 * across)
    core = volume[:, hv // 4 : 3 * hv // 4, x0 : x0 + int(0.6 * across)]
    core = _gaussian(core, 0.75) - _gaussian(core, 3.0)
    edge = max(1, d // 10)
    lows, highs = [], []
    for strip in torch.tensor_split(core, 3, dim=2):
        profile = strip.reshape(d, -1).std(dim=1)
        profile = F.avg_pool1d(profile[None, None], 5, stride=1, padding=2)[0, 0]
        background = torch.cat([profile[:edge], profile[-edge:]]).median()
        above = torch.nonzero(
            profile > background + 0.5 * (profile.max() - background)
        )[:, 0]
        lows.append(int(above.min()))
        highs.append(int(above.max()))
    low = (min(lows) - d // 2) * spacing - margin
    high = (max(highs) - d // 2) * spacing + margin
    half = d * spacing / 2
    low, high = max(low, -half), min(high, half)
    return (low + high) / 2, high - low


def _gaussian(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur (sigma in voxels), applied in Fourier space."""
    shape = volume.shape
    device = volume.device
    fz = torch.fft.fftfreq(shape[0], device=device)[:, None, None]
    fy = torch.fft.fftfreq(shape[1], device=device)[None, :, None]
    fx = torch.fft.rfftfreq(shape[2], device=device)[None, None, :]
    kernel = torch.exp(-2 * (math.pi * sigma) ** 2 * (fz**2 + fy**2 + fx**2))
    return torch.fft.irfftn(torch.fft.rfftn(volume) * kernel, s=shape)


def _volume_shape(
    tilt_series: TiltSeries,
    thickness: float,
    image_shape: tuple[int, int],
    cover_all_tilts: bool,
) -> tuple[int, int, int]:
    """(d, h, w) of the tomogram: y along the tilt axis, x perpendicular to it."""
    h, w = image_shape
    spacing = tilt_series.pixel_spacing
    axis = math.radians(float(tilt_series.tilt_axis_angle.float().mean()))
    # the tilt axis points along (-sin a, cos a) in image (x, y)
    along = h * abs(math.cos(axis)) + w * abs(math.sin(axis))
    across = w * abs(math.cos(axis)) + h * abs(math.sin(axis))
    if cover_all_tilts:
        largest = float(tilt_series.tilt_angles.abs().max())
        across /= max(math.cos(math.radians(largest)), 1 / 3)
    return (max(1, round(thickness / spacing)), round(along), round(across))


def _match(
    image: torch.Tensor, reprojection: torch.Tensor, lowpass: float, max_shift: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift (px) that moves `reprojection` onto `image`, and the peak NCC."""
    h, w = image.shape
    image = image - image.mean()
    reprojection = reprojection - reprojection.mean()
    p = h // 2, w // 2
    a = F.pad(image, [p[1], p[1], p[0], p[0]])
    b = F.pad(reprojection, [p[1], p[1], p[0], p[0]])
    hh, ww = a.shape
    product = torch.fft.rfft2(a) * torch.conj(torch.fft.rfft2(b))
    fy = torch.fft.fftfreq(hh, device=image.device)[:, None]
    fx = torch.fft.rfftfreq(ww, device=image.device)[None, :]
    product = product * (torch.sqrt(fy**2 + fx**2) <= lowpass)
    correlation = torch.fft.fftshift(torch.fft.irfft2(product, s=(hh, ww)))
    norm = torch.sqrt((image**2).sum() * (reprojection**2).sum()).clamp(min=1e-12)
    correlation = correlation / norm
    cy, cx = hh // 2, ww // 2
    yy = torch.arange(hh, device=image.device)[:, None] - cy
    xx = torch.arange(ww, device=image.device)[None, :] - cx
    correlation = torch.where(yy**2 + xx**2 <= max_shift**2, correlation, -torch.inf)
    index = int(correlation.argmax())
    py, px = index // ww, index % ww
    peak = correlation[py, px]
    dy = _vertex(correlation, py, px, 0)
    dx = _vertex(correlation, py, px, 1)
    shift = torch.tensor([py - cy + dy, px - cx + dx], device=image.device)
    return shift, peak


def _vertex(c: torch.Tensor, py: int, px: int, dim: int) -> float:
    """Sub-pixel offset of a parabola through the peak and its neighbors."""
    n = c.shape[dim]
    i = py if dim == 0 else px
    if i == 0 or i == n - 1:
        return 0.0
    if dim == 0:
        minus, mid, plus = c[py - 1, px], c[py, px], c[py + 1, px]
    else:
        minus, mid, plus = c[py, px - 1], c[py, px], c[py, px + 1]
    if not (torch.isfinite(minus) and torch.isfinite(plus)):
        return 0.0
    denominator = float(minus - 2 * mid + plus)
    if abs(denominator) < 1e-12:
        return 0.0
    return max(-0.5, min(0.5, 0.5 * float(minus - plus) / denominator))


def _ramp_filter(
    images: torch.Tensor, tilt_series: TiltSeries, lowpass: float
) -> torch.Tensor:
    """Ramp filter perpendicular to the tilt axis, with a cosine low-pass."""
    _, h, w = images.shape
    device = images.device
    fy = torch.fft.fftfreq(h, device=device)[:, None]
    fx = torch.fft.rfftfreq(w, device=device)[None, :]
    axis = torch.deg2rad(tilt_series.tilt_axis_angle.to(device).float())
    if axis.ndim == 0:
        axis = axis.expand(len(images))
    # the tilt axis points along (-sin a, cos a) in (x, y): the perpendicular
    # is (cos a, sin a)
    perpendicular = (
        fx[None] * torch.cos(axis)[:, None, None]
        + fy[None] * torch.sin(axis)[:, None, None]
    ).abs()
    radius = torch.sqrt(fy**2 + fx**2)
    envelope = torch.where(
        radius <= lowpass, 0.5 * (1 + torch.cos(math.pi * radius / lowpass)), 0.0
    )
    weights = (perpendicular + 1.0 / max(h, w)) * envelope
    return torch.fft.irfft2(torch.fft.rfft2(images) * weights, s=(h, w))


def _taper(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Cosine taper over the outer 10% of each image dimension."""

    def window(n: int) -> torch.Tensor:
        edge = max(1, round(0.1 * n))
        ramp = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, edge, device=device)))
        core = torch.ones(n - 2 * edge, device=device)
        return torch.cat([ramp, core, ramp.flip(0)])

    return window(h)[:, None] * window(w)[None, :]


def _remove_tomogram_translation(
    shifts: torch.Tensor, matrices: torch.Tensor
) -> torch.Tensor:
    """Remove the part of per-tilt shifts that a tomogram translation explains."""
    a = matrices.reshape(-1, 3).double().cpu()
    b = shifts.reshape(-1, 1).double().cpu()
    t = torch.linalg.lstsq(a, b).solution
    rigid = (a @ t).reshape(shifts.shape)
    return shifts - rigid.to(shifts.dtype).to(shifts.device)
