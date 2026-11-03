"""Track image patches through a tilt series by stretched cross-correlation."""

import math

import torch
import torch.nn.functional as F
from torch_affine_utils.transforms_2d import R, S
from torch_subpixel_crop import subpixel_crop_2d
from torch_tilt_series import preprocess_tilt_series_images

from torch_tiltxcorr.tiltxcorr import tiltxcorr
from torch_tiltxcorr.utils import (
    apply_stretch_perpendicular_to_tilt_axis,
    calculate_cross_correlation,
    taper_image_edges,
)


def tiltxcorr_patches(
    tilt_series: torch.Tensor,  # (b, h, w)
    tilt_angles: torch.Tensor,  # (b, )
    tilt_axis_angle: float,
    shifts: torch.Tensor | None = None,  # (b, 2)
    patch_size: int = 128,
    patch_overlap: float = 0.33,
    n_iterations: int = 2,
    pixel_spacing_angstroms: float | None = None,
    lowpass_angstroms: float | None = None,
    preprocess: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track a grid of image patches through a tilt series.

    The patch-tracking mode of IMOD's tiltxcorr: patches are laid out on a
    grid in the tilt closest to 0 and followed outwards, tilt by tilt. Each
    patch is cross-correlated with the patch at its predicted position in the
    next tilt, after stretching that patch perpendicular to the tilt axis by
    the ratio of the cosines of the two tilt angles. The predicted position
    assumes the patch center lies on the plane z = 0 of the coarse alignment
    (`shifts`); the correlation then measures where it actually is.

    A patch is a column through the specimen rather than a point, so its
    tracked center drifts with tilt as material enters and leaves it. To fit
    the tracks with `torch_tiltalign.fit_tilt_series`, cut them into short
    pieces with `chop_tracks`, as IMOD does with imodchopconts.

    Parameters
    ----------
    tilt_series : torch.Tensor
        `(b, h, w)` tilt images.
    tilt_angles : torch.Tensor
        `(b, )` tilt angles in degrees.
    tilt_axis_angle : float
        Tilt axis angle in degrees, as for `tiltxcorr`.
    shifts : torch.Tensor | None
        `(b, 2)` yx coarse-alignment shifts in pixels, in the convention of
        `tiltxcorr` (the shift that aligns each image to the tilt closest to
        0). Computed with `tiltxcorr` if None.
    patch_size : int
        Patch sidelength in pixels.
    patch_overlap : float
        Fractional overlap of neighboring patches in the starting grid.
    n_iterations : int
        Correlations per patch and tilt; after the first, the patch is
        re-extracted at the tracked position and correlated again.
    pixel_spacing_angstroms : float | None
        Pixel spacing, used with `lowpass_angstroms` for preprocessing.
    lowpass_angstroms : float | None
        Low-pass cutoff in Angstroms.
    preprocess : bool
        Preprocess images as in `tiltxcorr` (plane subtraction, bandpass,
        normalization).

    Returns
    -------
    tracks_yx : torch.Tensor
        `(n_patches, b, 2)` tracked patch centers, yx, in pixels relative to
        the image center (`h // 2, w // 2`) of the input images. NaN where the
        patch was not tracked.
    observed : torch.Tensor
        `(n_patches, b)` boolean, True where the patch was tracked.
    """
    b, h, w = tilt_series.shape
    device = tilt_series.device
    tilt_angles = torch.as_tensor(tilt_angles, device=device).float()

    if lowpass_angstroms is None or pixel_spacing_angstroms is None:
        lowpass_cycles_per_pixel = 0.5
    else:
        lowpass_cycles_per_pixel = pixel_spacing_angstroms / lowpass_angstroms
    if preprocess:
        tilt_series = preprocess_tilt_series_images(
            tilt_series, low=0.025, high=lowpass_cycles_per_pixel, falloff=0.025
        )
    if shifts is None:
        shifts = tiltxcorr(tilt_series, tilt_angles, tilt_axis_angle, preprocess=False)
    shifts = torch.as_tensor(shifts, device=device).float()

    # patch grid at the reference tilt, in coarse-aligned coordinates
    half = patch_size / 2
    step = max(1.0, patch_size * (1 - patch_overlap))
    ys = _grid_1d(h, half, step, device)
    xs = _grid_1d(w, half, step, device)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).reshape(-1, 2)
    n_patches = len(grid)

    tracks = torch.full((n_patches, b, 2), torch.nan, device=device)
    reference = int(torch.argmin(tilt_angles.abs()))
    tracks[:, reference] = grid - shifts[reference]

    order = torch.argsort(tilt_angles)
    position = int((order == reference).nonzero())
    for branch in (order[position:], order[: position + 1].flip(0)):
        for previous, current in zip(branch[:-1].tolist(), branch[1:].tolist()):
            tracks[:, current] = _track_to_next_tilt(
                images=tilt_series,
                positions=tracks[:, previous],
                previous=previous,
                current=current,
                tilt_angles=tilt_angles,
                tilt_axis_angle=tilt_axis_angle,
                shifts=shifts,
                patch_size=patch_size,
                n_iterations=n_iterations,
            )
    observed = torch.isfinite(tracks).all(dim=-1)
    return tracks, observed


def chop_tracks(
    tracks_yx: torch.Tensor,
    observed: torch.Tensor,
    tilt_angles: torch.Tensor,
    segment_length: int = 16,
    segment_overlap: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut tracks into overlapping pieces of consecutive tilts.

    Equivalent to IMOD's imodchopconts for patch-tracking models: each
    piece is fitted as a separate 3D point, which a patch is only
    approximately over a short range of tilts.

    Parameters
    ----------
    tracks_yx : torch.Tensor
        `(n_tracks, n_tilts, 2)` tracked positions.
    observed : torch.Tensor
        `(n_tracks, n_tilts)` boolean.
    tilt_angles : torch.Tensor
        `(n_tilts, )` tilt angles in degrees; consecutive means adjacent in
        tilt angle.
    segment_length : int
        Number of tilts per piece.
    segment_overlap : int
        Number of tilts shared by consecutive pieces.

    Returns
    -------
    tracks_yx : torch.Tensor
        `(n_pieces, n_tilts, 2)`, NaN outside each piece.
    observed : torch.Tensor
        `(n_pieces, n_tilts)` boolean.
    """
    if not 0 <= segment_overlap < segment_length:
        raise ValueError("segment_overlap must be in [0, segment_length)")
    order = torch.argsort(torch.as_tensor(tilt_angles, device=tracks_yx.device))
    step = segment_length - segment_overlap
    pieces, pieces_observed = [], []
    for track, seen in zip(tracks_yx, observed):
        tilts = order[seen[order]]  # observed tilts in angle order
        n = len(tilts)
        if n == 0:
            continue
        starts = list(range(0, max(n - segment_overlap, 1), step))
        if n > segment_length and starts[-1] + segment_length < n:
            starts.append(n - segment_length)
        for start in starts:
            chosen = tilts[start : start + segment_length]
            mask = torch.zeros_like(seen)
            mask[chosen] = True
            pieces.append(torch.where(mask[:, None], track, torch.nan))
            pieces_observed.append(mask)
    if not pieces:
        return tracks_yx[:0], observed[:0]
    return torch.stack(pieces), torch.stack(pieces_observed)


def _grid_1d(
    length: int, half: float, step: float, device: torch.device
) -> torch.Tensor:
    """Patch centers (relative to the image center) that fit inside `length`."""
    extent = length / 2 - half - 1
    if extent <= 0:
        return torch.zeros(1, device=device)
    n = int(math.floor(2 * extent / step)) + 1
    span = (n - 1) * step
    return torch.linspace(-span / 2, span / 2, n, device=device)


def _track_to_next_tilt(
    images: torch.Tensor,
    positions: torch.Tensor,  # (n, 2) yx in the previous tilt, from image center
    previous: int,
    current: int,
    tilt_angles: torch.Tensor,
    tilt_axis_angle: float,
    shifts: torch.Tensor,
    patch_size: int,
    n_iterations: int,
) -> torch.Tensor:
    """Positions of patches in `current`, given their positions in `previous`."""
    h, w = images.shape[-2:]
    center = torch.tensor([h // 2, w // 2], device=images.device).float()
    valid = torch.isfinite(positions).all(dim=-1)
    out = torch.full_like(positions, torch.nan)
    if not valid.any():
        return out
    cos_previous = math.cos(math.radians(float(tilt_angles[previous])))
    cos_current = math.cos(math.radians(float(tilt_angles[current])))
    scale = cos_previous / cos_current

    # predict: stretch about the tilt axis in coarse-aligned coordinates
    stretch = _stretch_matrix(tilt_axis_angle, 1 / scale, images.device)
    aligned = positions[valid] + shifts[previous]
    predicted = aligned @ stretch.T - shifts[current]

    reference = subpixel_crop_2d(
        images[previous], positions[valid] + center, sidelength=patch_size
    )
    reference = taper_image_edges(reference)
    estimate = predicted
    for _ in range(n_iterations):
        moving = subpixel_crop_2d(
            images[current], estimate + center, sidelength=patch_size
        )
        moving = _stretch_patches(moving, tilt_axis_angle, scale)
        moving = taper_image_edges(moving)
        p = patch_size // 2
        correlation = calculate_cross_correlation(
            F.pad(reference, [p] * 4), F.pad(moving, [p] * 4)
        )
        correlation = F.pad(correlation, [-p] * 4)
        shift = _subpixel_peaks(correlation)  # aligns moving to reference
        # back to unstretched pixels of the current tilt
        shift = shift @ _stretch_matrix(tilt_axis_angle, 1 / scale, images.device).T
        estimate = estimate - shift

    half = patch_size / 2
    inside = (estimate[:, 0].abs() <= h / 2 - half) & (
        estimate[:, 1].abs() <= w / 2 - half
    )
    estimate[~inside] = torch.nan
    out[valid] = estimate
    return out


def _stretch_patches(
    patches: torch.Tensor, tilt_axis_angle: float, scale: float
) -> torch.Tensor:
    """`apply_stretch_perpendicular_to_tilt_axis` for a `(n, h, w)` batch."""
    stretched = apply_stretch_perpendicular_to_tilt_axis(
        patches, tilt_axis_angle=tilt_axis_angle, scale_factor=scale
    )
    # a (n, h, w) batch is sampled as n channels and comes back as (h, w, n)
    if stretched.shape != patches.shape:
        stretched = stretched.permute(2, 0, 1)
    return stretched.contiguous()


def _stretch_matrix(
    tilt_axis_angle: float, factor: float, device: torch.device
) -> torch.Tensor:
    """2x2 yx matrix scaling by `factor` perpendicular to the tilt axis."""
    m = (
        R(tilt_axis_angle, yx=True, device=device)
        @ S((1.0, factor), device=device)
        @ R(-tilt_axis_angle, yx=True, device=device)
    )
    return m[:2, :2]


def _subpixel_peaks(correlation: torch.Tensor) -> torch.Tensor:
    """Peak position relative to the center of each image, with a parabola fit.

    `(n, h, w)` -> `(n, 2)` yx.
    """
    n, h, w = correlation.shape
    flat = correlation.reshape(n, -1).argmax(dim=-1)
    py, px = flat // w, flat % w
    py = py.clamp(1, h - 2)
    px = px.clamp(1, w - 2)
    idx = torch.arange(n, device=correlation.device)

    def vertex(minus: torch.Tensor, mid: torch.Tensor, plus: torch.Tensor):
        denominator = minus - 2 * mid + plus
        offset = (
            0.5
            * (minus - plus)
            / torch.where(
                denominator.abs() > 1e-12, denominator, torch.ones_like(denominator)
            )
        )
        return offset.clamp(-0.5, 0.5)

    c = correlation[idx, py, px]
    dy = vertex(correlation[idx, py - 1, px], c, correlation[idx, py + 1, px])
    dx = vertex(correlation[idx, py, px - 1], c, correlation[idx, py, px + 1])
    peak = torch.stack([py + dy, px + dx], dim=-1)
    return peak - torch.tensor([h // 2, w // 2], device=correlation.device)
