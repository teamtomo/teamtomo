import math

import pytest
import torch

from torch_refine_tilt_axis_angle import refine_tilt_axis_angle
from torch_refine_tilt_axis_angle.refine_tilt_axis_angle import _common_line_score


def _tilt_series_with_common_line(
    image_shape: tuple[int, int],
    tilt_axis_angle: float,
    n_tilts: int = 60,
    n_harmonics: int = 40,
    noise_std: float = 10.0,
    seed: int = 0,
) -> torch.Tensor:
    """Build a synthetic stack that shares Fourier content along one direction.

    Each image is `common(u) + independent_noise`, where `u` is the
    coordinate along `tilt_axis_angle` and `common` is identical across the
    stack. `common` is a broadband sum of random sinusoids so it populates a
    continuous ridge in Fourier space, rather than a couple of isolated
    frequency bins. Real tilt series behave the same way: every image's
    Fourier transform agrees along the line perpendicular to the tilt axis
    (the common line), which is exactly the structure
    `refine_tilt_axis_angle` is designed to detect.

    The defaults use a low SNR (as in real tilt series): with little noise,
    the window leakage around the ridge is just as coherent across the
    stack as the ridge itself, which flattens the top of the coherence peak.
    """
    h, w = image_shape
    generator = torch.Generator().manual_seed(seed)
    y = torch.linspace(-h / 2, h / 2, h)
    x = torch.linspace(-w / 2, w / 2, w)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rad = math.radians(tilt_axis_angle)
    u = yy * math.cos(rad) - xx * math.sin(rad)

    periods = torch.linspace(4, 40, n_harmonics)
    phases = torch.rand(n_harmonics, generator=generator) * 2 * torch.pi
    amplitudes = 0.5 + torch.rand(n_harmonics, generator=generator)
    common = sum(
        amplitudes[i] * torch.sin(2 * torch.pi * u / periods[i] + phases[i])
        for i in range(n_harmonics)
    )

    noise = noise_std * torch.randn((n_tilts, h, w), generator=generator)
    return common + noise


def _angular_distance(a: float, b: float) -> float:
    """Smallest difference between two angles, modulo the 180 deg line symmetry."""
    return abs((a - b + 90) % 180 - 90)


@pytest.mark.parametrize("image_shape", [(96, 96), (80, 128), (128, 80)])
@pytest.mark.parametrize("true_angle", [10.0, 55.0, 100.0, 150.0])
def test_refine_tilt_axis_angle_recovers_known_angle(image_shape, true_angle):
    """The common line direction should be recovered to sub-degree precision."""
    tilt_series = _tilt_series_with_common_line(image_shape, tilt_axis_angle=true_angle)

    result = refine_tilt_axis_angle(tilt_series)

    assert isinstance(result, float)
    assert _angular_distance(result, true_angle) < 1.0


def _projected_blob_tilt_series(
    image_shape: tuple[int, int],
    tilt_axis_angle: float,
    n_blobs: int = 40,
    sigma: float = 2.0,
    seed: int = 0,
) -> torch.Tensor:
    """Project a slab of 3D Gaussian blobs over a +/-60 deg tilt range.

    A stand-in for particles and fiducials in ice. Each blob centre is
    rotated by `Rz(tilt_axis_angle) @ Ry(tilt)` and projected along z; the
    projection of an isotropic 3D Gaussian is a 2D Gaussian at the projected
    centre, so the stack is computed analytically without resampling a
    volume. Unlike `_tilt_series_with_common_line`, the images only agree
    along the common line, and blob brightness varies strongly: the summed
    power spectrum (not normalized by the per-image power) peaks away from
    the true angle on this data.
    """
    h, w = image_shape
    generator = torch.Generator().manual_seed(seed)
    x = (torch.rand(n_blobs, generator=generator) - 0.5) * 0.7 * w
    y = (torch.rand(n_blobs, generator=generator) - 0.5) * 0.7 * h
    z = (torch.rand(n_blobs, generator=generator) - 0.5) * 0.08 * min(h, w)
    amplitudes = 1.0 + 6.0 * torch.rand(n_blobs, generator=generator)

    tilts = torch.deg2rad(torch.arange(-60.0, 60.1, 3.0))[:, None]  # (tilt, 1)
    rad = math.radians(tilt_axis_angle)
    x_tilted = x * torch.cos(tilts) + z * torch.sin(tilts)  # (tilt, blob)
    x_proj = x_tilted * math.cos(rad) - y * math.sin(rad)
    y_proj = x_tilted * math.sin(rad) + y * math.cos(rad)

    cy = torch.arange(h) - (h - 1) / 2
    cx = torch.arange(w) - (w - 1) / 2
    gy = torch.exp(-((cy - y_proj[..., None]) ** 2) / (2 * sigma**2))
    gx = torch.exp(-((cx - x_proj[..., None]) ** 2) / (2 * sigma**2))
    return torch.einsum("b,tby,tbx->tyx", amplitudes, gy, gx)


@pytest.mark.parametrize("image_shape", [(128, 128), (192, 192)])
@pytest.mark.parametrize("true_angle", [-5.0, 40.0, 100.0, 150.0])
def test_refine_tilt_axis_angle_recovers_angle_from_projected_blobs(
    image_shape, true_angle
):
    """Regression test: bright, uneven specimens must not bias the angle."""
    tilt_series = _projected_blob_tilt_series(image_shape, tilt_axis_angle=true_angle)

    result = refine_tilt_axis_angle(tilt_series)

    assert _angular_distance(result, true_angle) < 1.0


def test_refine_tilt_axis_angle_respects_search_window():
    """The result must lie within +/-90 deg of the initial guess, unwrapped."""
    true_angle = 100.0
    tilt_series = _tilt_series_with_common_line((96, 96), tilt_axis_angle=true_angle)

    initial_guess = 30.0
    result = refine_tilt_axis_angle(tilt_series, tilt_axis_angle=initial_guess)

    assert initial_guess - 90 <= result <= initial_guess + 90
    assert _angular_distance(result, true_angle) < 1.0


def test_refine_tilt_axis_angle_without_refinement_step():
    """Skipping refinement should still recover the angle to coarse precision."""
    true_angle = 42.0
    tilt_series = _tilt_series_with_common_line((96, 96), tilt_axis_angle=true_angle)

    coarse_angle_step = 1.0
    result = refine_tilt_axis_angle(
        tilt_series, coarse_angle_step=coarse_angle_step, refine=False
    )

    assert isinstance(result, float)
    assert _angular_distance(result, true_angle) <= coarse_angle_step


def test_refine_tilt_axis_angle_respects_radius_band():
    """min/max_fraction_of_nyquist should select which frequencies are used.

    Builds a stack containing two common lines at different angles, one
    carried entirely by long-period (low frequency) content and the other
    entirely by short-period (high frequency) content. Restricting the
    search to each frequency band in turn should recover the angle that
    lives in that band, proving the radius bounds actually take effect
    rather than just being accepted and ignored.
    """
    image_shape = (96, 96)
    h, w = image_shape
    angle_low, angle_high = 20.0, 110.0
    generator = torch.Generator().manual_seed(1)

    y = torch.linspace(-h / 2, h / 2, h)
    x = torch.linspace(-w / 2, w / 2, w)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    def _common(angle_deg: float, periods: list[float]) -> torch.Tensor:
        rad = math.radians(angle_deg)
        u = yy * math.cos(rad) - xx * math.sin(rad)
        phases = torch.rand(len(periods), generator=generator) * 2 * torch.pi
        return sum(
            torch.sin(2 * torch.pi * u / period + phase)
            for period, phase in zip(periods, phases, strict=True)
        )

    low_freq_signal = _common(angle_low, periods=list(torch.linspace(24, 40, 10)))
    high_freq_signal = _common(angle_high, periods=list(torch.linspace(4, 6, 10)))
    noise = 0.5 * torch.randn((30, h, w), generator=generator)
    tilt_series = low_freq_signal + high_freq_signal + noise

    low_band_result = refine_tilt_axis_angle(
        tilt_series, min_fraction_of_nyquist=0.02, max_fraction_of_nyquist=0.12
    )
    high_band_result = refine_tilt_axis_angle(
        tilt_series, min_fraction_of_nyquist=0.3, max_fraction_of_nyquist=0.6
    )

    # The low band only spans ~1-6 px from the Fourier origin, where a
    # one-pixel step subtends more than 10 deg, so so few bins cannot pin
    # the line's angle down to sub-degree precision. 2 deg still clearly
    # separates the two lines, which are 90 deg apart.
    assert _angular_distance(low_band_result, angle_low) < 2.0
    assert _angular_distance(high_band_result, angle_high) < 1.0


def test_refine_tilt_axis_angle_refine_step_improves_precision():
    """A finer refine_angle_step/refine_range should beat the coarse grid alone."""
    true_angle = 63.0
    tilt_series = _tilt_series_with_common_line((96, 96), tilt_axis_angle=true_angle)

    coarse_only = refine_tilt_axis_angle(
        tilt_series, coarse_angle_step=2.0, refine=False
    )
    refined = refine_tilt_axis_angle(
        tilt_series,
        coarse_angle_step=2.0,
        refine=True,
        refine_range=2.0,
        refine_angle_step=0.05,
    )

    coarse_error = _angular_distance(coarse_only, true_angle)
    refined_error = _angular_distance(refined, true_angle)
    assert refined_error < coarse_error
    assert refined_error < 0.5


def _make_indexable_spectrum(image_shape: tuple[int, int]) -> torch.Tensor:
    """A spectrum with a distinct value at every bin, for exact assertions."""
    h, w = image_shape
    return torch.arange(h * (w // 2 + 1), dtype=torch.float32).reshape(h, w // 2 + 1)


def test_common_line_score_interpolates_expected_bins():
    """Row wraparound, bilinear weights, and conjugate-symmetry flip.

    Uses a spectrum where every bin holds a distinct value
    (`5 * row + col` for an 8x8 image), so the values interpolated by
    `_common_line_score` can be checked against hand-computed expectations
    rather than merely trusting the code ran.
    """
    image_shape = (8, 8)
    spectrum = _make_indexable_spectrum(image_shape)
    diagonal = math.sqrt(2) / 8  # rho that lands on (row, col) = (1, 1) at 45 deg

    def lookup(angle: float, rho: float) -> float:
        return _common_line_score(
            torch.tensor([angle]), torch.tensor([rho]), spectrum, image_shape
        ).item()

    # on-grid points: theta=45 -> (1, 1); theta=-45 -> row -1 wraps to 7.
    assert lookup(45.0, diagonal) == pytest.approx(spectrum[1, 1].item(), abs=1e-3)
    assert lookup(-45.0, diagonal) == pytest.approx(spectrum[7, 1].item(), abs=1e-3)
    # theta=225 is the same line as 45 but cos<0, so the conjugate point
    # (1, 1) is looked up; likewise theta=135 flips to (-1, 1) -> (7, 1).
    assert lookup(225.0, diagonal) == pytest.approx(spectrum[1, 1].item(), abs=1e-3)
    assert lookup(135.0, diagonal) == pytest.approx(spectrum[7, 1].item(), abs=1e-3)

    # halfway between columns 2 and 3 on row 0.
    expected = (spectrum[0, 2] + spectrum[0, 3]).item() / 2
    assert lookup(0.0, 2.5 / 8) == pytest.approx(expected, abs=1e-3)
    # (row, col) = (-0.5, 0.5): interpolates across the row wraparound,
    # between rows 7 and 0.
    expected = spectrum[[7, 7, 0, 0], [0, 1, 0, 1]].mean().item()
    assert lookup(-45.0, 0.5 * diagonal) == pytest.approx(expected, abs=1e-3)


def test_common_line_score_masks_out_of_range_frequencies():
    """Neighbours outside the stored rfft columns contribute zero."""
    image_shape = (8, 8)
    spectrum = _make_indexable_spectrum(image_shape)

    def lookup(rho: float) -> float:
        return _common_line_score(
            torch.tensor([0.0]), torch.tensor([rho]), spectrum, image_shape
        ).item()

    # valid columns for a w=8 rfft are 0..4. rho=0.7 -> col 5.6: both
    # neighbours (5, 6) are out of range.
    assert lookup(0.7) == 0.0
    # rho=0.55 -> col 4.4: only the col-4 neighbour (weight 0.6) is in range.
    assert lookup(0.55) == pytest.approx(0.6 * spectrum[0, 4].item(), abs=1e-3)
    # rho=0.4 -> col 3.2: both neighbours are in range.
    expected = 0.8 * spectrum[0, 3].item() + 0.2 * spectrum[0, 4].item()
    assert lookup(0.4) == pytest.approx(expected, abs=1e-3)
