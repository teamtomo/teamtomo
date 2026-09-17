import math

import pytest
import torch

from torch_refine_tilt_axis_angle import refine_tilt_axis_angle
from torch_refine_tilt_axis_angle.refine_tilt_axis_angle import _common_line_power


def _tilt_series_with_common_line(
    image_shape: tuple[int, int],
    tilt_axis_angle: float,
    n_tilts: int = 30,
    n_harmonics: int = 20,
    noise_std: float = 1.0,
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

    assert _angular_distance(low_band_result, angle_low) < 1.0
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


def _make_indexable_power_sum(image_shape: tuple[int, int]) -> torch.Tensor:
    """A power spectrum with a distinct value at every bin, for exact assertions."""
    h, w = image_shape
    return torch.arange(h * (w // 2 + 1), dtype=torch.float32).reshape(h, w // 2 + 1)


def test_common_line_power_indexes_expected_bins():
    """Row wraparound, column lookup, and conjugate-symmetry flip, checked exactly.

    Uses a power spectrum where every bin holds a distinct value, so the bins
    picked out by `_common_line_power` can be checked against hand-computed
    expected indices rather than merely trusting the code ran.
    """
    image_shape = (8, 8)
    power_sum = _make_indexable_power_sum(image_shape)
    rhos = torch.tensor([0.25])

    # theta=0: fy=0, fx=0.25 -> row 0, col round(0.25*8)=2.
    # theta=80: fy=sin(80)*0.25=0.246, fx=cos(80)*0.25=0.043
    #   -> row round(0.246*8)=2, col round(0.043*8)=0.
    # theta=260 (=80+180): same line as theta=80, but cos<0 so the conjugate
    # point is looked up instead; should land on the exact same bin as
    # theta=80 despite theta=260's own (fy, fx) being different.
    angles = torch.tensor([0.0, 80.0, 260.0])
    power = _common_line_power(angles, rhos, power_sum, image_shape)

    expected = torch.tensor([power_sum[0, 2], power_sum[2, 0], power_sum[2, 0]])
    assert torch.equal(power, expected)


def test_common_line_power_masks_out_of_range_frequencies():
    """Frequencies that round outside the stored rfft columns contribute zero."""
    image_shape = (8, 8)
    power_sum = _make_indexable_power_sum(image_shape)

    # rho=0.6 is above Nyquist (0.5): at theta=0, col = round(0.6*8) = 5,
    # which is out of range for a w=8 rfft (valid columns are 0..4).
    out_of_range_power = _common_line_power(
        torch.tensor([0.0]), torch.tensor([0.6]), power_sum, image_shape
    )
    assert out_of_range_power.item() == 0.0

    # a valid rho at the same angle should pick up the real bin value instead.
    in_range_power = _common_line_power(
        torch.tensor([0.0]), torch.tensor([0.4]), power_sum, image_shape
    )
    assert in_range_power.item() == power_sum[0, 3]
