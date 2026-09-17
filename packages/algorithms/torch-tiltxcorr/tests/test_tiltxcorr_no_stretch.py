import torch
from torch_fourier_shift import fourier_shift_image_2d

from torch_tiltxcorr import tiltxcorr_no_stretch


def _generate_shifted_tilt_series(seed: int = 0, n_tilts: int = 5, size: int = 64):
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand((size, size), generator=generator)
    tilt_series = base.expand(n_tilts, size, size).clone()
    tilt_angles = torch.linspace(-10, 10, steps=n_tilts)

    # shifts are cumulative relative to the 0-tilt anchor -> keep it at 0
    shifts = torch.rand((n_tilts, 2), generator=generator) * 4 - 2
    shifts[tilt_angles == 0] = 0
    tilt_series = fourier_shift_image_2d(tilt_series, shifts=shifts)
    return tilt_series, tilt_angles, shifts


def test_tiltxcorr_no_stretch_recovers_shifts():
    tilt_series, tilt_angles, applied_shifts = _generate_shifted_tilt_series()
    ground_truth_shifts = -1 * applied_shifts

    estimated_shifts = tiltxcorr_no_stretch(
        tilt_series=tilt_series, tilt_angles=tilt_angles
    )
    error = torch.abs(estimated_shifts - ground_truth_shifts).max()
    assert error < 0.5


def test_tiltxcorr_no_stretch_preprocess_is_optional():
    """preprocess=False must run without error and change the result."""
    tilt_series, tilt_angles, _ = _generate_shifted_tilt_series()
    kwargs = dict(tilt_series=tilt_series, tilt_angles=tilt_angles)

    default_shifts = tiltxcorr_no_stretch(**kwargs)
    no_preprocess_shifts = tiltxcorr_no_stretch(**kwargs, preprocess=False)

    assert default_shifts.shape == (len(tilt_angles), 2)
    assert not torch.allclose(default_shifts, no_preprocess_shifts)
