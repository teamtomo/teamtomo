"""Tests for CTFFind-style 1D windowed GoF resolution."""

import pytest
import torch

from torch_ctf_estimation.estimate_ctf_1d.estimate_gof_resolution_1d import (
    estimate_gof_by_cycles,
    estimate_gof_resolution_1d,
    interpolate_cc_drop_angstroms,
    simulate_thin_abs_ctf_1d,
    simulate_thin_power_1d,
    thickness_from_first_node_angstroms,
)


def test_interpolate_cc_drop_crosses_threshold():
    spacing = torch.tensor([8.0, 6.0, 5.0, 4.0])
    cc = torch.tensor([0.9, 0.7, 0.4, 0.1])
    d = interpolate_cc_drop_angstroms(spacing, cc, 0.5)
    assert d == pytest.approx(6.0 - (2.0 / 3.0), abs=1e-6)


def test_thickness_from_first_node_scales_as_d_squared():
    t5 = thickness_from_first_node_angstroms(5.0, 300.0)
    t10 = thickness_from_first_node_angstroms(10.0, 300.0)
    assert t5 > 0.0
    assert abs(t10 / t5 - 4.0) < 1e-4


def test_gof_1d_recovers_known_thin_cutoff():
    """Thin CTF² truncated at 5 Å should drop windowed GoF near 5 Å."""
    n = 256
    pixel = 1.4
    defocus_um = 1.2
    voltage = 300.0
    cs = 2.7
    ac = 0.07
    model = simulate_thin_power_1d(
        n_samples=n,
        defocus_um=defocus_um,
        voltage_kev=voltage,
        spherical_aberration_mm=cs,
        amplitude_contrast=ac,
        pixel_spacing_angstroms=pixel,
    )
    n_real = 2 * (n - 1)
    freq = torch.fft.rfftfreq(n_real, d=pixel)
    assert freq.numel() == n
    cutoff_A = 5.0
    data = torch.where(freq <= (1.0 / cutoff_A), model, torch.zeros_like(model))
    band = (freq >= 1.0 / 30.0) & (freq <= 1.0 / 4.0)
    result = estimate_gof_resolution_1d(
        data[band],
        freq[band],
        model[band],
        defocus_um=defocus_um,
        voltage_kev=voltage,
        window_cycles=1.0,
    )
    assert abs(result.fit_res_A - cutoff_A) < 0.8
    assert result.thickness_from_node_A > 0.0


def test_gof_cycles_recovers_known_abs_ctf_cutoff():
    """|CTF| truncated at 5 Å should drop cycle NCC near 5 Å."""
    n = 256
    pixel = 1.4
    defocus_um = 1.2
    voltage = 300.0
    cs = 2.7
    ac = 0.07
    model = simulate_thin_abs_ctf_1d(
        n_samples=n,
        defocus_um=defocus_um,
        voltage_kev=voltage,
        spherical_aberration_mm=cs,
        amplitude_contrast=ac,
        pixel_spacing_angstroms=pixel,
    )
    n_real = 2 * (n - 1)
    freq = torch.fft.rfftfreq(n_real, d=pixel)
    cutoff_A = 5.0
    data = torch.where(freq <= (1.0 / cutoff_A), model, torch.zeros_like(model))
    band = (freq >= 1.0 / 30.0) & (freq <= 1.0 / 4.0)
    result = estimate_gof_by_cycles(
        data[band],
        freq[band],
        model[band],
        voltage_kev=voltage,
        equalize=True,
        cycles_per_window=1.0,
    )
    assert abs(result.fit_res_A - cutoff_A) < 1.0
    assert result.thickness_from_node_A > 0.0
