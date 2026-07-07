"""Tests for the high-level torch_scattering.projection() wrapper."""

import torch

from torch_scattering.projection import projection


def test_projection_vacuum_leaves_plane_wave_unchanged():
    """Zero potential (vacuum) must leave a normally-incident plane wave unchanged.

    With no material, there is nothing to scatter the beam, so the exit
    wave must equal the incident unit-amplitude, zero-phase plane wave.
    """
    potential = torch.zeros(1, 5, 8, 8, dtype=torch.complex64)
    exit_wave = projection(potential, pixel_size=1.0, energy=300.0)
    assert torch.allclose(exit_wave, torch.ones_like(exit_wave), atol=1e-5)


def test_projection_conserves_energy_for_real_potential():
    """A real (non-absorbing) potential volume must conserve wave energy.

    Summing a real potential along Z stays real, so the resulting
    transmission function is unitary and the exit wave energy must equal
    the incident plane wave's energy (H * W).
    """
    torch.manual_seed(0)
    potential = torch.randn(1, 5, 8, 8, dtype=torch.complex64).real.to(torch.complex64)
    exit_wave = projection(potential, pixel_size=1.0, energy=300.0)
    output_energy = exit_wave.abs().pow(2).sum()
    expected_energy = torch.tensor(8.0 * 8.0)
    assert torch.allclose(output_energy, expected_energy, rtol=1e-4)
