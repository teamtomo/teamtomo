"""Tests for the high-level torch_scattering.multislice() wrapper."""

import pytest
import torch

from torch_scattering.multislice import multislice


def test_multislice_vacuum_leaves_plane_wave_unchanged():
    """Zero potential (vacuum) must leave a normally-incident plane wave unchanged.

    With no material, there is nothing to scatter the beam. The incident
    wave is a uniform (constant) plane wave, whose Fourier transform is
    concentrated entirely at the zero-frequency component - and the
    propagator is always 1 at zero frequency - so it must pass through
    every slice unchanged.
    """
    potential = torch.zeros(1, 5, 8, 8, dtype=torch.complex64)
    exit_wave = multislice(potential, pixel_size=1.0, energy=300.0)
    assert torch.allclose(exit_wave, torch.ones_like(exit_wave), atol=1e-5)


def test_multislice_conserves_energy_for_real_potential():
    """A real (non-absorbing) potential volume must conserve wave energy.

    Every slice composes a unitary transmission function (real V) with the
    always-unitary Fresnel propagator, so after propagating through all
    slices, the total wave energy must equal the incident plane wave's
    energy (H * W, since the incident wave has unit amplitude everywhere).
    """
    torch.manual_seed(0)
    potential = torch.randn(1, 5, 8, 8, dtype=torch.complex64).real.to(torch.complex64)
    exit_wave = multislice(potential, pixel_size=1.0, energy=300.0)
    output_energy = exit_wave.abs().pow(2).sum()
    expected_energy = torch.tensor(8.0 * 8.0)
    assert torch.allclose(output_energy, expected_energy, rtol=1e-4)


def test_multislice_explicit_full_n_slices_matches_default():
    """n_slices == Z (explicit) must exactly match the n_slices=None default.

    Both should walk one true slice at a time, so the results must be
    bit-for-bit identical, not merely close.
    """
    torch.manual_seed(0)
    potential = torch.randn(1, 5, 8, 8, dtype=torch.complex64).real.to(torch.complex64)
    default_wave = multislice(potential, pixel_size=1.0, energy=300.0)
    explicit_wave = multislice(potential, pixel_size=1.0, energy=300.0, n_slices=5)
    assert torch.equal(default_wave, explicit_wave)


def test_multislice_grouped_conserves_energy_for_real_potential():
    """Grouping slices (n_slices < Z) must still conserve wave energy.

    Summed potential over a chunk is still real (non-absorbing), so the
    transmission function for that chunk is still unitary, and Fresnel
    propagation is always unitary - energy conservation must hold
    regardless of how slices are grouped.
    """
    torch.manual_seed(0)
    potential = torch.randn(1, 10, 8, 8, dtype=torch.complex64).real.to(torch.complex64)
    exit_wave = multislice(potential, pixel_size=1.0, energy=300.0, n_slices=3)
    output_energy = exit_wave.abs().pow(2).sum()
    expected_energy = torch.tensor(8.0 * 8.0)
    assert torch.allclose(output_energy, expected_energy, rtol=1e-4)


def test_multislice_grouped_differs_from_ungrouped():
    """Coarser grouping is a different (and less accurate) approximation.

    Grouping 10 slices into 3 chunks sums potential within each chunk
    before a single transmission+propagation step, which is physically
    different from stepping through all 10 slices individually - the
    two exit waves should not coincide.
    """
    torch.manual_seed(0)
    potential = torch.randn(1, 10, 8, 8, dtype=torch.complex64).real.to(torch.complex64)
    ungrouped = multislice(potential, pixel_size=1.0, energy=300.0)
    grouped = multislice(potential, pixel_size=1.0, energy=300.0, n_slices=3)
    assert not torch.allclose(ungrouped, grouped)


def test_multislice_grouped_pairs_with_empty_neighbor_match_full_resolution():
    """Merging (content, vacuum) pairs must not change the exit wave.

    16 slices grouped into 8 chunks of size 2 pairs up (0,1), (2,3),
    ..., (14,15). With content on every even slice and zero on every
    odd slice, each chunk's summed potential equals its even slice
    unchanged, so n_slices=16 and n_slices=8 must agree exactly.
    """
    torch.manual_seed(0)
    potential = torch.zeros(1, 16, 16, 16, dtype=torch.complex64)
    even_potential = torch.randn(8, 16, 16).to(torch.complex64)
    potential[0, 0::2] = even_potential

    wave_16 = multislice(potential, pixel_size=1.0, energy=300.0, n_slices=16)
    wave_8 = multislice(potential, pixel_size=1.0, energy=300.0, n_slices=8)
    assert torch.allclose(wave_16, wave_8, atol=1e-5)


@pytest.mark.parametrize("n_slices", [0, -1, 11])
def test_multislice_rejects_invalid_n_slices(n_slices):
    """n_slices must be in (0, Z]; 0, negative, and > Z are invalid."""
    potential = torch.zeros(1, 10, 8, 8, dtype=torch.complex64)
    with pytest.raises(ValueError):
        multislice(potential, pixel_size=1.0, energy=300.0, n_slices=n_slices)
