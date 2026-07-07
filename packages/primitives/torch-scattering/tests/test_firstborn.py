"""Tests for the high-level torch_scattering.firstborn() wrapper."""

import pytest
import torch
from torch_grid_utils import fftfreq_grid

from torch_scattering._core import fresnel_propagator, interaction_parameter
from torch_scattering.firstborn import firstborn


def test_firstborn_vacuum_leaves_plane_wave_unchanged():
    """Zero potential (vacuum) must leave a normally-incident plane wave unchanged.

    With no material, there is nothing to scatter the beam, so the exit
    wave must equal the incident unit-amplitude, zero-phase plane wave.
    """
    potential = torch.zeros(1, 5, 8, 8, dtype=torch.complex64)
    exit_wave = firstborn(potential, pixel_size=1.0, energy=300.0)
    assert torch.allclose(exit_wave, torch.ones_like(exit_wave), atol=1e-5)


def test_firstborn_matches_explicit_per_slice_loop():
    """The vectorized implementation must match an unvectorized reference loop.

    Independently builds each slice's Fresnel propagator one at a time
    (distance = remaining slices to the exit plane) and accumulates the
    scattered sum with a Python loop, rather than the batched propagator
    stack `firstborn` builds internally. Agreement checks that the batched
    distance assignment and broadcasting are wired up correctly.
    """
    torch.manual_seed(0)
    potential = torch.randn(1, 5, 8, 8, dtype=torch.complex64)
    pixel_size = 1.5
    energy = 300.0

    exit_wave = firstborn(potential, pixel_size=pixel_size, energy=energy)

    sigma = interaction_parameter(energy=energy)
    wavelength = 0.019687  # relativistic wavelength at 300 keV, Angstroms
    frequency_grid = fftfreq_grid(
        image_shape=(8, 8), rfft=False, spacing=pixel_size, norm=True
    )
    n_slices = potential.shape[-3]
    scattered_sum = torch.zeros(1, 8, 8, dtype=torch.complex64)
    for i in range(n_slices):
        distance = (n_slices - i) * pixel_size
        propagator = fresnel_propagator(
            frequency_grid, wavelength=wavelength, dz=distance
        )
        weighted = 1j * sigma * pixel_size * potential[:, i]
        scattered_sum += torch.fft.ifft2(torch.fft.fft2(weighted) * propagator)
    expected = 1 + scattered_sum

    assert torch.allclose(exit_wave, expected, atol=1e-5)


def test_firstborn_approximately_conserves_energy_for_weak_potential():
    """A weak real potential must leave the exit wave close to unit amplitude.

    First Born is the first-order (linearised) term of the Born series;
    for sigma*dz*V << 1 per slice, the scattered sum is small and
    ``1 + scattered`` has modulus close to 1 (approximate unitarity),
    unlike for strong potentials where the linear approximation breaks
    energy conservation.
    """
    torch.manual_seed(0)
    potential = 1e-4 * torch.randn(1, 5, 8, 8, dtype=torch.complex64).real.to(
        torch.complex64
    )
    exit_wave = firstborn(potential, pixel_size=1.0, energy=300.0)
    assert torch.allclose(exit_wave.abs(), torch.ones_like(exit_wave.abs()), atol=1e-3)


def test_firstborn_explicit_full_n_slices_matches_default():
    """n_slices == Z (explicit) must exactly match the n_slices=None default.

    Both should treat one true slice at a time, so the results must be
    bit-for-bit identical, not merely close.
    """
    torch.manual_seed(0)
    potential = torch.randn(1, 5, 8, 8, dtype=torch.complex64)
    default_wave = firstborn(potential, pixel_size=1.0, energy=300.0)
    explicit_wave = firstborn(potential, pixel_size=1.0, energy=300.0, n_slices=5)
    assert torch.equal(default_wave, explicit_wave)


def test_firstborn_grouped_differs_from_ungrouped():
    """Coarser grouping is a different (and less accurate) approximation.

    Grouping 10 slices into 3 chunks sums potential within each chunk
    before propagating it as one thicker slab, which is physically
    different from treating all 10 slices individually - the two exit
    waves should not coincide.
    """
    torch.manual_seed(0)
    potential = torch.randn(1, 10, 8, 8, dtype=torch.complex64)
    ungrouped = firstborn(potential, pixel_size=1.0, energy=300.0)
    grouped = firstborn(potential, pixel_size=1.0, energy=300.0, n_slices=3)
    assert not torch.allclose(ungrouped, grouped)


def test_firstborn_grouped_pairs_with_empty_neighbor_match_full_resolution():
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

    wave_16 = firstborn(potential, pixel_size=1.0, energy=300.0, n_slices=16)
    wave_8 = firstborn(potential, pixel_size=1.0, energy=300.0, n_slices=8)
    assert torch.allclose(wave_16, wave_8, atol=1e-5)


@pytest.mark.parametrize("n_slices", [0, -1, 11])
def test_firstborn_rejects_invalid_n_slices(n_slices):
    """n_slices must be in (0, Z]; 0, negative, and > Z are invalid."""
    potential = torch.zeros(1, 10, 8, 8, dtype=torch.complex64)
    with pytest.raises(ValueError):
        firstborn(potential, pixel_size=1.0, energy=300.0, n_slices=n_slices)
