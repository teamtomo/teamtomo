"""Tests for continuum solvent geometry and potentials."""

import math

import pandas as pd
import pytest
import torch
from torch_structure_manipulation import AtomicStructure

from torch_calculate_electrostatic_potential import (
    DEFAULT_ICE_POTENTIAL_V,
    GridConfig,
    constant_solvent_potential,
    distance_to_surface,
    potential_from_structure_3d,
    shang_sigworth_density,
    shang_sigworth_solvent_potential,
    solvent_occupancy_from_structure_3d,
    solvent_potential_from_structure_3d,
    solvated_potential_from_structure_3d,
    vdw_probe_occupancy,
    vdw_radii_for_atomic_numbers,
)


def _grid(shape=(21, 21, 21), voxel=1.0):
    return GridConfig.from_grid_shape_and_voxel_size(
        shape,
        (voxel,) * 3,
        center_zyx=(0.0, 0.0, 0.0),
        sublattice_radius=5.0,
        equal_length=False,
    )


def _carbon_structure():
    return AtomicStructure.from_dataframe(
        pd.DataFrame(
            {
                "x": [0.0],
                "y": [0.0],
                "z": [0.0],
                "element": ["C"],
                "atom": ["C"],
                "b_isotropic": [20.0],
                "occupancy": [1.0],
            }
        )
    )


def test_vdw_radii_known_and_unknown():
    z = torch.tensor([6, 8, 26, 99], dtype=torch.int64)
    radii = vdw_radii_for_atomic_numbers(z)
    assert radii[0].item() == pytest.approx(1.7)
    assert radii[1].item() == pytest.approx(1.52)
    assert radii[2].item() < 0
    assert radii[3].item() < 0


def test_occupancy_excludes_vdw_plus_probe_for_carbon():
    structure = _carbon_structure()
    grid = _grid()
    probe_radius = 1.4
    vdw_c = 1.7
    occupancy = solvent_occupancy_from_structure_3d(
        structure, grid, probe_radius=probe_radius, r_asymptote=7.5
    )
    # Center voxel should be protein (excluded).
    center = tuple(s // 2 for s in occupancy.shape)
    assert occupancy[center].item() == 0.0

    # A voxel just outside VdW+probe should be solvent.
    excluded_radius = vdw_c + probe_radius
    # Move ~ceil(excluded_radius)+1 voxels along x from center.
    offset = int(math.ceil(excluded_radius)) + 1
    assert occupancy[center[0], center[1], center[2] + offset].item() == 1.0

    # A voxel inside the excluded ball (along axis, within floor(excluded_radius)-1).
    inside = max(1, int(math.floor(excluded_radius)) - 1)
    assert occupancy[center[0], center[1], center[2] + inside].item() == 0.0


def test_shang_sigworth_density_bulk_and_interior():
    r_bulk = torch.tensor([10.0, 20.0])
    z_bulk = torch.tensor([8, 8], dtype=torch.int64)
    rho_bulk = shang_sigworth_density(r_bulk, z_bulk)
    assert torch.allclose(rho_bulk, torch.ones_like(rho_bulk), atol=0.05)

    r_in = torch.tensor([-2.0])
    z_in = torch.tensor([8], dtype=torch.int64)
    rho_in = shang_sigworth_density(r_in, z_in)
    assert rho_in.item() < 0.3


def test_polar_vs_nonpolar_density_differ():
    r = torch.tensor([1.5, 1.5])
    z = torch.tensor([6, 8], dtype=torch.int64)  # C nonpolar, O polar
    rho = shang_sigworth_density(r, z)
    assert not torch.isclose(rho[0], rho[1])


def test_constant_solvent_potential_scales_occupancy():
    occupancy = torch.tensor([0.0, 1.0, 0.5])
    pot = constant_solvent_potential(occupancy, ice_potential_V=3.6)
    assert torch.allclose(pot, occupancy * 3.6)


def test_shang_sigworth_solvent_potential_zeros_interior():
    dist = torch.tensor([-1.0, 0.0, 5.0])
    z = torch.tensor([8, 8, 8], dtype=torch.int64)
    pot = shang_sigworth_solvent_potential(
        dist, z, ice_potential_V=3.6, probe_radius=1.4
    )
    assert pot[0].item() == 0.0
    assert pot[1].item() == 0.0
    assert pot[2].item() > 3.0  # near bulk MIP


def test_compose_solvated_equals_dry_plus_solvent():
    structure = _carbon_structure()
    grid = _grid(shape=(15, 15, 15))
    dry = potential_from_structure_3d(structure, grid)
    solvent = solvent_potential_from_structure_3d(
        structure, grid, model="constant", ice_potential_V=DEFAULT_ICE_POTENTIAL_V
    )
    solvated = solvated_potential_from_structure_3d(
        structure,
        grid,
        model_water_potential=True,
        solvent_model="constant",
    )
    assert torch.allclose(solvated, dry + solvent)

    none = solvated_potential_from_structure_3d(
        structure, grid, model_water_potential=False
    )
    assert torch.equal(none, dry)


def test_shang_sigworth_is_default_when_modeling_water():
    structure = _carbon_structure()
    grid = _grid(shape=(15, 15, 15))
    default_water = solvated_potential_from_structure_3d(
        structure, grid, model_water_potential=True
    )
    explicit_ss = solvated_potential_from_structure_3d(
        structure,
        grid,
        model_water_potential=True,
        solvent_model="shang_sigworth",
    )
    assert torch.equal(default_water, explicit_ss)
    assert default_water.shape == tuple(int(s) for s in grid.grid_shape.tolist())
    assert default_water.dtype == grid.dtype
    assert default_water.device == grid.device
    # Far corner should be near bulk ice MIP (atomic potential ~0 there).
    assert default_water[0, 0, 0].item() > 2.5


def test_distance_to_surface_device_dtype():
    positions = torch.tensor([[0.0, 0.0, 0.0]])
    z = torch.tensor([6], dtype=torch.int64)
    grid = _grid(shape=(9, 9, 9))
    dist, nearest = distance_to_surface(positions, z, grid)
    assert dist.dtype == grid.dtype
    assert dist.device == grid.device
    assert nearest.dtype == torch.int64
    center = tuple(s // 2 for s in dist.shape)
    assert nearest[center].item() == 6
    # At atom center, surface distance ≈ -VdW(C).
    assert dist[center].item() == pytest.approx(-1.7, abs=0.6)


def test_vdw_probe_occupancy_from_dist_map():
    dist = torch.tensor([-1.0, 1.0, 2.0])
    occ = vdw_probe_occupancy(dist, probe_radius=1.4)
    assert torch.equal(occ, torch.tensor([0.0, 0.0, 1.0]))
