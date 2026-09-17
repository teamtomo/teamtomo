"""Tests for shared simulation geometry helpers."""

import numpy as np
import pandas as pd
import torch
from torch_structure_manipulation import df_to_atomxyz

from torch_fit_in_map import AlignmentResult, apply_alignment_to_structure
from torch_fit_in_map._geometry import (
    coords_xyz_from_simulation_voxels,
    coords_xyz_to_simulation_voxels,
    simulation_box_center_angstroms,
)


def _make_atoms() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-10, 10, size=(8, 3))
    return pd.DataFrame(
        {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "element": ["C"] * len(xyz),
        }
    )


def test_coords_xyz_simulation_voxel_round_trip():
    """Forward and inverse voxel conversion recover the original coordinates."""
    atoms = _make_atoms()
    box = 48
    px = 1.75
    coords = df_to_atomxyz(atoms)
    centroid = coords.mean(dim=0)

    vox = coords_xyz_to_simulation_voxels(coords, centroid, box, px)
    recovered = coords_xyz_from_simulation_voxels(vox, centroid, box, px)
    torch.testing.assert_close(coords, recovered)


def test_identity_alignment_round_trip_through_simulation_geometry():
    """Identity alignment recentres atoms at the simulation box centre."""
    atoms = _make_atoms()
    box = 40
    px = 2.0
    result = AlignmentResult(torch.eye(3), torch.zeros(3), score=1.0)

    out = apply_alignment_to_structure(
        atoms, result, pixel_size=px, box_shape=(box, box, box), sim_box_size=box
    )

    expected_centre = simulation_box_center_angstroms(box, px)
    centroid = out[["x", "y", "z"]].to_numpy().mean(0)
    np.testing.assert_allclose(centroid, [expected_centre] * 3, atol=1e-3)
