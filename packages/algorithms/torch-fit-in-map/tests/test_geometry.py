"""Tests for shared simulation geometry helpers."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch_structure_manipulation import df_to_atomxyz

from torch_fit_in_map import (
    AlignmentResult,
    apply_alignment_to_structure,
    crop_or_pad_to_shape,
)
from torch_fit_in_map._geometry import (
    coords_xyz_from_simulation_voxels,
    coords_xyz_to_simulation_voxels,
    crop_start_zyx,
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


@pytest.mark.parametrize("sim_box_size", [40, 41, 45, 47, 48, 50, 52, 53])
def test_crop_start_matches_crop_or_pad_to_shape(sim_box_size):
    """Simulation voxel p sits at p - crop_start after crop_or_pad_to_shape.

    The non-cubic target crops some axes and pads others, with even and odd
    size differences.
    """
    box_shape = (48, 45, 50)
    start = crop_start_zyx(sim_box_size, box_shape)
    for axis in range(3):
        # voxels hold 1 + their index along `axis`; padding stays 0
        shape = [1, 1, 1]
        shape[axis] = sim_box_size
        index = torch.arange(1, sim_box_size + 1, dtype=torch.float32).view(shape)
        volume = index.expand(sim_box_size, sim_box_size, sim_box_size).contiguous()

        out = crop_or_pad_to_shape(volume, box_shape).movedim(axis, 0)
        # through the middle of the other axes, which is never padding
        line = out[:, out.shape[1] // 2, out.shape[2] // 2]

        out_index = torch.nonzero(line).squeeze(-1)
        assert len(out_index) > 0
        torch.testing.assert_close(line[out_index] - 1, out_index.float() + start[axis])
