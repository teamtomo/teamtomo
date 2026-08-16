"""Tests for the tensor + DataFrame public API (no file I/O, no espcalculator)."""

import numpy as np
import pandas as pd
import torch


class _GaussianSimulator:
    """Minimal density simulator taking an ``atoms`` DataFrame (for tests only)."""

    sigma_A: float = 3.0

    def simulate(self, atoms, pixel_size, box_size, device=None):
        coords_zyx = atoms[["z", "y", "x"]].to_numpy(dtype=np.float32)
        centroid = coords_zyx.mean(0)
        box_centre_A = (box_size - 1) / 2.0 * pixel_size
        vox = torch.tensor(
            (coords_zyx - centroid + box_centre_A) / pixel_size, dtype=torch.float32
        )
        sigma_vox = self.sigma_A / pixel_size
        grid = torch.arange(box_size, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(grid, grid, grid, indexing="ij")
        coords = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
        diff = coords.unsqueeze(0) - vox.unsqueeze(1)
        d2 = diff.pow(2).sum(-1)
        density = torch.exp(-d2 / (2 * sigma_vox**2)).sum(0)
        return density.reshape(box_size, box_size, box_size).to(device or "cpu")


def _make_atoms() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-10, 10, size=(12, 3))
    return pd.DataFrame(
        {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "element": ["C"] * len(xyz),
        }
    )


def test_fit_pdb_in_map_accepts_dataframe():
    """fit_pdb_in_map should accept an atoms DataFrame and honor a custom simulator."""
    from torch_fit_in_map import ExhaustiveSearchConfig, fit_pdb_in_map

    atoms = _make_atoms()
    sim = _GaussianSimulator()
    box = 24
    px = 2.0
    reference = sim.simulate(atoms, px, box)

    result = fit_pdb_in_map(
        mobile_atoms=atoms,
        reference_map=reference,
        pixel_size_angstroms=px,
        box_size=box,
        simulator=sim,
        exhaustive_config=ExhaustiveSearchConfig(
            angular_step_degrees=90.0, pixel_size_angstroms=px
        ),
        gradient_config=None,
        verbose=False,
    )
    # Identical simulated mobile/reference → near-identity recovery.
    assert torch.allclose(result.rotation_matrix.cpu(), torch.eye(3), atol=0.2)


def test_fit_map_in_pdb_accepts_dataframe():
    """fit_map_in_pdb should accept a reference atoms DataFrame."""
    from torch_fit_in_map import ExhaustiveSearchConfig, fit_map_in_pdb

    atoms = _make_atoms()
    sim = _GaussianSimulator()
    box = 24
    px = 2.0
    mobile = sim.simulate(atoms, px, box)

    result = fit_map_in_pdb(
        mobile_map=mobile,
        reference_atoms=atoms,
        pixel_size_angstroms=px,
        box_size=box,
        simulator=sim,
        exhaustive_config=ExhaustiveSearchConfig(
            angular_step_degrees=90.0, pixel_size_angstroms=px
        ),
        gradient_config=None,
        verbose=False,
    )
    assert isinstance(result.score, float)


def test_transform_atoms_preserves_pairwise_distances():
    """transform_atoms applies a rigid transform, preserving inter-atom distances."""
    from torch_fit_in_map import AlignmentResult, transform_atoms

    atoms = _make_atoms()
    box = 32
    px = 1.5

    # A non-trivial rotation about z + a translation.
    theta = np.deg2rad(30.0)
    c, s = np.cos(theta), np.sin(theta)
    R = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=torch.float32
    )
    t = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    result = AlignmentResult(R, t, score=1.0)

    out = transform_atoms(
        atoms, result, pixel_size=px, box_shape=(box, box, box)
    )

    def _pdist(df):
        p = df[["x", "y", "z"]].to_numpy()
        return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

    np.testing.assert_allclose(_pdist(atoms), _pdist(out), atol=1e-4)


def test_transform_atoms_identity_centres_in_box():
    """With identity rotation and zero shift, atoms are centred at the box centre."""
    from torch_fit_in_map import AlignmentResult, transform_atoms

    atoms = _make_atoms()
    box = 40
    px = 2.0
    result = AlignmentResult(torch.eye(3), torch.zeros(3), score=1.0)

    out = transform_atoms(atoms, result, pixel_size=px, box_shape=(box, box, box))

    box_centre_A = (box - 1) / 2.0 * px
    centroid = out[["x", "y", "z"]].to_numpy().mean(0)
    np.testing.assert_allclose(centroid, [box_centre_A] * 3, atol=1e-3)
