"""Tests for the tensor + DataFrame public API."""

import numpy as np
import pandas as pd
import torch


class _GaussianSimulator:
    """Minimal density simulator taking an ``atoms`` DataFrame (for tests only)."""

    sigma_A: float = 3.0

    def simulate(self, atoms, pixel_size, box_size, device=None, config=None):
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


def test_fit_structure_in_map_accepts_dataframe():
    """fit_structure_in_map accepts a DataFrame and honors a custom simulator."""
    from torch_fit_in_map import ExhaustiveSearchConfig, fit_structure_in_map

    atoms = _make_atoms()
    sim = _GaussianSimulator()
    box = 24
    px = 2.0
    reference = sim.simulate(atoms, px, box)

    result = fit_structure_in_map(
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


def test_fit_map_in_structure_accepts_dataframe():
    """fit_map_in_structure accepts a reference-atoms DataFrame."""
    from torch_fit_in_map import ExhaustiveSearchConfig, fit_map_in_structure

    atoms = _make_atoms()
    sim = _GaussianSimulator()
    box = 24
    px = 2.0
    mobile = sim.simulate(atoms, px, box)

    result = fit_map_in_structure(
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


def test_apply_alignment_to_structure_preserves_dataframe_and_distances():
    """The structure transform preserves metadata and pairwise distances."""
    from torch_fit_in_map import AlignmentResult, apply_alignment_to_structure

    atoms = _make_atoms()
    atoms["label"] = [f"atom-{i}" for i in range(len(atoms))]
    box = 32
    px = 1.5

    # A non-trivial rotation about z + a translation.
    theta = np.deg2rad(30.0)
    c, s = np.cos(theta), np.sin(theta)
    R = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]], dtype=torch.float32)
    t = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    result = AlignmentResult(R, t, score=1.0)

    out = apply_alignment_to_structure(
        atoms, result, pixel_size=px, box_shape=(box, box, box)
    )

    def _pdist(df):
        p = df[["x", "y", "z"]].to_numpy()
        return np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)

    np.testing.assert_allclose(_pdist(atoms), _pdist(out), atol=1e-4)
    assert list(out.columns) == list(atoms.columns)
    assert out["label"].equals(atoms["label"])


def test_apply_alignment_to_structure_identity_centres_in_box():
    """With identity rotation and zero shift, atoms are centred at the box centre."""
    from torch_fit_in_map import AlignmentResult, apply_alignment_to_structure

    atoms = _make_atoms()
    box = 40
    px = 2.0
    result = AlignmentResult(torch.eye(3), torch.zeros(3), score=1.0)

    out = apply_alignment_to_structure(
        atoms, result, pixel_size=px, box_shape=(box, box, box)
    )

    box_centre_A = (box - 1) / 2.0 * px
    centroid = out[["x", "y", "z"]].to_numpy().mean(0)
    np.testing.assert_allclose(centroid, [box_centre_A] * 3, atol=1e-3)


def test_default_workspace_simulator_produces_zyx_volume_and_fits():
    """The production simulator generates finite ZYX data usable by fitting."""
    import torch_calculate_electrostatic_potential
    from torch_calculate_electrostatic_potential import (
        GridConfig,
        potential_from_structure_3d,
    )
    from torch_structure_manipulation import AtomicStructure

    from torch_fit_in_map import ExhaustiveSearchConfig, fit_structure_in_map
    from torch_fit_in_map._simulate import DEFAULT_POTENTIAL_SIMULATOR
    from torch_calculate_electrostatic_potential import default_sublattice_radius

    assert "torch_calculate_electrostatic_potential" in (
        torch_calculate_electrostatic_potential.__file__ or ""
    )

    atoms = pd.DataFrame(
        {
            "x": [-2.0, 1.0, 3.0],
            "y": [1.0, -2.0, 2.0],
            "z": [-1.0, 2.5, 0.5],
            "element": ["C", "N", "O"],
        }
    )
    box = 16
    pixel_size = 2.0
    volume = DEFAULT_POTENTIAL_SIMULATOR.simulate(atoms, pixel_size, box)

    assert volume.shape == (box, box, box)
    assert torch.isfinite(volume).all()
    assert volume.abs().sum() > 0

    structure = AtomicStructure.from_dataframe(atoms, device=volume.device)
    center_zyx = torch.full((3,), (box - 1) / 2 * pixel_size, device=volume.device)
    structure = structure.with_positions(
        structure.positions_zyx - structure.positions_zyx.mean(0) + center_zyx
    )
    grid = GridConfig.from_grid_shape_and_voxel_size(
        (box, box, box),
        (pixel_size, pixel_size, pixel_size),
        center_zyx=center_zyx,
        sublattice_radius=default_sublattice_radius(pixel_size),
        device=volume.device,
    )
    expected_zyx = potential_from_structure_3d(structure, grid)
    torch.testing.assert_close(volume, expected_zyx)

    result = fit_structure_in_map(
        mobile_atoms=atoms,
        reference_map=volume,
        pixel_size_angstroms=pixel_size,
        box_size=box,
        exhaustive_config=ExhaustiveSearchConfig(
            angular_step_degrees=90.0,
            pixel_size_angstroms=pixel_size,
        ),
        gradient_config=None,
        verbose=False,
    )
    assert np.isfinite(result.score)


def test_default_simulator_supports_bonded_scattering_factors():
    """Bonded Peng factors can be selected via PotentialSimulatorConfig."""
    from torch_fit_in_map import PotentialSimulatorConfig
    from torch_fit_in_map._simulate import DEFAULT_POTENTIAL_SIMULATOR

    atoms = pd.DataFrame(
        {
            "x": [0.0, 1.2],
            "y": [0.0, -0.4],
            "z": [0.0, 0.7],
            "element": ["C", "O"],
            "atom": ["CA", "O"],
            "bonded_environments": ["C(HHCC)", "O(C, amide)"],
            "molecule_type": ["protein", "protein"],
        }
    )
    config = PotentialSimulatorConfig(
        scattering_factors="peng_bonded", bonded_fallback="error"
    )
    volume = DEFAULT_POTENTIAL_SIMULATOR.simulate(
        atoms, pixel_size=2.0, box_size=12, config=config
    )
    assert volume.shape == (12, 12, 12)
    assert torch.isfinite(volume).all()
    assert volume.abs().sum() > 0
