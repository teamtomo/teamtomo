"""Tests for structure_transforms module."""

import numpy as np
import pandas as pd
import pytest
import torch

from torch_structure_manipulation.structure_transforms import (
    apply_rotation,
    apply_rotation_to_coords,
    apply_translation,
    apply_translation_to_coords,
    ball_query_atoms,
    calculate_center_from_tensors,
    center_structure,
    center_structure_from_coords,
    create_rotation_matrix_from_euler,
    df_to_atomzyx,
    find_atoms_in_ball,
    get_nucleic_acid_residues,
    get_protein_residues,
    remove_sidechains,
    separate_protein_rna,
)


class TestDfToAtomzyx:
    """Tests for df_to_atomzyx function."""

    def test_basic_extraction(self):
        """Test basic coordinate extraction."""
        df = pd.DataFrame(
            {
                "z": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "x": [7.0, 8.0, 9.0],
            }
        )
        result = df_to_atomzyx(df)
        expected = torch.tensor([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
        assert torch.allclose(result, expected.float())


class TestCenterStructure:
    """Tests for center_structure functions."""

    def test_center_at_origin_zyx(self):
        """Test centering at origin with zyx coordinates."""
        df = pd.DataFrame(
            {
                "z": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "x": [7.0, 8.0, 9.0],
            }
        )
        result = center_structure(df, zyx=True)
        centered_coords = result[["z", "y", "x"]].values
        # Mean should be at origin
        assert np.allclose(centered_coords.mean(axis=0), [0, 0, 0])

    def test_center_at_origin_xyz(self):
        """Test centering at origin with xyz coordinates."""
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "z": [7.0, 8.0, 9.0],
            }
        )
        result = center_structure(df, zyx=False)
        centered_coords = result[["x", "y", "z"]].values
        # Mean should be at origin
        assert np.allclose(centered_coords.mean(axis=0), [0, 0, 0])

    def test_center_at_specific_point_zyx(self):
        """Test centering at a specific point with zyx coordinates."""
        df = pd.DataFrame(
            {
                "z": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "x": [7.0, 8.0, 9.0],
            }
        )
        center_point = (2.0, 5.0, 8.0)  # (z, y, x)
        result = center_structure(df, center_point=center_point, zyx=True)
        centered_coords = result[["z", "y", "x"]].values
        # Mean should be at center_point
        assert np.allclose(centered_coords.mean(axis=0), center_point)

    def test_center_at_specific_point_xyz(self):
        """Test centering at a specific point with xyz coordinates."""
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [4.0, 5.0, 6.0],
                "z": [7.0, 8.0, 9.0],
            }
        )
        center_point = (2.0, 5.0, 8.0)  # (x, y, z)
        result = center_structure(df, center_point=center_point, zyx=False)
        centered_coords = result[["x", "y", "z"]].values
        # Mean should be at center_point
        assert np.allclose(centered_coords.mean(axis=0), center_point)

    def test_center_structure_from_coords_zyx(self):
        """Test centering from coordinate tensor with zyx coordinates."""
        coords = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = center_structure_from_coords(coords, zyx=True)
        # Mean should be at origin
        assert torch.allclose(result.mean(dim=0), torch.zeros(3))

    def test_center_structure_from_coords_xyz(self):
        """Test centering from coordinate tensor with xyz coordinates."""
        coords = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = center_structure_from_coords(coords, zyx=False)
        # Mean should be at origin
        assert torch.allclose(result.mean(dim=0), torch.zeros(3))

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["z", "y", "x"])
        result = center_structure(df, zyx=True)
        assert len(result) == 0


class TestCalculateCenterFromTensors:
    """Tests for calculate_center_from_tensors function."""

    def test_geometric_center(self):
        """Test geometric center calculation."""
        coords = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        center = calculate_center_from_tensors(coords, use_center_of_mass=False)
        expected = torch.tensor([4.0, 5.0, 6.0])  # Mean of coordinates
        assert torch.allclose(center, expected)

    def test_center_of_mass(self):
        """Test center of mass calculation."""
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        masses = torch.tensor([1.0, 2.0, 3.0])
        center = calculate_center_from_tensors(
            coords, use_center_of_mass=True, masses=masses
        )
        # Weighted average: (0*1 + 1*2 + 2*3) / 6 = 8/6 = 4/3
        expected = torch.tensor([4.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0])
        assert torch.allclose(center, expected)


class TestRotation:
    """Tests for rotation functions."""

    def test_create_rotation_matrix_90_deg_z(self):
        """Test 90-degree rotation around z-axis."""
        angles = torch.tensor([0.0, 0.0, 90.0])  # ZYZ order
        R = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
        # 90 deg rotation around z: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert torch.allclose(R, expected, atol=1e-6)

    def test_apply_rotation_to_coords_zyx(self):
        """Test rotation application to zyx coordinates."""
        # Create 90-degree rotation around z-axis
        angles = torch.tensor([0.0, 0.0, 90.0])
        R = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
        # Point at [0, 1, 0] in zyx (z=0, y=1, x=0)
        # In xyz: [0, 1, 0], after 90 deg z rotation: [-1, 0, 0]
        # Back to zyx: [0, 0, -1]
        coords = torch.tensor([[0.0, 1.0, 0.0]])
        rotated = apply_rotation_to_coords(coords, R, zyx=True)
        expected = torch.tensor([[0.0, 0.0, -1.0]])
        assert torch.allclose(rotated, expected, atol=1e-5)

    def test_apply_rotation_to_coords_xyz(self):
        """Test rotation application to xyz coordinates."""
        # Create 90-degree rotation around z-axis
        angles = torch.tensor([0.0, 0.0, 90.0])
        R = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
        # Point at [0, 1, 0] in xyz, after 90 deg z rotation: [-1, 0, 0]
        coords = torch.tensor([[0.0, 1.0, 0.0]])
        rotated = apply_rotation_to_coords(coords, R, zyx=False)
        expected = torch.tensor([[-1.0, 0.0, 0.0]])
        assert torch.allclose(rotated, expected, atol=1e-5)

    def test_apply_rotation_with_center_zyx(self):
        """Test rotation around a center point with zyx coordinates."""
        angles = torch.tensor([0.0, 0.0, 90.0])
        R = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
        coords = torch.tensor([[1.0, 1.0, 0.0]])
        center_point = (1.0, 1.0, 0.0)  # Rotate around itself (z, y, x)
        rotated = apply_rotation_to_coords(
            coords, R, center_point=center_point, zyx=True
        )
        # Should return to original position
        assert torch.allclose(rotated, coords, atol=1e-5)

    def test_apply_rotation_with_center_xyz(self):
        """Test rotation around a center point with xyz coordinates."""
        angles = torch.tensor([0.0, 0.0, 90.0])
        R = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
        coords = torch.tensor([[1.0, 1.0, 0.0]])
        center_point = (1.0, 1.0, 0.0)  # Rotate around itself (x, y, z)
        rotated = apply_rotation_to_coords(
            coords, R, center_point=center_point, zyx=False
        )
        # Should return to original position
        assert torch.allclose(rotated, coords, atol=1e-5)

    def test_apply_rotation_dataframe_zyx(self):
        """Test rotation on DataFrame with zyx coordinates."""
        df = pd.DataFrame(
            {
                "z": [0.0, 1.0],
                "y": [1.0, 0.0],
                "x": [0.0, 0.0],
            }
        )
        angles = torch.tensor([0.0, 0.0, 90.0])
        R = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
        result = apply_rotation(df, R, zyx=True)
        # Check that coordinates changed
        original_coords = df[["z", "y", "x"]].values
        rotated_coords = result[["z", "y", "x"]].values
        assert not np.allclose(rotated_coords, original_coords)

    def test_apply_rotation_dataframe_xyz(self):
        """Test rotation on DataFrame with xyz coordinates."""
        df = pd.DataFrame(
            {
                "x": [0.0, 0.0],
                "y": [1.0, 0.0],
                "z": [0.0, 1.0],
            }
        )
        angles = torch.tensor([0.0, 0.0, 90.0])
        R = create_rotation_matrix_from_euler(angles, order="ZYZ", degrees=True)
        result = apply_rotation(df, R, zyx=False)
        # Check that coordinates changed
        original_coords = df[["x", "y", "z"]].values
        rotated_coords = result[["x", "y", "z"]].values
        assert not np.allclose(rotated_coords, original_coords)


class TestTranslation:
    """Tests for translation functions."""

    def test_apply_translation_to_coords(self):
        """Test translation of coordinate tensor."""
        coords = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        translation = (1.0, 2.0, 3.0)  # Order matches coordinate order
        result = apply_translation_to_coords(coords, translation)
        expected = torch.tensor([[2.0, 4.0, 6.0], [5.0, 7.0, 9.0]])
        assert torch.allclose(result, expected)

    def test_apply_translation_dataframe_zyx(self):
        """Test translation on DataFrame with zyx coordinates."""
        df = pd.DataFrame(
            {
                "z": [1.0, 2.0],
                "y": [3.0, 4.0],
                "x": [5.0, 6.0],
            }
        )
        translation = (1.0, 1.0, 1.0)  # (dz, dy, dx)
        result = apply_translation(df, translation, zyx=True)
        expected = df[["z", "y", "x"]].values + np.array([1.0, 1.0, 1.0])
        result_coords = result[["z", "y", "x"]].values
        assert np.allclose(result_coords, expected)

    def test_apply_translation_dataframe_xyz(self):
        """Test translation on DataFrame with xyz coordinates."""
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
                "z": [5.0, 6.0],
            }
        )
        translation = (1.0, 1.0, 1.0)  # (dx, dy, dz)
        result = apply_translation(df, translation, zyx=False)
        expected = df[["x", "y", "z"]].values + np.array([1.0, 1.0, 1.0])
        result_coords = result[["x", "y", "z"]].values
        assert np.allclose(result_coords, expected)


class TestBallQueryAtoms:
    """Tests for ball_query_atoms function."""

    def test_ball_query_atoms_tensor_zyx(self):
        """Test ball query from tensor with zyx coordinates."""
        atomzyx = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        center = (0.0, 0.0, 0.0)  # (z, y, x)
        radius = 1.5
        inside_mask = ball_query_atoms(atomzyx, center, radius, zyx=True)
        # First two points should be inside, last two outside
        assert inside_mask.tolist() == [True, True, False, False]

    def test_ball_query_atoms_dataframe_zyx(self):
        """Test ball query from DataFrame with zyx coordinates."""
        df = pd.DataFrame(
            {
                "z": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 0.0, 0.0, 0.0],
                "x": [0.0, 0.0, 0.0, 0.0],
            }
        )
        center = (0.0, 0.0, 0.0)  # (z, y, x)
        radius = 1.5
        inside_mask = ball_query_atoms(df, center, radius, zyx=True)
        assert inside_mask.tolist() == [True, True, False, False]

    def test_ball_query_atoms_dataframe_xyz(self):
        """Test ball query from DataFrame with xyz coordinates."""
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0, 0.0],
            }
        )
        center = (0.0, 0.0, 0.0)  # (x, y, z) when zyx=False
        radius = 1.5
        inside_mask = ball_query_atoms(df, center, radius, zyx=False)
        # First two points should be inside, last two outside
        assert inside_mask.tolist() == [True, True, False, False]


class TestFindAtomsInBall:
    """Tests for find_atoms_in_ball function."""

    def test_find_atoms_in_ball_zyx(self):
        """Test find_atoms_in_ball with zyx coordinates."""
        df = pd.DataFrame(
            {
                "z": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 0.0, 0.0, 0.0],
                "x": [0.0, 0.0, 0.0, 0.0],
                "element": ["C", "C", "C", "C"],
            }
        )
        center = (0.0, 0.0, 0.0)  # (z, y, x)
        radius = 1.5
        inside_df, outside_df = find_atoms_in_ball(df, center, radius, zyx=True)
        assert len(inside_df) == 2
        assert len(outside_df) == 2
        assert set(inside_df.index) == {0, 1}
        assert set(outside_df.index) == {2, 3}

    def test_find_atoms_in_ball_xyz(self):
        """Test find_atoms_in_ball with xyz coordinates."""
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0, 0.0],
                "element": ["C", "C", "C", "C"],
            }
        )
        center = (0.0, 0.0, 0.0)  # (x, y, z) when zyx=False
        radius = 1.5
        inside_df, outside_df = find_atoms_in_ball(df, center, radius, zyx=False)
        assert len(inside_df) == 2
        assert len(outside_df) == 2
        assert set(inside_df.index) == {0, 1}
        assert set(outside_df.index) == {2, 3}


class TestRemoveSidechains:
    """Tests for remove_sidechains function."""

    def test_remove_sidechains_default(self):
        """Test removing sidechains with default backbone atoms."""
        df = pd.DataFrame(
            {
                "atom": ["N", "CA", "C", "O", "CB", "CG"],
                "z": [0.0] * 6,
                "y": [0.0] * 6,
                "x": [0.0] * 6,
            }
        )
        result = remove_sidechains(df)
        # Should keep N, CA, C, O, but not CB, CG
        assert len(result) == 4
        assert set(result["atom"]) == {"N", "CA", "C", "O"}

    def test_remove_sidechains_custom(self):
        """Test removing sidechains with custom backbone atoms."""
        df = pd.DataFrame(
            {
                "atom": ["N", "CA", "C", "O", "CB"],
                "z": [0.0] * 5,
                "y": [0.0] * 5,
                "x": [0.0] * 5,
            }
        )
        keep_atoms = ["N", "CA", "CB"]
        result = remove_sidechains(df, keep_backbone_atoms=keep_atoms)
        assert len(result) == 3
        assert set(result["atom"]) == {"N", "CA", "CB"}


class TestResidueFunctions:
    """Tests for residue-related functions."""

    def test_get_protein_residues(self):
        """Test getting protein residues."""
        residues = get_protein_residues()
        assert isinstance(residues, set)
        assert "ALA" in residues
        assert "GLY" in residues
        assert "MSE" in residues  # Non-standard amino acid

    def test_get_nucleic_acid_residues(self):
        """Test getting nucleic acid residues."""
        residues = get_nucleic_acid_residues()
        assert isinstance(residues, set)
        assert "A" in residues
        assert "U" in residues
        assert "DA" in residues  # DNA variant

    def test_separate_protein_rna(self):
        """Test separating protein and RNA components."""
        df = pd.DataFrame(
            {
                "residue": ["ALA", "GLY", "A", "U", "UNK"],
                "z": [0.0] * 5,
                "y": [0.0] * 5,
                "x": [0.0] * 5,
            }
        )
        with pytest.warns(UserWarning, match="unrecognized residue types"):
            protein_df, nucleic_df = separate_protein_rna(df)
        assert len(protein_df) == 2
        assert len(nucleic_df) == 2
        assert set(protein_df["residue"]) == {"ALA", "GLY"}
        assert set(nucleic_df["residue"]) == {"A", "U"}
