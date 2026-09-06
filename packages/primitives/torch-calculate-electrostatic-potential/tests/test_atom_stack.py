import gemmi
import pytest
import torch
from torch_structure_manipulation import AtomicStructure

from torch_calculate_electrostatic_potential import AtomStack, GridConfig


class TestAtomStackConstruction:
    def test_from_coords_and_names(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        stack = AtomStack.from_coords_and_names(coords, ["C", "O"])
        assert stack.num_atoms == 2
        assert stack.atomic_numbers.tolist() == [
            gemmi.Element("C").atomic_number,
            gemmi.Element("O").atomic_number,
        ]
        assert stack.atom_params_a.shape == (2, 5)
        assert stack.atom_params_b.shape == (2, 5)

    def test_from_coords_and_atomic_numbers(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        stack = AtomStack.from_coords_and_atomic_numbers(coords, torch.tensor([6, 8]))
        assert stack.num_atoms == 2
        assert stack.atomic_numbers.tolist() == [6, 8]

    def test_scalar_bfactor_and_occupancy_broadcast(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        stack = AtomStack.from_coords_and_names(
            coords, ["C", "N", "O"], atom_bfactors=15.0, atom_occupancies=0.9
        )
        assert stack.atom_bfactors.shape == ()
        assert stack.atom_occupancies.shape == ()

    def test_rejects_bad_position_shape(self):
        with pytest.raises(ValueError):
            AtomStack.from_coords_and_names(torch.zeros(3, 2), ["C", "N", "O"])

    def test_rejects_atomic_number_length_mismatch(self):
        coords = torch.zeros(3, 3)
        with pytest.raises(ValueError):
            AtomStack.from_coords_and_atomic_numbers(coords, torch.tensor([6, 7]))

    def test_rejects_bfactor_trailing_dim_mismatch(self):
        coords = torch.zeros(3, 3)
        with pytest.raises(ValueError):
            AtomStack.from_coords_and_names(
                coords, ["C", "N", "O"], atom_bfactors=torch.zeros(5)
            )

    def test_batched_positions(self):
        coords = torch.randn(4, 3, 3)
        stack = AtomStack.from_coords_and_names(coords, ["C", "N", "O"])
        assert stack.num_atoms == 3
        assert stack.atom_pos_zyx.shape == (4, 3, 3)

    def test_wraps_atomic_structure(self):
        stack = AtomStack.from_coords_and_names(torch.zeros(2, 3), ["C", "O"])
        assert isinstance(stack.structure, AtomicStructure)
        assert stack.structure.positions_zyx is stack.atom_pos_zyx
        assert stack.structure.b_factors is stack.atom_bfactors

    def test_batched_positions_broadcast_scalar_properties(self):
        coords = torch.randn(4, 3, 3)
        stack = AtomStack.from_coords_and_names(
            coords,
            ["C", "N", "O"],
            atom_bfactors=15.0,
            atom_occupancies=0.75,
        )
        assert stack.structure.batch_shape == (4,)
        assert stack.atom_bfactors.shape == ()
        assert stack.atom_occupancies.shape == ()


class TestAtomStackScatteringPotential:
    def _grids(self):
        grid3d = GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.0, 1.0),
            left_bottom_point=(-5.0, -5.0, -5.0),
            right_upper_point=(5.0, 5.0, 5.0),
            sublattice_radius=5.0,
        )
        grid2d = GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.0),
            left_bottom_point=(-5.0, -5.0),
            right_upper_point=(5.0, 5.0),
            sublattice_radius=5.0,
        )
        return grid3d, grid2d

    def test_to_scattering_potential_3d_matches_direct_call(self):
        from torch_calculate_electrostatic_potential.potential import (
            calculate_scattering_potential_3d,
        )

        coords = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.3, -0.7], [-1.2, 2.0, 0.5]])
        stack = AtomStack.from_coords_and_names(
            coords, ["C", "N", "O"], atom_bfactors=20.0
        )
        grid3d, _ = self._grids()

        via_stack = stack.to_scattering_potential_3d(grid3d)
        direct = calculate_scattering_potential_3d(
            stack.atom_pos_zyx,
            stack.atom_bfactors,
            stack.atom_params_a,
            stack.atom_params_b,
            grid3d,
            atom_occupancies=stack.atom_occupancies,
        )
        assert torch.equal(via_stack, direct)

    def test_to_scattering_potential_2d_drops_z(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.3, -0.7], [-1.2, 2.0, 0.5]])
        stack = AtomStack.from_coords_and_names(
            coords, ["C", "N", "O"], atom_bfactors=20.0
        )
        _, grid2d = self._grids()

        img = stack.to_scattering_potential_2d(grid2d)
        assert img.shape == tuple(grid2d.grid_shape.tolist())

        stack_shifted = AtomStack.from_coords_and_names(
            coords + torch.tensor([10.0, 0.0, 0.0]),
            ["C", "N", "O"],
            atom_bfactors=20.0,
        )
        img_shifted = stack_shifted.to_scattering_potential_2d(grid2d)
        assert torch.allclose(img, img_shifted)

    def test_occupancy_weighting_changes_output(self):
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.3, -0.7], [-1.2, 2.0, 0.5]])
        grid3d, _ = self._grids()
        full = AtomStack.from_coords_and_names(
            coords, ["C", "N", "O"], atom_bfactors=20.0
        )
        partial = AtomStack.from_coords_and_names(
            coords,
            ["C", "N", "O"],
            atom_bfactors=20.0,
            atom_occupancies=torch.tensor([1.0, 0.5, 0.8]),
        )
        assert not torch.allclose(
            full.to_scattering_potential_3d(grid3d),
            partial.to_scattering_potential_3d(grid3d),
        )

    def test_gradient_flows_through_atom_stack(self):
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.3, -0.7], [-1.2, 2.0, 0.5]],
            requires_grad=True,
        )
        stack = AtomStack.from_coords_and_names(
            pos, ["C", "N", "O"], atom_bfactors=20.0
        )
        grid3d, _ = self._grids()
        out = stack.to_scattering_potential_3d(grid3d)
        out.sum().backward()
        assert pos.grad is not None
        assert torch.isfinite(pos.grad).all()
        assert pos.grad.abs().sum() > 0

    def test_batched_ensemble(self):
        coords = torch.randn(5, 3, 3)
        stack = AtomStack.from_coords_and_names(
            coords, ["C", "N", "O"], atom_bfactors=20.0
        )
        grid3d, _ = self._grids()
        vol = stack.to_scattering_potential_3d(grid3d)
        assert vol.shape == (5, *grid3d.grid_shape.tolist())
