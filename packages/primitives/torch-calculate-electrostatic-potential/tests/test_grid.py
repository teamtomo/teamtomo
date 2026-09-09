import pytest
import torch

from torch_calculate_electrostatic_potential import GridConfig, default_sublattice_radius


class TestGridConfigConstruction:
    def test_valid_3d(self):
        grid = GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.0, 1.0),
            left_bottom_point=(-5.0, -5.0, -5.0),
            right_upper_point=(5.0, 5.0, 5.0),
            sublattice_radius=4.0,
        )
        assert grid.ndim == 3
        assert grid.grid_shape.tolist() == [11, 11, 11]
        assert grid.grid_flat_size == 11**3

    def test_valid_2d(self):
        grid = GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.0),
            left_bottom_point=(-5.0, -5.0),
            right_upper_point=(5.0, 5.0),
            sublattice_radius=4.0,
        )
        assert grid.ndim == 2
        assert grid.grid_shape.tolist() == [11, 11]

    def test_rejects_bad_ndim(self):
        with pytest.raises(ValueError):
            GridConfig(grid_shape=(4, 4, 4, 4), voxel_size=(1.0, 1.0, 1.0, 1.0))
        with pytest.raises(ValueError):
            GridConfig(grid_shape=(4,), voxel_size=(1.0,))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            GridConfig(grid_shape=(4, 4, 4), voxel_size=(1.0, 1.0))

    def test_from_grid_shape_and_corner_points_derives_voxel_size(self):
        grid = GridConfig.from_grid_shape_and_corner_points(
            grid_shape=(11, 11, 11),
            left_bottom_point=(-5.0, -5.0, -5.0),
            right_upper_point=(5.0, 5.0, 5.0),
        )
        assert torch.allclose(grid.voxel_size, torch.tensor([1.0, 1.0, 1.0]))

    @pytest.mark.parametrize(
        ("shape", "voxel_size", "center_name", "center"),
        [
            ((5, 7, 9), (1.0, 2.0, 0.5), "center_zyx", (2.0, -1.0, 3.0)),
            ((6, 8), (1.5, 0.25), "center_yx", (-2.0, 4.0)),
        ],
    )
    def test_centered_grid_constructor(self, shape, voxel_size, center_name, center):
        grid = GridConfig.from_grid_shape_and_voxel_size(
            shape,
            voxel_size,
            **{center_name: center},
            equal_length=False,
        )
        midpoint = (grid.left_bottom_point + grid.right_upper_point) / 2
        assert torch.allclose(midpoint, torch.tensor(center))

    def test_center_name_must_match_dimensions(self):
        with pytest.raises(ValueError, match="center_zyx requires a 3D"):
            GridConfig.from_grid_shape_and_voxel_size(
                (5, 5), (1.0, 1.0), center_zyx=(0.0, 0.0, 0.0)
            )


class TestSquareOrCubic:
    def test_default_pads_anisotropic_3d_to_cubic(self):
        grid = GridConfig.from_grid_shape_and_voxel_size(
            grid_shape=(2, 3, 4), voxel_size=(1.0, 1.0, 1.0)
        )
        assert grid.grid_shape.tolist() == [4, 4, 4]
        assert torch.allclose(grid.left_bottom_point, torch.tensor([-1.0, 0.0, 0.0]))
        assert torch.allclose(grid.right_upper_point, torch.tensor([2.0, 3.0, 3.0]))

    def test_default_pads_anisotropic_2d_to_square(self):
        grid = GridConfig.from_grid_shape_and_voxel_size(
            grid_shape=(5, 9), voxel_size=(1.0, 1.0)
        )
        assert grid.grid_shape.tolist() == [9, 9]

    def test_equal_length_false_preserves_anisotropic_shape(self):
        grid = GridConfig.from_grid_shape_and_voxel_size(
            grid_shape=(2, 3, 4), voxel_size=(1.0, 1.0, 1.0), equal_length=False
        )
        assert grid.grid_shape.tolist() == [2, 3, 4]
        assert torch.allclose(grid.left_bottom_point, torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(grid.right_upper_point, torch.tensor([1.0, 2.0, 3.0]))

    def test_padding_is_symmetric_and_contains_original_span(self):
        grid_padded = GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.2, 0.9),
            left_bottom_point=(-5.0, -4.0, -6.0),
            right_upper_point=(5.0, 4.0, 6.0),
            sublattice_radius=4.5,
        )
        grid_raw = GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.2, 0.9),
            left_bottom_point=(-5.0, -4.0, -6.0),
            right_upper_point=(5.0, 4.0, 6.0),
            sublattice_radius=4.5,
            equal_length=False,
        )
        assert (grid_padded.grid_shape == grid_padded.grid_shape.max()).all()
        assert not (grid_raw.grid_shape == grid_raw.grid_shape.max()).all()
        # original span is contained within the padded span, symmetric to within 1 voxel
        lo_pad = grid_raw.left_bottom_point - grid_padded.left_bottom_point
        hi_pad = grid_padded.right_upper_point - grid_raw.right_upper_point
        assert (lo_pad >= -1e-5).all() and (hi_pad >= -1e-5).all()
        assert torch.allclose(
            lo_pad / grid_padded.voxel_size, hi_pad / grid_padded.voxel_size, atol=1.0
        )

    def test_already_cubic_grid_is_unaffected(self):
        kwargs = {
            "voxel_size": (1.0, 1.0, 1.0),
            "left_bottom_point": (-5.0, -5.0, -5.0),
            "right_upper_point": (5.0, 5.0, 5.0),
            "sublattice_radius": 4.0,
        }
        grid_default = GridConfig.from_voxel_size_and_corner_points(**kwargs)
        grid_raw = GridConfig.from_voxel_size_and_corner_points(
            **kwargs, equal_length=False
        )
        assert torch.equal(grid_default.grid_shape, grid_raw.grid_shape)
        assert torch.allclose(
            grid_default.left_bottom_point, grid_raw.left_bottom_point
        )
        assert torch.allclose(
            grid_default.right_upper_point, grid_raw.right_upper_point
        )

    def test_padded_grid_stencil_voxels_stay_in_bounds(self):
        grid = GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.2, 0.9),
            left_bottom_point=(-5.0, -4.0, -6.0),
            right_upper_point=(5.0, 4.0, 6.0),
            sublattice_radius=4.5,
        )
        points = torch.tensor([[0.3, -1.2, 5.9], [-4.9, 3.8, -5.9], [0.0, 0.0, 0.0]])
        flat_indices, _ = grid.get_atom_stencil_voxels(points)
        assert (flat_indices >= 0).all()
        assert (flat_indices < grid.grid_flat_size).all()

    def test_caller_owned_tensors_are_not_mutated(self):
        """torch.as_tensor can return the caller's own tensor without copying;
        the cubic-padding mutation must not corrupt it."""
        grid_shape = torch.tensor([2, 3, 4], dtype=torch.int64)
        left = torch.tensor([0.0, 0.0, 0.0])
        right = torch.tensor([1.0, 2.0, 3.0])
        GridConfig(
            grid_shape=grid_shape,
            voxel_size=(1.0, 1.0, 1.0),
            left_bottom_point=left,
            right_upper_point=right,
        )
        assert grid_shape.tolist() == [2, 3, 4]
        assert left.tolist() == [0.0, 0.0, 0.0]
        assert right.tolist() == [1.0, 2.0, 3.0]


class TestStencil:
    def test_stencil_dimensions_are_odd_and_bounded_by_grid(self):
        grid = GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.0, 1.0),
            left_bottom_point=(-3.0, -3.0, -3.0),
            right_upper_point=(3.0, 3.0, 3.0),
            sublattice_radius=100.0,  # deliberately larger than the grid
        )
        stencil = grid.stencil
        assert (stencil.dimensions % 2 == 1).all()
        assert torch.all(stencil.dimensions <= grid.grid_shape)

    def test_stencil_is_cached(self):
        grid = GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.0, 1.0),
            left_bottom_point=(-5.0, -5.0, -5.0),
            right_upper_point=(5.0, 5.0, 5.0),
            sublattice_radius=4.0,
        )
        assert grid.stencil is grid.stencil


class TestAtomAnchoring:
    def test_stencil_voxels_within_grid_bounds(self):
        grid = GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.2, 0.9),
            left_bottom_point=(-5.0, -4.0, -6.0),
            right_upper_point=(5.0, 4.0, 6.0),
            sublattice_radius=4.5,
        )
        points = torch.tensor(
            [[0.3, -1.2, 5.9], [-4.9, 3.8, -5.9], [0.0, 0.0, 0.0], [4.9, 3.9, 5.9]]
        )
        flat_indices, coords = grid.get_atom_stencil_voxels(points)
        assert flat_indices.shape == (4, grid.stencil.flat_indices.shape[0])
        assert coords.shape == (4, grid.stencil.flat_indices.shape[0], 3)
        assert (flat_indices >= 0).all()
        assert (flat_indices < grid.grid_flat_size).all()

    def test_batched_leading_dims_match_unbatched(self):
        grid = GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.0, 1.0),
            left_bottom_point=(-5.0, -5.0, -5.0),
            right_upper_point=(5.0, 5.0, 5.0),
            sublattice_radius=4.0,
        )
        points = torch.tensor([[0.3, -1.2, 2.9], [1.0, -2.0, 0.5]])
        unbatched_flat, unbatched_coords = grid.get_atom_stencil_voxels(points)

        batched = points.unsqueeze(0).repeat(3, 1, 1)
        batched_flat, batched_coords = grid.get_atom_stencil_voxels(batched)
        assert batched_flat.shape == (3, *unbatched_flat.shape)
        assert torch.equal(batched_flat[1], unbatched_flat)
        assert torch.allclose(batched_coords[1], unbatched_coords)

    def test_convert_cubic_index_to_flat_index_is_row_major(self):
        grid = GridConfig.from_grid_shape_and_voxel_size(
            grid_shape=(2, 3, 4), voxel_size=(1.0, 1.0, 1.0), equal_length=False
        )
        # last axis fastest: (0,0,1) -> 1, (0,1,0) -> 4, (1,0,0) -> 12
        indices = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
        flat = grid.convert_cubic_index_to_flat_index(indices)
        assert flat.tolist() == [0, 1, 4, 12]


def test_default_sublattice_radius():
    assert default_sublattice_radius(1.0) == 5.0
    assert default_sublattice_radius(2.0) == 6.0
    with pytest.raises(ValueError, match="pixel_spacing"):
        default_sublattice_radius(0.0)
