import pytest
import torch

from torch_grid_utils import (
    patch_grid,
    patch_grid_centers,
    patch_grid_indices,
    patch_grid_lazy,
)


def test_patch_grid_centers_2d_basic():
    """Test basic 2D patch grid centers."""
    centers = patch_grid_centers(
        image_shape=(10, 10),
        patch_shape=(3, 3),
        patch_step=(2, 2),
        distribute_patches=False,
    )
    assert centers.shape == (4, 4, 2)  # 4x4 grid, 2 coordinates
    assert centers.dtype == torch.long


def test_patch_grid_centers_2d_distribute():
    """Test 2D patch grid centers with distribute_patches=True."""
    centers = patch_grid_centers(
        image_shape=(10, 10),
        patch_shape=(3, 3),
        patch_step=(2, 2),
        distribute_patches=True,
    )
    assert centers.shape[0] == 4  # Should have 4 patches in each dimension
    assert centers.shape[1] == 4
    assert centers.shape[2] == 2


def test_patch_grid_centers_3d_basic():
    """Test basic 3D patch grid centers."""
    centers = patch_grid_centers(
        image_shape=(8, 10, 10),
        patch_shape=(2, 3, 3),
        patch_step=(2, 2, 2),
        distribute_patches=False,
    )
    assert centers.shape == (3, 4, 4, 3)  # 3x4x4 grid, 3 coordinates
    assert centers.dtype == torch.long


def test_patch_grid_indices_2d_basic():
    """Test basic 2D patch grid indices."""
    idx_h, idx_w = patch_grid_indices(
        image_shape=(10, 10),
        patch_shape=(3, 3),
        patch_step=(2, 2),
        distribute_patches=False,
    )
    assert isinstance(idx_h, torch.Tensor)
    assert isinstance(idx_w, torch.Tensor)
    # Check that indices have the right shape for broadcasting
    assert len(idx_h.shape) == 4
    assert len(idx_w.shape) == 4


def test_patch_grid_indices_3d_basic():
    """Test basic 3D patch grid indices."""
    idx_d, idx_h, idx_w = patch_grid_indices(
        image_shape=(8, 10, 10),
        patch_shape=(2, 3, 3),
        patch_step=(2, 2, 2),
        distribute_patches=False,
    )
    assert isinstance(idx_d, torch.Tensor)
    assert isinstance(idx_h, torch.Tensor)
    assert isinstance(idx_w, torch.Tensor)
    # Check that indices have the right shape for broadcasting
    assert len(idx_d.shape) == 6
    assert len(idx_h.shape) == 6
    assert len(idx_w.shape) == 6


def test_patch_grid_2d_basic():
    """Test basic 2D patch grid extraction."""
    image = torch.randn(10, 10)
    patch_shape = (3, 3)
    patch_step = (2, 2)

    patches, centers = patch_grid(
        images=image,
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=False,
    )

    # Check output shapes
    assert patches.shape == (4, 4, 3, 3)  # 4x4 grid of 3x3 patches
    assert centers.shape == (4, 4, 2)  # 4x4 grid, 2 coordinates

    # Check that patches are extracted correctly
    assert patches[0, 0].shape == (3, 3)


def test_patch_grid_2d_batch():
    """Test 2D patch grid with batch dimension."""
    images = torch.randn(2, 10, 10)
    patch_shape = (3, 3)
    patch_step = (2, 2)

    patches, centers = patch_grid(
        images=images,
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=False,
    )

    # Check output shapes
    assert patches.shape == (2, 4, 4, 3, 3)  # batch x 4x4 grid of 3x3 patches
    assert centers.shape == (4, 4, 2)


def test_patch_grid_3d_basic():
    """Test basic 3D patch grid extraction."""
    image = torch.randn(8, 10, 10)
    patch_shape = (2, 3, 3)
    patch_step = (2, 2, 2)

    patches, centers = patch_grid(
        images=image,
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=False,
    )

    # Check output shapes
    assert patches.shape == (3, 4, 4, 2, 3, 3)  # 3x4x4 grid of 2x3x3 patches
    assert centers.shape == (3, 4, 4, 3)  # 3x4x4 grid, 3 coordinates


def test_patch_grid_2d_distribute():
    """Test 2D patch grid with distribute_patches=True."""
    image = torch.randn(10, 10)
    patch_shape = (3, 3)
    patch_step = (2, 2)

    patches, _ = patch_grid(
        images=image,
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=True,
    )

    # Should still produce patches
    assert len(patches.shape) == 4
    assert patches.shape[-2:] == patch_shape


def test_patch_grid_lazy_2d_basic():
    """Test basic lazy patch grid for 2D."""
    image = torch.randn(10, 10)
    patch_shape = (3, 3)
    patch_step = (2, 2)

    lazy_patches, centers = patch_grid_lazy(
        images=image,
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=False,
    )

    # Check that centers are computed
    assert centers.shape == (4, 4, 2)

    # Check that lazy grid has correct shape property
    assert lazy_patches.shape == (4, 4, 3, 3)

    # Check that we can access patches (slicing first grid dimension)
    subset = lazy_patches[0:2]
    assert subset.shape == (2, 4, 3, 3)

    # Check that indexing works (returns a tensor)
    indexed = lazy_patches[0, 0]
    assert isinstance(indexed, torch.Tensor)
    assert indexed.shape[-2:] == patch_shape  # Last two dims should be patch shape


def test_patch_grid_lazy_3d_basic():
    """Test basic lazy patch grid for 3D."""
    image = torch.randn(8, 10, 10)
    patch_shape = (2, 3, 3)
    patch_step = (2, 2, 2)

    lazy_patches, centers = patch_grid_lazy(
        images=image,
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=False,
    )

    # Check that centers are computed
    assert centers.shape == (3, 4, 4, 3)

    # Check that lazy grid has correct shape property
    assert lazy_patches.shape == (3, 4, 4, 2, 3, 3)


def test_patch_grid_lazy_random_subset():
    """Test lazy patch grid random subset extraction."""
    image = torch.randn(10, 10)
    patch_shape = (3, 3)
    patch_step = (2, 2)

    lazy_patches, _ = patch_grid_lazy(
        images=image,
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=False,
    )

    # Get random subset
    patches, subset_centers = lazy_patches.random_subset(n_patches=5)

    assert patches.shape[0] == 5  # Should have 5 patches
    assert subset_centers.shape[0] == 5
    assert patches.shape[-2:] == patch_shape


def test_patch_grid_lazy_cache():
    """Test lazy patch grid caching."""
    image = torch.randn(10, 10)
    patch_shape = (3, 3)
    patch_step = (2, 2)

    lazy_patches, _ = patch_grid_lazy(
        images=image,
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=False,
    )

    # Access same patches twice - should use cache
    subset1 = lazy_patches[0:2, 0:2]
    subset2 = lazy_patches[0:2, 0:2]

    assert torch.allclose(subset1, subset2)

    # Clear cache
    lazy_patches.clear_cache()
    assert len(lazy_patches._cache) == 0


def test_patch_grid_error_mismatched_dims():
    """Test that mismatched dimensions raise an error."""
    image = torch.randn(10, 10)

    with pytest.raises(
        ValueError, match="patch shape and step must have the same number"
    ):
        patch_grid(
            images=image,
            patch_shape=(3, 3),
            patch_step=(2,),  # Mismatched dimensions
            distribute_patches=False,
        )


def test_patch_grid_centers_error_mismatched_dims():
    """Test that mismatched dimensions raise an error in patch_grid_centers."""
    with pytest.raises(ValueError, match="image shape, patch length and patch step"):
        patch_grid_centers(
            image_shape=(10, 10),
            patch_shape=(3, 3, 3),  # Mismatched dimensions
            patch_step=(2, 2),
            distribute_patches=False,
        )


def test_patch_grid_centers_error_unsupported_dim():
    """Test that unsupported dimensions raise an error."""
    with pytest.raises(NotImplementedError, match="only 2D and 3D"):
        patch_grid_centers(
            image_shape=(10, 10, 10, 10),  # 4D not supported
            patch_shape=(3, 3, 3, 3),
            patch_step=(2, 2, 2, 2),
            distribute_patches=False,
        )


def test_patch_grid_device():
    """Test that patch grid works on different devices."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        image = torch.randn(10, 10, device=device)

        patches, centers = patch_grid(
            images=image,
            patch_shape=(3, 3),
            patch_step=(2, 2),
            distribute_patches=False,
        )

        # Normalize devices for comparison (cuda == cuda:0)
        assert patches.device.type == device.type
        assert patches.device.index == (device.index or 0)
        assert centers.device.type == device.type
        assert centers.device.index == (device.index or 0)


def test_patch_grid_centers_device():
    """Test that patch grid centers respect device parameter."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        centers = patch_grid_centers(
            image_shape=(10, 10),
            patch_shape=(3, 3),
            patch_step=(2, 2),
            distribute_patches=False,
            device=device,
        )

        # Normalize devices for comparison (cuda == cuda:0)
        assert centers.device.type == device.type
        assert centers.device.index == (device.index or 0)
