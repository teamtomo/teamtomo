import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from torch_fourier_shell_correlation import fsc

from torch_fourier_slice import (
    backproject_2d_to_3d,
    backproject_2d_to_3d_multichannel,
    project_2d_to_1d,
    project_3d_to_2d,
    project_3d_to_2d_multichannel,
)
from torch_fourier_slice.slice_insertion import insert_central_slices_rfft_3d

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_project_3d_to_2d_rotation_center(device):
    # rotation center should be at position of DC in DFT
    volume = torch.zeros((32, 32, 32), device=device)
    volume[16, 16, 16] = 1

    # make projections
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=100),
        device=device,
    )
    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
    )

    assert device in str(projections.device)
    # check max is always at (16, 16), implying point (16, 16) never moves
    for image in projections:
        max_idx = torch.argmax(image)
        i, j = divmod(max_idx.item(), 32)
        assert (i, j) == (16, 16)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_project_2d_to_1d_rotation_center(device):
    # rotation center should be at position of DC in DFT
    image = torch.zeros((32, 32), device=device)
    image[16, 16] = 1

    # make projections
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=2, size=100),
        device=device,
    )
    projections = project_2d_to_1d(
        image=image,
        rotation_matrices=rotation_matrices,
    )

    assert device in str(projections.device)
    # check max is always at (16), implying point (16) never moves
    for image in projections:
        i = torch.argmax(image)
        assert i == 16


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_3d_2d_projection_backprojection_cycle(cube, device):
    # make projections
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=1500),
        device=device,
    )
    projections = project_3d_to_2d(
        volume=cube.to(device),
        rotation_matrices=rotation_matrices,
    )

    # reconstruct
    volume = backproject_2d_to_3d(
        images=projections,
        rotation_matrices=rotation_matrices,
    )

    assert device in str(projections.device)
    assert device in str(volume.device)

    # calculate FSC between the ground truth volume and the reconstruction
    # move to cpu as a workaround for FSC not running on GPU
    _fsc = fsc(cube.to("cpu"), volume.float().to("cpu"))
    assert torch.all(_fsc[-10:] > 0.99)  # few low res shells at 0.98...


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_3d_2d_projection_backprojection_cycle_leading_dims(cube, device):
    # make projections
    size = cube.shape[-1]
    rotation_matrices = torch.rand((4, 5, 3, 3), device=device)
    projections = project_3d_to_2d(
        volume=cube.to(device),
        rotation_matrices=rotation_matrices,
    )

    assert device in str(projections.device)
    assert projections.shape == (4, 5, size, size)

    # reconstruct
    volume = backproject_2d_to_3d(
        images=projections,
        rotation_matrices=rotation_matrices,
    )

    assert device in str(volume.device)
    assert volume.shape == (size,) * 3


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_3d_to_2d_projection_backprojection_cycle_multichannel(device):
    channels, slices, size = 4, 8, 10
    volumes_shape = (channels, size, size, size)
    projections_shape = (slices, channels, size, size)
    # a volume with 4 channels
    volumes = torch.rand(volumes_shape, device=device)  # (c, d, d, d)
    # a rotation matrix for each tilt -> (n, 3, 3)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=slices),
        device=device,
    )

    # run batched back projection
    projections = project_3d_to_2d_multichannel(volumes, rotation_matrices)
    assert projections.shape == projections_shape
    assert device in str(projections.device)

    # run batched back projection
    result = backproject_2d_to_3d_multichannel(projections, rotation_matrices)
    assert result.shape == volumes_shape
    assert device in str(result.device)


@pytest.mark.parametrize(
    "dtype, device",
    ((p0, p1) for p0, p1 in zip([torch.float32, torch.float64], DEVICES, strict=False)),
)
def test_dtypes_slice_insertion(dtype, device):
    images = torch.rand((10, 28, 28), dtype=dtype, device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=10),
        dtype=dtype,
        device=device,
    )
    result = backproject_2d_to_3d(images, rotation_matrices)
    assert result.dtype == dtype
    assert device in str(result.device)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_backprojection_friedel_symmetry_x0_plane(cube, device):
    """Test that the x=0 plane in the Fourier transform has Friedel symmetry.

    For a real-valued volume, its Fourier transform must satisfy:
    F(x, y, z) = conj(F(-x, -y, -z))

    In the rfft representation, the x=0 plane should satisfy:
    F(0, y, z) = conj(F(0, -y, -z))

    This test uses rotations that specifically insert values into the x=0 plane
    to ensure Friedel symmetry is properly enforced during interpolation.
    """
    # 0 and 180 rotation of the inserted slices
    # this checks for small inconsistencies of the rotation matrices
    angles = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 180.0]]).T  # (b, 3)
    rot_x = Rotation.from_euler("ZYZ", angles=angles, degrees=True)
    rot_x = torch.tensor(rot_x.as_matrix(), device=device)

    # Create specific rotation matrices that will insert into x=0 plane
    # and need correctly handle Friedel symmetric insertion
    rot_y_90 = torch.tensor(
        [
            [0, 0, 1],  # x' = z
            [0, 1, 0],  # y' = y
            [-1, 0, 0],  # z' = -x
        ],
        dtype=torch.float32,
        device=device,
    )

    # 90-degree rotation around z-axis
    rot_z_90 = torch.tensor(
        [
            [0, -1, 0],  # x' = -y
            [1, 0, 0],  # y' = x
            [0, 0, 1],  # z' = z
        ],
        dtype=torch.float32,
        device=device,
    )

    # Combine targeted rotations with some random ones
    random_rotations = torch.tensor(
        special_ortho_group.rvs(dim=3, size=4, random_state=42),
        dtype=torch.float32,
        device=device,
    )

    # cat all test matrices
    rotation_matrices = torch.cat(
        [rot_x, rot_y_90.unsqueeze(0), rot_z_90.unsqueeze(0), random_rotations], dim=0
    )

    projections = project_3d_to_2d(
        volume=cube.to(device),
        rotation_matrices=rotation_matrices,
    )

    # Process projections to DFT format for insertion
    images = projections
    images = torch.fft.fftshift(images, dim=(-2, -1))  # volume center to array origin
    images = torch.fft.rfftn(images, dim=(-2, -1))
    images = torch.fft.fftshift(images, dim=(-2,))  # actual fftshift

    # Insert directly into 3D DFT using low-level API
    volume_shape = (cube.shape[-1], cube.shape[-1], cube.shape[-1])
    volume_fft, _ = insert_central_slices_rfft_3d(
        image_rfft=images,
        volume_shape=volume_shape,
        rotation_matrices=rotation_matrices,
    )

    # Extract the x=0 plane (first index in the rfft output)
    x0_plane = volume_fft[:, :, 0]  # shape (d, d)

    # For Friedel symmetry: F(0, y, z) = conj(F(0, -y, -z))
    d = x0_plane.shape[0]

    # Sum all absolute differences
    total_error = 0.0
    num_pairs = 0

    # Only loop over half the plane since Friedel pairs are symmetric
    for z in range(d // 2 + 1):  # 0 to d//2 inclusive
        if z < d // 2:
            # Bottom half: check all y values
            y_range = range(d)
        else:  # z == d // 2
            # Center line: only check up to and including center
            y_range = range(d // 2 + 1)

        for y in y_range:
            # Skip the DC component at center
            if z == d // 2 and y == d // 2:
                continue

            # Calculate mirrored indices
            z_mirror = (d - z) % d
            y_mirror = (d - y) % d

            # Check Friedel symmetry
            val = x0_plane[z, y]
            val_mirror_conj = torch.conj(x0_plane[z_mirror, y_mirror])

            # Accumulate absolute error
            total_error += torch.abs(val - val_mirror_conj).item()
            num_pairs += 1

    # Check mean error per pair is small
    mean_error = total_error / num_pairs if num_pairs > 0 else 0.0
    print(mean_error)
    assert mean_error < 1e-5, (
        f"Friedel symmetry violated: mean error = {mean_error:.9f}"
    )


@pytest.mark.parametrize("device", DEVICES)
def test_project_3d_to_2d_with_transform_matrix_identity(device):
    """Test project_3d_to_2d with identity transform matrix."""
    volume = torch.randn((32, 32, 32), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=5),
        device=device,
    )

    # Identity transform matrix
    identity_matrix = torch.eye(2, device=device, dtype=torch.float32)

    # Project with and without transform matrix
    projections_no_transform = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
    )

    projections_with_identity = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        transform_matrix=identity_matrix,
    )

    # Identity transform should approximately preserve the projection
    # (allowing for small interpolation differences)
    assert projections_with_identity.shape == projections_no_transform.shape
    assert torch.allclose(
        projections_with_identity, projections_no_transform, atol=1e-5
    )


@pytest.mark.parametrize("device", DEVICES)
def test_project_3d_to_2d_with_transform_matrix_scaling(device):
    """Test project_3d_to_2d with scaling transform matrix."""
    volume = torch.randn((32, 32, 32), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=3),
        device=device,
    )

    # Scaling transform matrix (2x in y direction)
    transform_matrix = torch.tensor(
        [[2.0, 0.0], [0.0, 1.0]], device=device, dtype=torch.float32
    )

    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        transform_matrix=transform_matrix,
    )

    # Check output shape and properties
    assert projections.shape == (3, 32, 32)
    assert device in str(projections.device)
    assert projections.dtype == volume.dtype


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "transform_matrix",
    [
        torch.tensor([[1.5, 0.0], [0.0, 1.0]]),  # Stretch in y
        torch.tensor([[1.0, 0.0], [0.0, 0.7]]),  # Compress in x
        torch.tensor([[1.2, 0.3], [0.1, 1.1]]),  # Anisotropic
    ],
)
def test_project_3d_to_2d_with_transform_matrix_various(device, transform_matrix):
    """Test project_3d_to_2d with various transformation matrices."""
    volume = torch.randn((24, 24, 24), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=4),
        device=device,
    )

    # Move transform matrix to device
    transform_matrix = transform_matrix.to(device=device, dtype=torch.float32)

    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        transform_matrix=transform_matrix,
    )

    # Check output shape and properties
    assert projections.shape == (4, 24, 24)
    assert device in str(projections.device)
    assert projections.dtype == volume.dtype

    # Check that output is not all zeros
    assert not torch.allclose(projections, torch.zeros_like(projections), atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_project_3d_to_2d_multichannel_with_transform_matrix(device):
    """Test project_3d_to_2d_multichannel with transform matrix."""
    channels, slices, size = 3, 5, 16
    volume = torch.randn((channels, size, size, size), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=slices),
        device=device,
    )

    # Scaling transform matrix
    transform_matrix = torch.tensor(
        [[1.5, 0.0], [0.0, 1.0]], device=device, dtype=torch.float32
    )

    projections = project_3d_to_2d_multichannel(
        volume=volume,
        rotation_matrices=rotation_matrices,
        transform_matrix=transform_matrix,
    )

    # Check output shape and properties
    assert projections.shape == (slices, channels, size, size)
    assert device in str(projections.device)
    assert projections.dtype == volume.dtype


@pytest.mark.parametrize("device", DEVICES)
def test_project_3d_to_2d_with_transform_matrix_batched(device):
    """Test project_3d_to_2d with transform matrix and batched rotations."""
    volume = torch.randn((20, 20, 20), device=device)
    # Multiple batch dimensions
    rotation_matrices = torch.rand((2, 3, 3, 3), device=device)

    # Anisotropic transform matrix
    transform_matrix = torch.tensor(
        [[1.2, 0.2], [0.1, 1.1]], device=device, dtype=torch.float32
    )

    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        transform_matrix=transform_matrix,
    )

    # Check output shape matches batch dimensions
    assert projections.shape == (2, 3, 20, 20)
    assert device in str(projections.device)


@pytest.mark.parametrize("device", DEVICES)
def test_project_3d_to_2d_with_ewald_curvature(device):
    """Test project_3d_to_2d with Ewald sphere curvature enabled."""
    volume = torch.randn((32, 32, 32), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=5),
        device=device,
    )

    # Project with Ewald curvature
    projections_ewald = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Check output shape and properties
    assert projections_ewald.shape == (5, 32, 32)
    assert device in str(projections_ewald.device)
    assert projections_ewald.dtype == volume.dtype

    # Check that output is not all zeros
    assert not torch.allclose(
        projections_ewald, torch.zeros_like(projections_ewald), atol=1e-10
    )


@pytest.mark.parametrize("device", DEVICES)
def test_project_3d_to_2d_ewald_vs_flat(device):
    """Test that Ewald curvature produces different results than flat slice."""
    volume = torch.randn((24, 24, 24), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=3),
        device=device,
    )

    # Project with flat slice (default)
    projections_flat = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=False,
    )

    # Project with Ewald curvature
    projections_ewald = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Results should be different (Ewald curvature changes the slice)
    assert projections_flat.shape == projections_ewald.shape
    assert not torch.allclose(projections_flat, projections_ewald, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("voltage_kv", [100.0, 200.0, 300.0, 400.0])
def test_project_3d_to_2d_ewald_different_voltages(device, voltage_kv):
    """Test project_3d_to_2d with different acceleration voltages."""
    volume = torch.randn((20, 20, 20), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=2),
        device=device,
    )

    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=voltage_kv,
        ewald_px_size=1.0,
    )

    # Check output shape and properties
    assert projections.shape == (2, 20, 20)
    assert device in str(projections.device)
    assert not torch.allclose(projections, torch.zeros_like(projections), atol=1e-10)


@pytest.mark.parametrize("device", DEVICES)
def test_project_3d_to_2d_ewald_flip_sign(device):
    """Test that flip_sign produces different results."""
    volume = torch.randn((24, 24, 24), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=2),
        device=device,
    )

    # Project with normal Ewald curvature
    projections_normal = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_flip_sign=False,
        ewald_px_size=1.0,
    )

    # Project with flipped sign
    projections_flipped = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_flip_sign=True,
        ewald_px_size=1.0,
    )

    # Results should be different
    assert projections_normal.shape == projections_flipped.shape
    assert not torch.allclose(projections_normal, projections_flipped, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("px_size", [0.5, 1.0, 2.0, 5.0])
def test_project_3d_to_2d_ewald_different_pixel_sizes(device, px_size):
    """Test project_3d_to_2d with different pixel sizes."""
    volume = torch.randn((20, 20, 20), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=2),
        device=device,
    )

    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=px_size,
    )

    # Check output shape and properties
    assert projections.shape == (2, 20, 20)
    assert device in str(projections.device)


@pytest.mark.parametrize("device", DEVICES)
def test_project_3d_to_2d_multichannel_with_ewald_curvature(device):
    """Test project_3d_to_2d_multichannel with Ewald sphere curvature."""
    channels, slices, size = 3, 4, 16
    volume = torch.randn((channels, size, size, size), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=slices),
        device=device,
    )

    # Project with Ewald curvature
    projections = project_3d_to_2d_multichannel(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Check output shape and properties
    assert projections.shape == (slices, channels, size, size)
    assert device in str(projections.device)
    assert projections.dtype == volume.dtype


@pytest.mark.parametrize("device", DEVICES)
def test_project_3d_to_2d_multichannel_ewald_vs_flat(device):
    """Test that multichannel Ewald curvature produces different results."""
    channels, slices, size = 2, 3, 16
    volume = torch.randn((channels, size, size, size), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=slices),
        device=device,
    )

    # Project with flat slice
    projections_flat = project_3d_to_2d_multichannel(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=False,
    )

    # Project with Ewald curvature
    projections_ewald = project_3d_to_2d_multichannel(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Results should be different
    assert projections_flat.shape == projections_ewald.shape
    assert not torch.allclose(projections_flat, projections_ewald, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_backproject_2d_to_3d_with_ewald_curvature(device):
    """Test backproject_2d_to_3d with Ewald sphere curvature."""
    images = torch.randn((5, 32, 32), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=5),
        device=device,
    )

    # Backproject with Ewald curvature
    reconstruction = backproject_2d_to_3d(
        images=images,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Check output shape and properties
    assert reconstruction.shape == (32, 32, 32)
    assert device in str(reconstruction.device)
    assert reconstruction.dtype == images.dtype


@pytest.mark.parametrize("device", DEVICES)
def test_backproject_2d_to_3d_ewald_vs_flat(device):
    """Test that Ewald curvature produces different backprojection results."""
    images = torch.randn((4, 24, 24), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=4),
        device=device,
    )

    # Backproject with flat slice
    reconstruction_flat = backproject_2d_to_3d(
        images=images,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=False,
    )

    # Backproject with Ewald curvature
    reconstruction_ewald = backproject_2d_to_3d(
        images=images,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Results should be different
    assert reconstruction_flat.shape == reconstruction_ewald.shape
    assert not torch.allclose(reconstruction_flat, reconstruction_ewald, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_backproject_2d_to_3d_multichannel_with_ewald_curvature(device):
    """Test backproject_2d_to_3d_multichannel with Ewald sphere curvature."""
    channels, slices, size = 3, 4, 16
    images = torch.randn((slices, channels, size, size), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=slices),
        device=device,
    )

    # Backproject with Ewald curvature
    reconstruction = backproject_2d_to_3d_multichannel(
        images=images,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Check output shape and properties
    assert reconstruction.shape == (channels, size, size, size)
    assert device in str(reconstruction.device)
    assert reconstruction.dtype == images.dtype


@pytest.mark.parametrize("device", DEVICES)
def test_backproject_2d_to_3d_multichannel_ewald_vs_flat(device):
    """Test that multichannel Ewald curvature produces different backprojection."""
    channels, slices, size = 2, 3, 16
    images = torch.randn((slices, channels, size, size), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=slices),
        device=device,
    )

    # Backproject with flat slice
    reconstruction_flat = backproject_2d_to_3d_multichannel(
        images=images,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=False,
    )

    # Backproject with Ewald curvature
    reconstruction_ewald = backproject_2d_to_3d_multichannel(
        images=images,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Results should be different
    assert reconstruction_flat.shape == reconstruction_ewald.shape
    assert not torch.allclose(reconstruction_flat, reconstruction_ewald, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_projection_backprojection_cycle_with_ewald(device):
    """Test projection-backprojection cycle with Ewald curvature."""
    volume = torch.randn((24, 24, 24), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=100),
        device=device,
    )

    # Project with Ewald curvature
    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Backproject with Ewald curvature (must match!)
    reconstruction = backproject_2d_to_3d(
        images=projections,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    assert device in str(projections.device)
    assert device in str(reconstruction.device)
    assert reconstruction.shape == volume.shape

    # Calculate FSC between original and reconstruction
    # Move to cpu as a workaround for FSC not running on GPU
    _fsc = fsc(volume.to("cpu"), reconstruction.float().to("cpu"))
    # With Ewald curvature, reconstruction quality may be different
    # but should still be reasonable (lower threshold than flat case)
    assert torch.all(_fsc[-10:] > 0.85)


@pytest.mark.parametrize("device", DEVICES)
def test_projection_backprojection_cycle_ewald_mismatch(device):
    """Test that mismatched Ewald settings produce different results."""
    volume = torch.randn((20, 20, 20), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=50),
        device=device,
    )

    # Project with Ewald curvature
    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Backproject WITHOUT Ewald curvature (mismatch)
    reconstruction_mismatch = backproject_2d_to_3d(
        images=projections,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=False,
    )

    # Backproject WITH Ewald curvature (match)
    reconstruction_match = backproject_2d_to_3d(
        images=projections,
        rotation_matrices=rotation_matrices,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Mismatched should produce different results than matched
    assert reconstruction_mismatch.shape == reconstruction_match.shape
    assert not torch.allclose(reconstruction_mismatch, reconstruction_match, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_ewald_curvature_combined_with_transform_matrix(device):
    """Test that Ewald curvature can be combined with transform matrix."""
    volume = torch.randn((24, 24, 24), device=device)
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(dim=3, size=3),
        device=device,
    )

    # Scaling transform matrix
    transform_matrix = torch.tensor(
        [[1.5, 0.0], [0.0, 1.0]], device=device, dtype=torch.float32
    )

    # Project with both Ewald curvature and transform matrix
    projections = project_3d_to_2d(
        volume=volume,
        rotation_matrices=rotation_matrices,
        transform_matrix=transform_matrix,
        apply_ewald_curvature=True,
        ewald_voltage_kv=300.0,
        ewald_px_size=1.0,
    )

    # Check output shape and properties
    assert projections.shape == (3, 24, 24)
    assert device in str(projections.device)
    assert not torch.allclose(projections, torch.zeros_like(projections), atol=1e-10)
