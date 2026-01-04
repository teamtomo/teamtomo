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
    assert (
        mean_error < 1e-5
    ), f"Friedel symmetry violated: mean error = {mean_error:.9f}"
