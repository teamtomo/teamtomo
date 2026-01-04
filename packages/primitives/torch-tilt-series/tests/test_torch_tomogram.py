import pytest
import torch

from torch_tilt_series import Tomogram

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="can only move device with CUDA support"
)
def test_device_move():
    """Test that tensors can move to other device."""
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations
    images = torch.zeros((3, 10, 10))  # 3 tilts, 10x10 images

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
    )
    assert "cpu" == str(tomogram.images.device)

    tomogram.to("cuda")

    # Check shape and type
    assert "cuda" in str(tomogram.images.device)
    assert "cuda" in str(tomogram.device)
    assert "cuda" in str(tomogram.tilt_angles.device)
    assert "cuda" in str(tomogram.tilt_axis_angle.device)
    assert "cuda" in str(tomogram.sample_translations.device)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_projection_matrices(device):
    """Test that projection matrices are computed correctly."""
    # Create simple test data
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations
    images = torch.zeros((3, 10, 10), device=device)  # 3 tilts, 10x10 images

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
        device=device,
    )

    # Get projection matrices
    matrices = tomogram.projection_matrices

    # Check shape and type
    assert matrices.shape == (3, 4, 4)  # 3 tilts, 4x4 matrices
    assert matrices.dtype == torch.float32
    assert device in str(matrices.device)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_project_points(device):
    """Test that points can be projected from 3D to 2D."""
    # Create simple test data
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations
    images = torch.zeros((3, 10, 10))  # 3 tilts, 10x10 images

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
        device=device,
    )

    # Create a single 3D point at the origin
    points_zyx = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    # Project the point
    projected_yx = tomogram.project_points(points_zyx)

    # Check shape and type
    assert projected_yx.shape == (1, 3, 2)  # 1 point, 3 tilts, 2D coordinates
    assert projected_yx.dtype == torch.float32
    assert device in str(projected_yx.device)

    # For a point at the origin with no translations, the y-coordinate should be 0
    # for all tilts, and the x-coordinate should depend on the tilt angle
    assert torch.allclose(
        projected_yx[0, :, 0], torch.tensor([0.0, 0.0, 0.0], device=device)
    )


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_extract_particle_tilt_series(device):
    """Test that particle tilt series can be extracted."""
    # Create simple test data
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations

    # Create simple images with a pattern
    images = torch.zeros((3, 32, 32))
    images[:, 14:18, 14:18] = 1.0  # Create a small square in the center

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
        device=device,
    )

    # Create a single 3D point at the origin
    points_zyx = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    # Extract particle tilt series
    particle_tilt_series = tomogram.extract_particle_tilt_series(
        points_zyx, sidelength=8
    )

    # Check shape and type
    assert particle_tilt_series.shape == (1, 3, 8, 8)  # 1 point, 3 tilts, 8x8 images
    assert particle_tilt_series.dtype == torch.float32
    assert device in str(particle_tilt_series.device)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_reconstruct_subvolume(device):
    """Test that a subvolume can be reconstructed."""
    # Create simple test data
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations

    # Create simple images with a pattern
    images = torch.zeros((3, 32, 32))
    images[:, 14:18, 14:18] = 1.0  # Create a small square in the center

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
        device=device,
    )

    # Create a single 3D point at the origin
    point_zyx = torch.tensor([0.0, 0.0, 0.0], device=device)

    # Reconstruct subvolume
    subvolume = tomogram.reconstruct_subvolume(point_zyx, sidelength=8)

    # Check shape and type
    assert subvolume.shape == (8, 8, 8)  # 8x8x8 volume
    assert subvolume.dtype == torch.float32
    assert device in str(subvolume.device)


@pytest.mark.parametrize(
    "device",
    DEVICES,
)
def test_reconstruct_tomogram(device):
    """Test that a tomogram can be reconstructed."""
    # Create simple test data
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))  # 3 tilts, 2D translations

    # Create simple images with a pattern
    images = torch.zeros((3, 32, 32))
    images[:, 14:18, 14:18] = 1.0  # Create a small square in the center

    # Initialize tomogram
    tomogram = Tomogram(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
        device=device,
    )

    # Reconstruct a small tomogram
    volume = tomogram.reconstruct_tomogram((16, 16, 16), sidelength=8)

    # Check shape and type
    assert volume.shape == (16, 16, 16)  # 16x16x16 volume
    assert volume.dtype == torch.float32
    assert device in str(volume.device)
