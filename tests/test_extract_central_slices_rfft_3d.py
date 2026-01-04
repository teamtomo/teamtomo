"""Unit tests pertaining to the 'extract_central_slices_rfft_3d' function."""

import os

import pytest
import torch
import urllib3
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from ttsim3d.models import Simulator, SimulatorConfig

from torch_fourier_slice.slice_extraction import extract_central_slices_rfft_3d

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

# NOTE: These are sets of ZYZ Euler angles, in degrees, which should produce mirrored
# projections (i.e. are 180 degrees apart in orientation space) *and* have ambiguous
# points lying around x=0 in Fourier space. Slice extraction should handle these cases
# exactly and produce conjugate slices.
ORIENTATION_SYMMETRY_PAIRS = [
    ([0, 0, 0], [0, 0, 180]),
    ([0, 0, 0], [180, 0, 0]),
    ([0, 20, 0], [0, 20, 180]),
    ([0, 20, 90], [0, 20, 270]),
    ([0, 60, 0], [0, 60, 180]),
    ([90, 90, 0], [90, 90, 180]),
    ([45, 0, 0], [45, 0, 180]),
    ([45, 90, 0], [45, 90, 180]),
]


def setup_slice_volume(
    pixel_spacing: float = 1.0, volume_dim: int = 128
) -> torch.Tensor:
    """Downloads and prepares a test volume from a set PDB file using ttsim3D."""
    tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Download PDB file
    url = "https://files.rcsb.org/download/pdb_00007myi.cif"
    pdb_path = os.path.join(tmp_dir, "pdb_00007myi.cif")
    if not os.path.exists(pdb_path):
        http = urllib3.PoolManager()
        response = http.request("GET", url)
        with open(pdb_path, "wb") as f:
            f.write(response.data)

    # Instantiate the configuration object
    sim_conf = SimulatorConfig(
        voltage=300.0,  # in keV
        apply_dose_weighting=True,
        dose_start=0.0,  # in e-/A^2
        dose_end=35.0,  # in e-/A^2
        upsampling=2,
    )

    # Instantiate the simulator
    sim = Simulator(
        pdb_filepath=pdb_path,
        pixel_spacing=pixel_spacing,  # Angstroms
        volume_shape=(volume_dim, volume_dim, volume_dim),
        b_factor_scaling=1.0,
        additional_b_factor=0.0,
        simulator_config=sim_conf,
    )

    return sim.run()


@pytest.mark.parametrize("device", DEVICES)
def test_extract_central_slices_rfft_3d(device: str):
    """Tests the extract_central_slices_rfft_3d function (standard, no edge cases)."""
    volume = setup_slice_volume().to(device)
    image_shape = volume.shape

    # Prepare volume for slicing in RFFT
    volume_rfft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    volume_rfft = torch.fft.rfftn(volume_rfft, dim=(-3, -2, -1))
    volume_rfft = torch.fft.fftshift(volume_rfft, dim=(-3, -2))

    # Prepare random rotation matrices
    num_slices = 10
    rotation_matrices = special_ortho_group.rvs(dim=3, size=num_slices)

    slices = extract_central_slices_rfft_3d(
        volume_rfft=volume_rfft,
        image_shape=image_shape,
        rotation_matrices=torch.tensor(rotation_matrices, device=device),
    )

    assert slices.device.type == device
    assert torch.is_complex(slices)
    assert slices.shape == (num_slices, image_shape[-2], image_shape[-1] // 2 + 1)

    # Check that the slice has Friedel symmetry along x=0
    x_zero_line = slices[:, :, 0]
    slice_half = x_zero_line[:, 1 : x_zero_line.shape[1] // 2]
    slice_conj_half = torch.conj(x_zero_line[:, -(x_zero_line.shape[1] // 2 - 1) :])
    slice_conj_half = torch.flip(slice_conj_half, dims=[1])

    assert torch.allclose(
        slice_half, slice_conj_half, atol=1e-6
    ), "Extracted slices do not obey Friedel symmetry along x=0 (expected for rfft)."


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("angle_pair", ORIENTATION_SYMMETRY_PAIRS)
def test_extract_central_slices_rfft_3d_symmetry_cases(
    device: str, angle_pair: tuple[list[float], list[float]]
):
    """Tests extract_central_slices_rfft_3d on symmetry-related orientations."""
    volume = setup_slice_volume().to(device)

    # Prepare volume for slicing in RFFT
    volume_rfft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    volume_rfft = torch.fft.rfftn(volume_rfft, dim=(-3, -2, -1))
    volume_rfft = torch.fft.fftshift(volume_rfft, dim=(-3, -2))

    left_angles, right_angles = angle_pair

    left_rot = Rotation.from_euler("ZYZ", left_angles, degrees=True).as_matrix()
    right_rot = Rotation.from_euler("ZYZ", right_angles, degrees=True).as_matrix()

    left_rot = torch.from_numpy(left_rot).to(device)
    right_rot = torch.from_numpy(right_rot).to(device)

    # Extract slices for both orientations
    left_slice = extract_central_slices_rfft_3d(
        volume_rfft=volume_rfft,
        image_shape=volume.shape,
        rotation_matrices=left_rot,
    )
    right_slice = extract_central_slices_rfft_3d(
        volume_rfft=volume_rfft,
        image_shape=volume.shape,
        rotation_matrices=right_rot,
    )

    # Check that the slices are conjugates of each other
    assert torch.allclose(
        left_slice, torch.conj(right_slice), atol=1e-6
    ), f"Slices for angles {left_angles} and {right_angles} are not conjugates."


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("angle_pair", ORIENTATION_SYMMETRY_PAIRS)
def test_extract_central_slices_rfft_3d_symmetry_cases_large_volume(
    device: str, angle_pair: tuple[list[float], list[float]]
):
    """Tests extract_central_slices_rfft_3d on symmetry-related orientations."""
    volume = setup_slice_volume().to(device)

    # Prepare volume for slicing in RFFT
    volume_rfft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    volume_rfft = torch.fft.rfftn(volume_rfft, dim=(-3, -2, -1))
    volume_rfft = torch.fft.fftshift(volume_rfft, dim=(-3, -2))

    left_angles, right_angles = angle_pair

    left_rot = Rotation.from_euler("ZYZ", left_angles, degrees=True).as_matrix()
    right_rot = Rotation.from_euler("ZYZ", right_angles, degrees=True).as_matrix()

    left_rot = torch.from_numpy(left_rot).to(device)
    right_rot = torch.from_numpy(right_rot).to(device)

    d = volume.shape[0] * 4
    image_shape = (d, d, d)

    # Extract slices for both orientations
    left_slice = extract_central_slices_rfft_3d(
        volume_rfft=volume_rfft,
        image_shape=image_shape,
        rotation_matrices=left_rot,
    )
    right_slice = extract_central_slices_rfft_3d(
        volume_rfft=volume_rfft,
        image_shape=image_shape,
        rotation_matrices=right_rot,
    )

    # Check that the slices are conjugates of each other
    assert torch.allclose(
        left_slice, torch.conj(right_slice), atol=1e-6
    ), f"Slices for angles {left_angles} and {right_angles} are not conjugates."


@pytest.mark.parametrize("device", DEVICES)
def test_extract_central_slices_rfft_3d_symmetry_random_angles(device: str):
    """Generates and tests random angle pairs that should produce conjugate slices."""
    NUM_ORIENTATIONS = 100

    left_angles = torch.zeros((NUM_ORIENTATIONS, 3))
    left_angles[:, 0] = torch.rand(NUM_ORIENTATIONS) * 360
    left_angles[:, 1] = torch.rand(NUM_ORIENTATIONS) * 180
    left_angles[:, 2] = torch.rand(NUM_ORIENTATIONS) * 360

    volume = setup_slice_volume().to(device)

    # Prepare volume for slicing in RFFT
    volume_rfft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    volume_rfft = torch.fft.rfftn(volume_rfft, dim=(-3, -2, -1))
    volume_rfft = torch.fft.fftshift(volume_rfft, dim=(-3, -2))

    # Angles on the right side are rotated 180 degrees for in-plane rotation
    right_offset = Rotation.from_euler("ZYZ", [0, 0, 180], degrees=True).as_matrix()

    for i in range(NUM_ORIENTATIONS):
        left_rot = Rotation.from_euler(
            "ZYZ", left_angles[i].numpy(), degrees=True
        ).as_matrix()
        right_rot = left_rot @ right_offset

        left_rot = torch.from_numpy(left_rot).to(device)
        right_rot = torch.from_numpy(right_rot).to(device)

        # Extract slices for both orientations
        left_slice = extract_central_slices_rfft_3d(
            volume_rfft=volume_rfft,
            image_shape=volume.shape,
            rotation_matrices=left_rot,
        )
        right_slice = extract_central_slices_rfft_3d(
            volume_rfft=volume_rfft,
            image_shape=volume.shape,
            rotation_matrices=right_rot,
        )

        assert torch.allclose(
            left_slice, torch.conj(right_slice), atol=1e-6
        ), "Slices are not conjugates."
