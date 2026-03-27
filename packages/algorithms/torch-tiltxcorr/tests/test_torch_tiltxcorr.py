import torch
import pytest
import numpy as np
import einops

from torch_affine_utils.transforms_3d import Ry, Rz
from torch_affine_utils.utils import homogenise_coordinates
from torch_image_interpolation import insert_into_image_3d
from torch_fourier_slice import project_3d_to_2d
from torch_fourier_shift import fourier_shift_image_2d

from torch_tiltxcorr import tiltxcorr


def _generate_shifted_tilt_series(
    shift_magnitude: float = 5.0,
    d: int = 128,
    n_points_on_plane: int = 100,
    tilt_angles_deg: torch.Tensor | None = None,
    tilt_axis_angle: float = 85.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """
    Generate a synthetic tilt series with known shifts for testing.

    Parameters
    ----------
    shift_magnitude : float
        Maximum magnitude of random shifts in pixels
    d : int
        Volume dimension (cubic volume d x d x d)
    n_points_on_plane : int
        Number of random points to place on the plane
    tilt_angles_deg : torch.Tensor, optional
        Tilt angles in degrees. If None, uses linspace(-60, 60, 41)
    tilt_axis_angle : float
        Tilt axis angle in degrees
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tilt_series : torch.Tensor
        Generated tilt series with shifts applied (n_tilts, h, w)
    tilt_angles : torch.Tensor
        Tilt angles in degrees (n_tilts,)
    tilt_axis_angle : float
        Tilt axis angle in degrees
    applied_shifts : torch.Tensor
        Shifts that were applied to create misalignment (n_tilts, 2)
    """
    # Setup dimensions
    d2 = d // 2
    volume_center_zyx = torch.tensor([d2, d2, d2]).float()

    # Generate points on an xy plane
    # Create regular grid across xy
    y = torch.linspace(-d2, d2, steps=d)
    x = torch.linspace(-d2, d2, steps=d)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yx = einops.rearrange([yy, xx], "yx h w -> h w yx")
    zyx = torch.nn.functional.pad(yx, (1, 0), value=0)
    xy_plane_zyxw = homogenise_coordinates(zyx)
    xy_plane_zyxw_col = einops.rearrange(xy_plane_zyxw, "h w zyxw -> (h w) zyxw 1")

    # Randomly sample points from grid
    b = len(xy_plane_zyxw_col)
    rng = np.random.default_rng(seed=seed)
    idx_subset = rng.choice(b, size=(n_points_on_plane,), replace=False)

    # Get points and remove homogeneous coordinate
    points_in_scope_zyxw_col = xy_plane_zyxw_col[idx_subset]
    points_in_scope_zyxw = einops.rearrange(
        points_in_scope_zyxw_col, "b zyxw 1 -> b zyxw"
    )
    points_in_scope_zyx = points_in_scope_zyxw[..., :3]

    # Create 3D volume with points on the plane
    points_in_volume = points_in_scope_zyx + volume_center_zyx
    volume_in_scope = torch.zeros((d, d, d)).float()
    volume_in_scope, _ = insert_into_image_3d(
        values=torch.ones((n_points_on_plane,)),
        coordinates=points_in_volume,
        image=volume_in_scope,
    )

    # Setup tilt angles
    if tilt_angles_deg is None:
        tilt_angles_deg = torch.linspace(-60, 60, steps=41)

    # Setup scope2detector transformation
    M_scope2detector = Rz(tilt_axis_angle, zyx=True) @ Ry(tilt_angles_deg, zyx=True)

    # Generate tilt series
    M_scope2detector_rot = M_scope2detector[:, :3, :3]
    M_scope2detector_rot_inv = torch.linalg.pinv(M_scope2detector_rot)
    tilt_series = project_3d_to_2d(
        volume=volume_in_scope,
        rotation_matrices=M_scope2detector_rot_inv,
        zyx_matrices=True,
    )

    # Generate random shifts with 0-degree tilt at 0 shift (anchor)
    shifts = rng.uniform(
        low=-shift_magnitude, high=shift_magnitude, size=(len(tilt_angles_deg), 2)
    )
    shifts = torch.tensor(shifts).float()
    shifts[tilt_angles_deg == 0] = 0  # Keep 0-degree tilt as anchor

    # Apply shifts to misalign tilt series
    tilt_series = fourier_shift_image_2d(tilt_series, shifts=shifts)

    return tilt_series, tilt_angles_deg, tilt_axis_angle, shifts


@pytest.mark.parametrize(
    "shift_magnitude, max_error",
    [
        (0.0, 0.05),  # zero shift baseline
        (2.0, 0.4),  # small shifts
        (5.0, 1.2),  # medium shifts
        (10.0, 1.6),  # large shifts
    ],
)
def test_tiltxcorr_shift_estimation(shift_magnitude: float, max_error: float):
    """
    Test recovery of image shifts in a tilt series.

    The tiltxcorr function should return shifts that when applied will
    recenter the misaligned projection images.
    """
    # Generate tilt series with known shifts
    tilt_series, tilt_angles, tilt_axis_angle, applied_shifts = (
        _generate_shifted_tilt_series(
            shift_magnitude=shift_magnitude,
            d=128,
            n_points_on_plane=100,
            tilt_axis_angle=85.0,
            seed=42,
        )
    )

    # Run tiltxcorr to estimate recentering shifts
    ground_truth_shifts = -1 * applied_shifts
    estimated_shifts = tiltxcorr(
        tilt_series=tilt_series,
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
    )

    # Verify estimated shifts match ground truth within tolerance
    error = torch.abs(estimated_shifts - ground_truth_shifts).max()
    assert error < max_error, (
        f"Shift estimation failed:\n"
        f"  Shift magnitude: {shift_magnitude} pixels\n"
        f"  Max error: {error:.3f} pixels (max allowed: {max_error} pixels)\n"
        f"  Mean error: {torch.abs(estimated_shifts - ground_truth_shifts).mean():.3f} pixels"
    )
