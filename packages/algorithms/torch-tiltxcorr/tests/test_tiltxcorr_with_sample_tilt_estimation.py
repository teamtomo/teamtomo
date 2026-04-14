"""Integration tests for tiltxcorr_with_sample_tilt using simulated data."""

import torch
import torch.nn.functional as F
import pytest
import numpy as np
import einops

from torch_affine_utils.transforms_3d import Rx, Ry, Rz
from torch_affine_utils.utils import homogenise_coordinates
from torch_image_interpolation import insert_into_image_3d
from torch_fourier_slice import project_3d_to_2d

from torch_tiltxcorr import tiltxcorr_with_sample_tilt_estimation


def _generate_tilted_plane_tilt_series(
    sample_tilt_x: float = 0.0,
    sample_tilt_y: float = 0.0,
    d: int = 128,
    n_points_on_plane: int = 100,
    tilt_angles_deg: torch.Tensor | None = None,
    tilt_axis_angle: float = 85.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Generate a synthetic tilt series from points on a tilted plane.

    Parameters
    ----------
    sample_tilt_x : float
        Sample tilt around x-axis in degrees
    sample_tilt_y : float
        Sample tilt around y-axis in degrees
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
        Generated tilt series (n_tilts, h, w)
    tilt_angles : torch.Tensor
        Tilt angles in degrees (n_tilts,)
    tilt_axis_angle : float
        Tilt axis angle in degrees
    """
    # Setup dimensions
    d2 = d // 2
    volume_center_zyx = torch.tensor([d2, d2, d2]).float()

    # Setup plane geometry - sample tilt in microscope
    M_tilt_sample_in_scope = Ry(sample_tilt_y, zyx=True) @ Rx(sample_tilt_x, zyx=True)

    # Generate points on a tilted xy plane in the microscope
    # Create regular grid across xy
    y = torch.linspace(-d2, d2, steps=d)
    x = torch.linspace(-d2, d2, steps=d)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yx = einops.rearrange([yy, xx], "yx h w -> h w yx")
    zyx = F.pad(yx, (1, 0), value=0)
    xy_plane_zyxw = homogenise_coordinates(zyx)
    xy_plane_zyxw_col = einops.rearrange(xy_plane_zyxw, "h w zyxw -> (h w) zyxw 1")

    # Randomly sample points from grid
    b = len(xy_plane_zyxw_col)
    rng = np.random.default_rng(seed=seed)
    idx_subset = rng.choice(b, size=(n_points_on_plane,), replace=False)

    # Apply sample tilt transformation to points
    points_in_scope_zyxw_col = xy_plane_zyxw_col[idx_subset]
    points_in_scope_zyxw_col = M_tilt_sample_in_scope @ points_in_scope_zyxw_col
    points_in_scope_zyxw = einops.rearrange(
        points_in_scope_zyxw_col, "b zyxw 1 -> b zyxw"
    )
    points_in_scope_zyx = points_in_scope_zyxw[..., :3]

    # Create 3D volume with points on the tilted plane
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

    return tilt_series, tilt_angles_deg, tilt_axis_angle


@pytest.mark.parametrize(
    "sample_tilt_x, sample_tilt_y, max_error",
    [
        # Y-axis only (stage tilt direction - should be estimated well)
        (0.0, 0.0, 0.05),  # zero tilt baseline
        (0.0, 15.0, 0.05),  # positive Y tilt
        (0.0, -12.0, 0.05),  # negative Y tilt
        (0.0, 25.0, 0.1),  # larger positive Y tilt
        # Combined X and Y tilts (more realistic scenario, expect inaccuracies)
        (3.0, 10.0, 0.75),  # small X, moderate Y
        (-4.0, 10.0, 0.75),  # negative X, positive Y
        (5.0, -10.0, 0.75),  # positive X, negative Y
    ],
)
def test_tiltxcorr_with_sample_tilt_estimation(
    sample_tilt_x: float,
    sample_tilt_y: float,
    max_error: float,
):
    """
    Test recovery of sample tilt with various combinations of X and Y tilts.

    The tiltxcorr_with_sample_tilt function optimizes a single angle parameter,
    which primarily captures the sample tilt around the microscope stage tilt axis).
    X-axis tilts are additional sample tilt perpendicular to the tilt axis.
    """
    # Generate tilt series with specified sample tilts
    tilt_series, tilt_angles, tilt_axis_angle = _generate_tilted_plane_tilt_series(
        sample_tilt_x=sample_tilt_x,
        sample_tilt_y=sample_tilt_y,
        d=128,
        n_points_on_plane=100,
        tilt_axis_angle=85.0,
        seed=42,
    )

    # Run tiltxcorr_with_sample_tilt
    shifts, optimal_sample_tilt = tiltxcorr_with_sample_tilt_estimation(
        tilt_series=tilt_series,
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_tilt_range=(-30.0, 30.0),
        max_iter=15,
    )

    # Verify estimated sample tilt is close to expected value
    error = abs(optimal_sample_tilt - sample_tilt_y)
    assert error < max_error, (
        f"Sample tilt recovery failed:\n"
        f"  Input: sample_tilt_x={sample_tilt_x}°, sample_tilt_y={sample_tilt_y}°\n"
        f"  Expected: {sample_tilt_y}°\n"
        f"  Got: {optimal_sample_tilt:.3f}°\n"
        f"  Error: {error:.3f}° (max allowed: {max_error}°)"
    )
