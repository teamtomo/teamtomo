"""Generate finer angular search around multiple selected Euler angles."""

from typing import Literal

import torch

from torch_so3.uniform_so3_sampling import get_uniform_euler_angles

EPS = 1e-6


def get_local_high_resolution_angles(
    coarse_phi_step: float = 2.5,
    coarse_theta_step: float = 2.5,
    coarse_psi_step: float = 1.5,
    fine_phi_step: float = 0.1,
    fine_theta_step: float = 0.1,
    fine_psi_step: float = 0.1,
    base_grid_method: Literal["uniform", "healpix", "cartesian"] = "uniform",
) -> torch.Tensor:
    """Local orientation refinement from a coarse to fine grid.

    This function creates a high-resolution grid of Euler angles around the pole to
    sample all points within the coarse grid. It creates a "disk" (cap) of view
    directions on the S^2 (theta, phi) sphere and also samples the in-plane angle (psi).

    For the "uniform" and "healpix" base grids, the cap is theta in
    [0, coarse_theta_step] with phi over the full circle: near the pole (theta, phi)
    are polar coordinates of the tilt (theta is how far, phi is which way). Near the
    pole the in-plane part of Rz(phi) Ry(theta) Rz(psi) is phi + psi, so psi is
    measured from phi: each offset is Rz(phi) Ry(theta) Rz(psi - phi), a tilt of theta
    toward phi followed by an in-plane rotation of psi in
    [-coarse_psi_step, coarse_psi_step]. The offsets therefore cover rotations about
    all three axes, whether composed on the left or the right of a rotation.

    Parameters
    ----------
    coarse_phi_step : float
        Coarse step size for phi in degrees. Only used by the "cartesian" base grid.
    coarse_theta_step : float
        Coarse step size for theta in degrees.
    coarse_psi_step : float
        Coarse step size for psi in degrees.
    fine_phi_step : float
        Finer step size for phi in degrees. Only used by the "cartesian" base grid.
    fine_theta_step : float
        Finer step size for theta in degrees.
    fine_psi_step : float
        Finer step size for psi in degrees.
    base_grid_method : Literal["uniform", "healpix", "cartesian"]
        Method to generate the base grid.

    Returns
    -------
    euler_angles : torch.Tensor
        Tensor of shape (N, 3) containing Euler angles in degrees, where N is the
        number of angles generated. Angles exist around pole (0, 0, 0) and define grid
        to search over.
    """
    if base_grid_method == "cartesian":
        euler_angles = get_uniform_euler_angles(
            phi_step=fine_phi_step,
            theta_step=fine_theta_step,
            psi_step=fine_psi_step,
            phi_min=-coarse_phi_step,  # Completely sample for uniform base grid?
            phi_max=coarse_phi_step,
            theta_min=-coarse_theta_step,
            theta_max=coarse_theta_step,
            psi_min=-coarse_psi_step,
            psi_max=coarse_psi_step + EPS,
            base_grid_method=base_grid_method,
        )
    else:
        # A cap of view directions around the pole, then an in-plane offset. Near the
        # pole (theta, phi) are polar coordinates of the tilt: theta is how far, phi is
        # which way. So theta starts at 0 and phi runs over the full circle; limiting
        # phi to a few degrees keeps a single tilt direction and loses a rotation axis.
        euler_angles = get_uniform_euler_angles(
            psi_step=fine_psi_step,
            theta_step=fine_theta_step,
            phi_min=0.0,
            phi_max=360.0,
            theta_min=0.0,
            theta_max=coarse_theta_step,
            psi_min=-coarse_psi_step,
            psi_max=coarse_psi_step + EPS,
            base_grid_method=base_grid_method,
        )
        # Near the pole the in-plane part of Rz(phi) Ry(theta) Rz(psi) is phi + psi.
        # Measure psi from phi, so each offset is a tilt of theta toward phi,
        # Rz(phi) Ry(theta) Rz(-phi), followed by an in-plane rotation of psi.
        phi = euler_angles[:, 0]
        euler_angles[:, 2] = (
            torch.remainder(euler_angles[:, 2] - phi + 180.0, 360.0) - 180.0
        )

    return euler_angles


def get_roll_angles(
    psi_step: float = 0.5,
    psi_min: float = -10.0,
    psi_max: float = 10.0,
    theta_step: float = 0.5,
    theta_min: float = -10.0,
    theta_max: float = 10.0,
    roll_axis: torch.Tensor = None,
    roll_axis_step: float = 2.0,
) -> torch.Tensor:
    """Generate a grid of Euler angles for rotations around Z and a roll axis.

    This function generates ZYZ Euler angles (in degrees) that represent:
    1. Rotations around the Z axis (primary rotation)
    2. Rotations around a "roll axis" that's in the XY plane
       (orthogonal to Z but not necessarily aligned with Y)

    If no specific roll axis is provided, it samples multiple possible roll axes
    by varying angles in the XY plane and using complementary rotations that
    sum to the desired psi angle.

    Parameters
    ----------
    psi_step : float, optional
        Step size for the in-plane rotation (psi) in degrees, by default 0.5
    psi_min : float, optional
        Minimum value for psi in degrees, by default -10.0
    psi_max : float, optional
        Maximum value for psi in degrees, by default 10.0
    theta_step : float, optional
        Step size for the roll rotation (theta) in degrees, by default 0.5
    theta_min : float, optional
        Minimum value for theta in degrees, by default -10.0
    theta_max : float, optional
        Maximum value for theta in degrees, by default 10.0
    roll_axis : torch.Tensor, optional
        If provided, specifies a fixed roll axis as [x, y] in Cartesian coordinates.
        If None, searches through multiple roll axes, by default None.
    roll_axis_step : float, optional
        Angular step for searching different roll axes in degrees, by default 2.0

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 3) containing Euler angles (phi, theta, psi) in degrees
        representing the sampled rotations.
    """
    # Generate arrays for psi (rotation around Z) and theta (roll angle)
    psi_values = torch.arange(psi_min, psi_max + EPS, psi_step, dtype=torch.float32)
    theta_values = torch.arange(
        theta_min, theta_max + EPS, theta_step, dtype=torch.float32
    )

    # Set up the roll axis search
    if roll_axis is None:
        # Search for roll axes by sampling angles in the XY plane
        roll_angles = torch.arange(0, 180, roll_axis_step, dtype=torch.float32)
    else:
        # Use the given roll axis [x,y] to determine the roll angle
        # Convert from Cartesian coordinates to angle in the XY plane
        roll_angle = torch.rad2deg(torch.atan2(roll_axis[1], roll_axis[0]))
        # Normalize to 0-180 range
        if roll_angle < 0:
            roll_angle += 180
        roll_angles = torch.tensor([roll_angle], dtype=torch.float32)

    # Create meshgrid for all dimensions
    roll_grid, theta_grid, psi_grid = torch.meshgrid(
        roll_angles, theta_values, psi_values, indexing="ij"
    )

    # Flatten all grids
    roll_flat = roll_grid.reshape(-1)
    theta_flat = theta_grid.reshape(-1)
    psi_flat = psi_grid.reshape(-1)

    # For each combination (I do extrinsic in my head so go backwards):
    # 1. First Euler angle (phi): Rotate to align back with the psi_angle
    # 2. Second Euler angle (theta): Apply theta rotation around roll axis
    # 3. Third Euler angle (psi): Rotate to align with roll axis

    # Second Euler angle: theta rotation around roll axis
    theta = theta_flat

    # Third Euler angle: rotate back to achieve the desired psi rotation
    # We rotate back by (roll_angle - psi_value) to get a net psi rotation
    psi = 360 - roll_flat

    # Adjust back-rotation to achieve desired rotation axis angle
    phi = roll_flat + psi_flat

    # Ensure phi is within [0, 360) range
    phi = phi % 360

    # Stack the Euler angles
    all_angles = torch.stack([phi, theta, psi], dim=1)

    return all_angles
