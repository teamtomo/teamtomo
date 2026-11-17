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

    Near the pole (theta, phi) are polar coordinates of the tilt: theta is how far and
    phi is which way. The cap is therefore every tilt up to coarse_theta_step, in every
    direction. The in-plane part of Rz(phi) Ry(theta) Rz(psi) near the pole is
    phi + psi, so psi is measured from phi: each offset is Rz(phi) Ry(theta)
    Rz(psi - phi), a tilt of theta toward phi followed by an in-plane rotation of psi
    in [-coarse_psi_step, coarse_psi_step]. The offsets cover rotations about all
    three axes, whether composed on the left or the right of a rotation, and always
    include the identity (0, 0, 0).

    The base grid sets how the cap of tilts is sampled:

    - "uniform": theta rings fine_theta_step apart, with a phi step of about
      fine_theta_step / sin(theta) on each ring.
    - "healpix": HEALPix pixel centres in the cap, plus the pole itself (HEALPix has
      no pixel centred on it).
    - "cartesian": a square lattice of tilts, fine_theta_step apart along the two tilt
      axes, inside the disk of radius coarse_theta_step.

    Parameters
    ----------
    coarse_phi_step : float
        Unused; kept for backwards compatibility. phi always covers the full circle.
    coarse_theta_step : float
        Coarse step size for theta in degrees: the radius of the cap of tilts.
    coarse_psi_step : float
        Coarse step size for psi in degrees: the in-plane range is +- this.
    fine_phi_step : float
        Unused; kept for backwards compatibility.
    fine_theta_step : float
        Finer step size for theta (the tilt) in degrees.
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
        euler_angles = _cartesian_tilt_cap(
            coarse_theta_step, coarse_psi_step, fine_theta_step, fine_psi_step
        )
    else:
        # A cap of view directions around the pole, then an in-plane offset. Theta
        # starts at 0 and phi runs over the full circle; limiting phi to a few degrees
        # keeps a single tilt direction and loses a rotation axis.
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
        # The base grid's theta rings can overshoot theta_max when coarse_theta_step
        # isn't a whole multiple of fine_theta_step; keep the cap.
        euler_angles = euler_angles[euler_angles[:, 1] <= coarse_theta_step + EPS]
        if not bool((euler_angles[:, 1].abs() < EPS).any()):
            # No base-grid point on the pole (HEALPix): add the untilted offsets.
            psi_all = _psi_values(coarse_psi_step, fine_psi_step, euler_angles.dtype)
            pole = torch.stack(
                [torch.zeros_like(psi_all), torch.zeros_like(psi_all), psi_all], dim=1
            )
            euler_angles = torch.cat([pole, euler_angles], dim=0)
        # Measure psi from phi, so each offset is a tilt of theta toward phi,
        # Rz(phi) Ry(theta) Rz(-phi), followed by an in-plane rotation of psi.
        euler_angles[:, 2] = _wrap_180(euler_angles[:, 2] - euler_angles[:, 0])

    return euler_angles


def _psi_values(
    coarse_psi_step: float, fine_psi_step: float, dtype: torch.dtype
) -> torch.Tensor:
    """In-plane offsets in [-coarse_psi_step, coarse_psi_step], as the base grids."""
    if fine_psi_step <= 0.0 or coarse_psi_step <= 0.0:
        return torch.zeros(1, dtype=dtype)
    return torch.arange(
        -coarse_psi_step, coarse_psi_step + EPS, fine_psi_step, dtype=dtype
    )


def _wrap_180(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles in degrees to [-180, 180)."""
    return torch.remainder(angles + 180.0, 360.0) - 180.0


def _cartesian_tilt_cap(
    coarse_theta_step: float,
    coarse_psi_step: float,
    fine_theta_step: float,
    fine_psi_step: float,
) -> torch.Tensor:
    """Square lattice of tilts inside the cap, times the in-plane offsets.

    A tilt with rotation vector (t_x, t_y, 0) is Rz(phi) Ry(theta) Rz(-phi) with
    theta = |t| and (-sin(phi), cos(phi)) = t / |t|.
    """
    dtype = torch.float64
    if fine_theta_step > 0.0 and coarse_theta_step > 0.0:
        n = int(coarse_theta_step / fine_theta_step + EPS)
        steps = torch.arange(-n, n + 1, dtype=dtype) * fine_theta_step
    else:
        steps = torch.zeros(1, dtype=dtype)
    t_x, t_y = torch.meshgrid(steps, steps, indexing="ij")
    t_x, t_y = t_x.flatten(), t_y.flatten()
    theta = torch.hypot(t_x, t_y)
    keep = theta <= coarse_theta_step + EPS
    t_x, t_y, theta = t_x[keep], t_y[keep], theta[keep]
    phi = torch.remainder(torch.rad2deg(torch.atan2(-t_x, t_y)), 360.0)

    psi_all = _psi_values(coarse_psi_step, fine_psi_step, dtype)
    phi_mesh = phi.repeat(psi_all.numel())
    theta_mesh = theta.repeat(psi_all.numel())
    psi_mesh = psi_all.repeat_interleave(phi.numel())
    return torch.stack([phi_mesh, theta_mesh, _wrap_180(psi_mesh - phi_mesh)], dim=1)


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
