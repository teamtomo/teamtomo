"""Generates a set of Euler angles that uniformly samples SO(3) using Hopf fibration."""

import warnings
from typing import Literal, Optional

import torch

from torch_so3.base_s2_grid import (
    cartesian_base_grid,
    healpix_base_grid,
    uniform_base_grid,
)


def get_uniform_euler_angles(
    psi_step: float = 1.5,
    theta_step: float = 2.5,
    phi_step: Optional[float] = None,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    theta_min: float = 0.0,
    theta_max: float = 180.0,
    psi_min: float = 0.0,
    psi_max: float = 360.0,
    base_grid_method: Literal["uniform", "healpix", "cartesian"] = "uniform",
) -> torch.Tensor:
    """Generate sets of uniform Euler angles (ZYZ) using Hopf fibration.

    Parameters
    ----------
    psi_step: float, optional
        Angular step for psi in degrees. Default is 1.5 degrees.
    theta_step: float, optional
        Angular step for theta in degrees. Default is 2.5
        degrees.
    phi_step: float, optional
        Angular step for phi rotation in degrees. Only used when base_grid_method is
        "cartesian". Default is 2.5 degrees.
    phi_min: float, optional
        Minimum value for phi in degrees. Default is 0.0.
    phi_max: float, optional
        Maximum value for phi in degrees. Default is 360.0.
    theta_min: float, optional
        Minimum value for theta in degrees. Default is 0.0.
    theta_max: float, optional
        Maximum value for theta in degrees. Default is 180.0.
    psi_min: float, optional
        Minimum value for psi in degrees. Default is 0.0.
    psi_max: float, optional
        Maximum value for psi in degrees. Default is 360.0.
    base_grid_method: str, optional
        String literal specifying the method to generate the base grid. Default is
        "uniform". Options are "uniform", "healpix", and "cartesian".

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 3) containing Euler angles in degrees, where N is the
        number of angles generated.
    """
    # TODO: Validation of inputs, wrapping between zero and 2*pi, etc.

    if base_grid_method == "cartesian":
        # Handle cartesian_base_grid separately since it has a different signature
        actual_phi_step = 2.5 if phi_step is None else phi_step
        base_grid = cartesian_base_grid(
            theta_step=theta_step,
            phi_step=actual_phi_step,
            theta_min=theta_min,
            theta_max=theta_max,
            phi_min=phi_min,
            phi_max=phi_max,
        )
    else:
        # Handle uniform and healpix grids
        if base_grid_method == "uniform":
            base_grid_mth = uniform_base_grid
        elif base_grid_method == "healpix":
            base_grid_mth = healpix_base_grid
        else:
            raise ValueError(f"Invalid base grid method {base_grid_method}.")

        # Check if phi_step was specified for non-cartesian methods
        if phi_step is not None:
            warnings.warn(
                f"phi_step is being ignored for {base_grid_method} method.",
                stacklevel=2,
            )

        base_grid = base_grid_mth(
            theta_step=theta_step,
            theta_min=theta_min,
            theta_max=theta_max,
            phi_min=phi_min,
            phi_max=phi_max,
        )

    # Mesh-grid-like operation to include the in-plane rotation
    if psi_min >= psi_max:
        psi_all = torch.tensor([psi_min], dtype=torch.float64)
    else:
        psi_all = torch.arange(psi_min, psi_max, psi_step, dtype=torch.float64)

    psi_mesh = psi_all.repeat_interleave(base_grid.size(0))
    base_grid = base_grid.repeat(psi_all.size(0), 1)

    # Ordering of angles is (phi, theta, psi) for ZYZ intrinsic rotations
    # psi is the in-plane rotation
    all_angles = torch.cat([base_grid, psi_mesh[:, None]], dim=1)

    return all_angles
