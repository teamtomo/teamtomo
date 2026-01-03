"""Functions for generating a base grid on the S^2 unit sphere."""

import platform
import warnings

import numpy as np
import torch
if platform.system() == "Windows":
    warnings.warn("healpy cannot be installed on Windows systems.", stacklevel=2)
else:
    import healpy as hp


def uniform_base_grid(
    theta_step: float = 2.5,
    theta_min: float = 0.0,
    theta_max: float = 180.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
) -> torch.Tensor:
    """Generate a uniform base grid on the S^2 sphere.

    phi step is calculated by the position on the sphere (sin(theta))


    Parameters
    ----------
    theta_step : float, optional
        Angular step for theta in degrees. Default is 2.5
    theta_min : float, optional
        Minimum value for theta in degrees. Default is 0.0.
    theta_max : float, optional
        Maximum value for theta in degrees. Default is 180.0.
    phi_min : float, optional
        Minimum value for phi in degrees. Default is 0.0.
    phi_max : float, optional
        Maximum value for phi in degrees. Default is 360.0.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2) containing theta and phi values in degrees, where N is
        the number of angles pairs generated.
    """
    theta_step_rad = np.deg2rad(theta_step)
    phi_min_rad = np.deg2rad(phi_min)
    phi_max_rad = np.deg2rad(phi_max)

    # generate uniform set of theta values
    theta_all = np.arange(
        theta_min, theta_max + theta_step, theta_step, dtype=np.float64
    )
    theta_all = np.deg2rad(theta_all)

    # Phi step increment is modulated by the position on the sphere (sin(theta)), but
    # don't allow it to exceed the maximum step size
    phi_max_step_rad = phi_max_rad - phi_min_rad
    phi_step_all = np.abs(theta_step_rad / (np.sin(theta_all) + 1e-6))
    phi_step_all = np.clip(phi_step_all, a_min=None, a_max=phi_max_step_rad)
    if phi_max_step_rad > 0.0:
        phi_step_all = phi_max_step_rad / np.round(phi_max_step_rad / phi_step_all)
    else:
        phi_step_all *= 0.0

    # Now generate the angle pairs
    angle_pairs = []
    for i, phi_step in enumerate(phi_step_all):
        if phi_min_rad >= phi_max_rad or phi_step <= 0.0:
            phi_values = np.array([phi_min_rad], dtype=np.float64)
        else:
            phi_values = np.arange(phi_min_rad, phi_max_rad, phi_step, dtype=np.float64)
        theta_values = np.full_like(phi_values, theta_all[i])
        angle_pairs.append(np.stack([phi_values, theta_values], axis=1))

    # Convert back to degrees
    angle_pairs = np.rad2deg(np.concatenate(angle_pairs))

    return torch.tensor(angle_pairs, dtype=torch.float64)


def healpix_base_grid(
    theta_step: float = 2.5,
    theta_min: float = 0.0,
    theta_max: float = 180.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
) -> torch.Tensor:
    """Generate a base grid on the S^2 sphere using HEALPix.

    phi step is calculated by the position on the sphere

    Parameters
    ----------
    theta_step : float, optional
        Angular step for theta in degrees. Default is 2.5
    theta_min : float, optional
        Minimum value for theta in degrees. Default is 0.0.
    theta_max : float, optional
        Maximum value for theta in degrees. Default is 180.0.
    phi_min : float, optional
        Minimum value for phi in degrees. Default is 0.0.
    phi_max : float, optional
        Maximum value for phi in degrees. Default is 360.0.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2) containing theta and phi values in degrees, where N is
        the number of angles pairs generated.
    """
    if platform.system() == "Windows":
        raise ImportError("healpy cannot be installed on Windows systems.")

    theta_step_rad = np.deg2rad(theta_step)

    estimated_num_pixels = int(4 * np.pi / (theta_step_rad * theta_step_rad))

    # Find the next largest npix value for a valid healpix grid
    exact_num_pixels = 0
    nside = np.ceil(np.sqrt(estimated_num_pixels / 12))
    while nside < 36:
        exact_num_pixels = 12 * nside**2
        if exact_num_pixels >= estimated_num_pixels:
            break
        nside += 1

    # Check to make sure the break condition was met
    if exact_num_pixels < estimated_num_pixels:
        raise ValueError("No valid nside found")

    # Generate the base grid
    pixels = np.arange(exact_num_pixels).astype(np.int64)
    theta_values, phi_values = hp.pix2ang(int(nside), pixels)
    theta_values = torch.tensor(np.rad2deg(theta_values), dtype=torch.float64)
    phi_values = torch.tensor(np.rad2deg(phi_values), dtype=torch.float64)

    # Remove values outside the desired range
    valid_indices = (
        (theta_values >= theta_min)
        & (theta_values <= theta_max)
        & (phi_values >= phi_min)
        & (phi_values <= phi_max)
    )

    # NOTE: This is doing batched indexing to not exceed memory limits
    if not torch.all(valid_indices):
        theta_result = []
        phi_result = []
        valid_indices = torch.nonzero(valid_indices).squeeze()
        for i in range(0, len(valid_indices), 256):
            batch_indices = valid_indices[i : i + 256]
            theta_result.append(theta_values[batch_indices])
            phi_result.append(phi_values[batch_indices])

        theta_values = torch.cat(theta_result)
        phi_values = torch.cat(phi_result)

    angle_pairs = torch.stack([phi_values, theta_values], dim=1)

    return angle_pairs


def cartesian_base_grid(
    theta_step: float = 2.5,
    phi_step: float = 1.5,
    theta_min: float = 0.0,
    theta_max: float = 180.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
) -> torch.Tensor:
    """Generate a base grid on the S^2 sphere using a cartesian grid.

    phi step is now set explicitly. This will oversample the poles.

    Parameters
    ----------
    theta_step : float, optional
        Angular step for theta in degrees. Default is 2.5
    phi_step : float, optional
        Angular step for phi in degrees. Default is 1.5 degrees.
    theta_min : float, optional
        Minimum value for theta in degrees. Default is 0.0.
    theta_max : float, optional
        Maximum value for theta in degrees. Default is 180.0.
    phi_min : float, optional
        Minimum value for phi in degrees. Default is 0.0.
    phi_max : float, optional
        Maximum value for phi in degrees. Default is 360.0.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2) containing theta and phi values in degrees, where N is
        the number of angles pairs generated.
    """
    # Generate the base grid
    phi_values = torch.arange(
        phi_min,
        phi_max,
        phi_step,
        dtype=torch.float64,
    )

    theta_values = torch.arange(
        theta_min,
        theta_max,
        theta_step,
        dtype=torch.float64,
    )

    grid = torch.meshgrid(phi_values, theta_values, indexing="ij")
    euler_angles = torch.stack(grid, dim=-1).reshape(-1, 2)

    return euler_angles
