"""Functions for generating a base grid on the S^2 unit sphere."""

import platform
import warnings

import numpy as np
import torch

if platform.system() == "Windows":
    warnings.warn("healpy cannot be installed on Windows systems.", stacklevel=2)
else:
    import healpy as hp

from torch_so3.hierarchical_s2_grid import HierarchicalS2Grid


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
        # At the pole (theta=0), all phi values are equivalent
        # If 0.0 is in range, use it as the canonical value
        if np.abs(theta_all[i]) < 1e-10 and phi_min_rad < 0 <= phi_max_rad:
            phi_values = np.array([0.0], dtype=np.float64)
        theta_values = np.full_like(phi_values, theta_all[i])
        angle_pairs.append(np.stack([phi_values, theta_values], axis=1))

    # Convert back to degrees
    angle_pairs = np.rad2deg(np.concatenate(angle_pairs))

    return torch.tensor(angle_pairs, dtype=torch.float64)


def _nside_from_theta_step(theta_step: float) -> int:
    """Infer the smallest valid HEALPix nside that covers ``theta_step`` resolution.

    Uses ``nside = ceil(sqrt(N/12))`` where ``N`` is the target pixel count from the
    angular step; this always satisfies ``12 * nside**2 >= N``.

    Parameters
    ----------
    theta_step : float
        Angular step for theta in degrees.

    Returns
    -------
    int
        The smallest valid HEALPix nside that covers ``theta_step`` resolution.
    """
    theta_step_rad = np.deg2rad(theta_step)
    estimated_num_pixels = int(4 * np.pi / (theta_step_rad**2))
    nside = int(np.ceil(np.sqrt(estimated_num_pixels / 12)))
    return nside


def _is_power_of_two(n: int) -> bool:
    """Check whether ``n`` is a positive power of two."""
    return n >= 1 and (n & (n - 1)) == 0


def _next_power_of_two_ge(n: int) -> int:
    """Smallest power of two that is >= ``n`` (``n`` must be >= 1)."""
    return 1 << (n - 1).bit_length()


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

    nside = _nside_from_theta_step(theta_step)
    exact_num_pixels = 12 * nside**2

    # Generate the base grid
    pixels = np.arange(exact_num_pixels).astype(np.int64)
    theta_values, phi_values = hp.pix2ang(nside, pixels)
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


def healpix_sectored_base_grid(
    nside_coarse: int,
    nside_fine: int | None = None,
    theta_step: float = 2.5,
    theta_min: float = 0.0,
    theta_max: float = 180.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
) -> torch.Tensor:
    """Generate a two-level HEALPix S2 grid grouped by coarse sector.

    The sphere is divided into ``n_sectors = 12 * nside_coarse**2`` equal-area sectors
    using a coarse HEALPix resolution.  Within each sector a finer HEALPix grid samples
    the direction space.  Because HEALPix is equal-area, every sector contains exactly
    the same number of fine pixels ``k2 = (nside_fine / nside_coarse)**2``.

    Optional ``theta_*`` and ``phi_*`` bounds select **which coarse sectors to keep**:
    if **any** fine-pixel centre in a sector lies within the bounds, the **entire**
    sector is retained (all ``k2`` directions, including centres that fall outside the
    bounds).  Sectors with no centres in range are omitted.  This matches
    :func:`healpix_base_grid` comparisons (no phi wrap-around across 0/360).

    Parameters
    ----------
    nside_coarse : int
        HEALPix ``nside`` for the coarse sector grid. Must be a power of two >= 1.
        ``n_sectors = 12 * nside_coarse**2``.
    nside_fine : int, optional
        HEALPix ``nside`` for the fine sampling inside each sector.  Must be a power
        of two >= ``nside_coarse``.  If ``None``, inferred from ``theta_step`` and
        rounded up to the nearest power of two.
    theta_step : float, optional
        Angular step in degrees used to infer ``nside_fine`` when ``nside_fine`` is
        ``None``.  Ignored if ``nside_fine`` is provided.  Default is 2.5.
    theta_min, theta_max, phi_min, phi_max : float, optional
        In degrees.  Used only to decide sector inclusion; see above.  Defaults cover
        the full sphere (all sectors kept).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(n_kept, k2, 2)`` where ``n_kept <= 12 * nside_coarse**2``,
        last dimension is ``(phi, theta)`` in degrees, and
        ``k2 = (nside_fine // nside_coarse)**2``.  If no sector qualifies, shape
        ``(0, k2, 2)``.
    """
    if platform.system() == "Windows":
        raise ImportError("healpy cannot be installed on Windows systems.")

    nside_coarse = int(nside_coarse)
    if not _is_power_of_two(nside_coarse):
        raise ValueError(
            f"nside_coarse must be a power of 2 and >= 1, got {nside_coarse!r}."
        )

    if nside_fine is not None:
        nside_fine = int(nside_fine)
        if not _is_power_of_two(nside_fine):
            raise ValueError(
                f"nside_fine must be a power of 2 and >= 1, got {nside_fine!r}."
            )
    else:
        target_nside = _nside_from_theta_step(theta_step)
        nside_fine = max(nside_coarse, _next_power_of_two_ge(target_nside))

    if nside_fine < nside_coarse:
        raise ValueError(
            f"nside_fine ({nside_fine}) must be >= nside_coarse ({nside_coarse})."
        )

    # depth_diff levels separate the coarse sector grid from the fine grid
    depth_diff = int(np.log2(nside_fine // nside_coarse))
    grid = HierarchicalS2Grid(nside_finest=nside_fine, n_levels=depth_diff + 1)

    keep_mask, theta_deg, phi_deg = grid.sector_bounds_mask(
        grid.coarsest_level,
        grid.finest_level,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
    )

    k2 = theta_deg.shape[1]
    if not np.any(keep_mask):  # no sectors qualified
        return torch.empty((0, k2, 2), dtype=torch.float64)

    angle_pairs = np.stack([phi_deg[keep_mask], theta_deg[keep_mask]], axis=-1)
    return torch.tensor(angle_pairs, dtype=torch.float64)


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
