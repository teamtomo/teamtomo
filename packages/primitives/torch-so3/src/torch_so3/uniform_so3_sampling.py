"""Generates a set of Euler angles that uniformly samples SO(3) using Hopf fibration."""

import warnings
from typing import Literal, Optional

import torch

from torch_so3.base_s2_grid import (
    cartesian_base_grid,
    healpix_base_grid,
    healpix_sectored_base_grid,
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


def get_sectored_euler_angles(
    nside_coarse: int,
    nside_fine: Optional[int] = None,
    theta_step: float = 2.5,
    psi_step: float = 1.5,
    psi_min: float = 0.0,
    psi_max: float = 360.0,
    theta_min: float = 0.0,
    theta_max: float = 180.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
) -> torch.Tensor:
    """Generate ZYZ Euler angles grouped by coarse HEALPix sector.

    The sphere is divided into ``n_sectors = 12 * nside_coarse**2`` equal-area sectors.
    Within each sector a finer HEALPix grid (controlled by ``nside_fine`` or
    ``theta_step``) provides the directional sampling, and those directions are combined
    with a full sweep of the in-plane angle ``psi``.  Coarse sectors are dropped unless
    at least one fine-pixel centre lies in the given ``(theta, phi)`` window; if kept,
    the sector still contains **all** ``k2`` fine directions.

    Parameters
    ----------
    nside_coarse : int
        HEALPix ``nside`` for the coarse sector grid.  ``n_sectors = 12 *
        nside_coarse**2``.
    nside_fine : int, optional
        HEALPix ``nside`` for the fine sampling inside each sector.  Must be >=
        ``nside_coarse``.  If ``None``, inferred from ``theta_step``.
    theta_step : float, optional
        Angular step in degrees used to infer ``nside_fine`` when ``nside_fine`` is
        ``None``.  Default is 2.5.
    psi_step : float, optional
        Angular step for the in-plane rotation ``psi`` in degrees.  Default is 1.5.
    psi_min : float, optional
        Minimum value for ``psi`` in degrees.  Default is 0.0.
    psi_max : float, optional
        Maximum (exclusive) value for ``psi`` in degrees.  Default is 360.0.
    theta_min, theta_max, phi_min, phi_max : float, optional
        In degrees.  Passed to
        :func:`~torch_so3.base_s2_grid.healpix_sectored_base_grid` to select coarse
        sectors (any fine centre in range keeps the full sector).

    Returns
    -------
    torch.Tensor
        Euler angles in degrees of shape ``(n_kept, n_per_sector, 3)`` with columns
        ``(phi, theta, psi)``.  ``n_per_sector = k2 * n_psi`` where
        ``k2 = (nside_fine / nside_coarse)**2`` and
        ``n_psi = len(arange(psi_min, psi_max, psi_step))``.  ``n_kept`` is the number
        of coarse sectors with at least one fine direction in the angular window.
    """
    # S2 base grid: (n_kept, k2, 2)
    s2 = healpix_sectored_base_grid(
        nside_coarse=nside_coarse,
        nside_fine=nside_fine,
        theta_step=theta_step,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
    )
    n_sectors, k2, _ = s2.shape

    # Build psi grid (same logic as get_uniform_euler_angles)
    if psi_min >= psi_max:
        psi_all = torch.tensor([psi_min], dtype=torch.float64)
    else:
        psi_all = torch.arange(psi_min, psi_max, psi_step, dtype=torch.float64)
    n_psi = psi_all.size(0)

    # Tile S2 points within each sector: each of k2 directions repeated n_psi times
    # s2: (n_sectors, k2, 2) -> (n_sectors, k2*n_psi, 2)
    phi_theta = s2.repeat_interleave(n_psi, dim=1)

    # Tile psi values: [psi0, psi1, ...] repeated k2 times per sector
    # psi_all: (n_psi,) -> (n_sectors, k2*n_psi, 1)
    psi_col = psi_all.repeat(k2).unsqueeze(0).expand(n_sectors, -1).unsqueeze(-1)

    # Ordering of angles is (phi, theta, psi) for ZYZ intrinsic rotations.
    # Return shape: (n_kept, k2 * n_psi, 3) — one row per sector kept after angular
    # filtering; along dim 1, each of the k2 directions is followed by all n_psi psi
    # values (repeat_interleave order).
    return torch.cat([phi_theta, psi_col], dim=-1).contiguous()
