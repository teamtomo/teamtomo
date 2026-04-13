"""Ewald sphere curvature calculations for Fourier slice extraction."""

import torch
from torch_ctf import calculate_relativistic_electron_wavelength


def _calculate_ewald_z(
    voltage_kv: float,
    k_xy: torch.Tensor,
    flip_sign: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the Ewald sphere z-offset Δz(|k|) for a given acceleration voltage.

    The basic relation used here is

        Δz(|k|) = λ / (2 d^2)  with  d = 1 / |k|

    i.e.

        Δz(|k|) = (λ |k|^2) / 2

    where
        - λ is the wavelength (computed from voltage),
        - |k| is the spatial frequency magnitude in inverse length units.

    Parameters
    ----------
    voltage_kv:
        Acceleration voltage in kV. Wavelength is computed from this using
        relativistic electron wavelength formula. Converted to energy in eV
        (energy = voltage_kv * 1e3) for the calculation.
    k_xy:
        In-plane spatial frequency magnitude |k_xy| (e.g. 1 / Å). This is
        typically constructed from a 2D Fourier grid as

            |k_xy| = sqrt(k_x^2 + k_y^2)

    flip_sign:
        If True, return -Δz so that the curvature can be applied in the
        opposite direction.
    eps:
        Small clamp value to avoid division by zero at |k| ≈ 0.

    Returns
    -------
    delta_z:
        Δz(|k|) with the same shape as `k_xy`.
    """
    energy_ev = voltage_kv * 1e3
    wavelength = calculate_relativistic_electron_wavelength(energy=energy_ev)
    wavelength = wavelength * 1e10  # convert to Angstroms
    k_xy_clamped = torch.clamp(k_xy, min=eps)
    delta_z = wavelength * (k_xy_clamped**2) / 2.0
    if flip_sign:
        delta_z = -delta_z
    return delta_z


def _apply_ewald_curvature(
    freq_grid: torch.Tensor,
    voltage_kv: float,
    flip_sign: bool = False,
    px_size: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Modify the z-coordinate of a central slice frequency grid to follow a curve.

    Parameters
    ----------
    freq_grid:
        `(h, w, 3)` tensor of zyx frequencies (cycles / pixel), as returned by
        `_central_slice_fftfreq_grid`.
    voltage_kv:
        Acceleration voltage in kV. Wavelength is computed from this using
        relativistic electron wavelength formula.
    flip_sign:
        If True, apply the curvature with opposite sign.
    px_size:
        Pixel size (e.g. Å / pixel). Used to convert grid frequencies
        (cycles / pixel) to physical units.
    eps:
        Clamp value to avoid division by zero in the Ewald formula.

    Returns
    -------
    curved_grid:
        `(h, w, 3)` tensor of zyx frequencies lying on the curved surface.
    """
    # freq_grid[..., 0] = z, 1 = y, 2 = x  (zyx convention)
    kx_grid = freq_grid[..., 2]
    ky_grid = freq_grid[..., 1]

    # Magnitude in grid units (cycles / pixel)
    k_xy_grid = torch.sqrt(kx_grid**2 + ky_grid**2)

    # Convert to physical units (1 / length)
    k_xy_phys = k_xy_grid / px_size

    delta_z_phys = _calculate_ewald_z(
        voltage_kv=voltage_kv,
        k_xy=k_xy_phys,
        flip_sign=flip_sign,
        eps=eps,
    )

    # Back to grid units along z (cycles / pixel)
    delta_z_grid = delta_z_phys * px_size

    curved_grid = freq_grid.clone()
    curved_grid[..., 0] = delta_z_grid
    return curved_grid
