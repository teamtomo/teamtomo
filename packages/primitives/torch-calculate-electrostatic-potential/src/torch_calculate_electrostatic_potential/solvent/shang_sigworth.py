"""Shang & Sigworth (2012) continuum solvent density."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Table 1, Shang & Sigworth, J Struct Biol 180 (2012).
@dataclass(frozen=True)
class _ShangSigworthParams:
    a2: float
    a3: float
    r1: float
    r2: float
    r3: float
    sig1: float
    sig2: float
    sig3: float


_POLAR = _ShangSigworthParams(
    a2=0.2,
    a3=-0.15,
    r1=0.5,
    r2=1.7,
    r3=1.7,
    sig1=1.0,
    sig2=1.77,
    sig3=1.06,
)
_NONPOLAR = _ShangSigworthParams(
    a2=0.15,
    a3=-0.12,
    r1=1.0,
    r2=2.2,
    r3=3.6,
    sig1=1.0,
    sig2=1.77,
    sig3=0.85,
)

# Carbon uses the nonpolar hydration profile (TEM-sim convention).
_NONPOLAR_ATOMIC_NUMBER = 6
_SQRT2 = math.sqrt(2.0)


def _density_from_params(
    rad_dist: torch.Tensor, params: _ShangSigworthParams
) -> torch.Tensor:
    """Eq. 1, Shang & Sigworth (2012)."""
    erf_term = 0.5 + 0.5 * torch.erf(
        (rad_dist - params.r1) / (_SQRT2 * params.sig1)
    )
    g2 = params.a2 * torch.exp(
        -((rad_dist - params.r2) ** 2) / (2.0 * params.sig2**2)
    )
    g3 = params.a3 * torch.exp(
        -((rad_dist - params.r3) ** 2) / (2.0 * params.sig3**2)
    )
    return erf_term + g2 + g3


def shang_sigworth_density(
    dist_map: torch.Tensor,
    nearest_atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    """Normalized continuum water density from surface distance.

    Density asymptotes to ~1 in bulk solvent. Carbon (Z=6) uses the nonpolar
    Table-1 parameters; all other nearest atoms use the polar set.

    Parameters
    ----------
    dist_map : torch.Tensor
        Surface distances in Angstroms.
    nearest_atomic_numbers : torch.Tensor
        Nearest-atom Z map (same shape as ``dist_map``).

    Returns
    -------
    torch.Tensor
        Relative water density (dimensionless), same shape as ``dist_map``.
    """
    if dist_map.shape != nearest_atomic_numbers.shape:
        raise ValueError(
            "dist_map and nearest_atomic_numbers must have the same shape, "
            f"got {tuple(dist_map.shape)} and {tuple(nearest_atomic_numbers.shape)}"
        )
    polar = _density_from_params(dist_map, _POLAR)
    nonpolar = _density_from_params(dist_map, _NONPOLAR)
    is_nonpolar = nearest_atomic_numbers == _NONPOLAR_ATOMIC_NUMBER
    # Unvisited voxels (Z=0) use polar bulk profile (ρ→1 for large dist).
    return torch.where(is_nonpolar, nonpolar, polar)
