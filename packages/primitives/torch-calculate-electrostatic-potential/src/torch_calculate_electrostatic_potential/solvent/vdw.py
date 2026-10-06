"""Van der Waals radii for solvent-exclusion geometry."""

from __future__ import annotations

import torch

# Wikipedia / TEM-simulator set. Negative or missing → unknown (skip atom).
VDW_RADII_A: dict[int, float] = {
    1: 1.1,  # H
    6: 1.7,  # C
    7: 1.55,  # N
    8: 1.52,  # O
    11: 2.27,  # Na
    12: 1.73,  # Mg
    15: 1.8,  # P
    16: 1.8,  # S
    26: -1.0,  # Fe (unknown in this table)
    79: 1.66,  # Au
}

_MAX_KNOWN_Z = max(VDW_RADII_A.keys())


def vdw_radii_for_atomic_numbers(
    atomic_numbers: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Look up VdW radii (Angstroms) for each atomic number.

    Unknown or explicitly unavailable elements return a negative radius so
    callers can skip them.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        Integer atomic numbers, any shape.
    device : torch.device | None
        Device for the output table lookup tensors.
    dtype : torch.dtype
        Floating dtype for radii.

    Returns
    -------
    torch.Tensor
        Radii in Angstroms, same shape as ``atomic_numbers``. Values ``< 0``
        mean the atom should be skipped for solvent geometry.
    """
    if device is None:
        device = atomic_numbers.device
    table = torch.full((_MAX_KNOWN_Z + 1,), -1.0, dtype=dtype, device=device)
    for z, radius in VDW_RADII_A.items():
        table[z] = radius
    z = atomic_numbers.to(device=device, dtype=torch.int64)
    out_of_range = (z < 0) | (z > _MAX_KNOWN_Z)
    z_clamped = z.clamp(0, _MAX_KNOWN_Z)
    radii = table[z_clamped]
    radii = torch.where(out_of_range, torch.full_like(radii, -1.0), radii)
    return radii
