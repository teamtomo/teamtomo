"""Translate the canonical-API pose / Ewald arguments into kernel conventions."""

from __future__ import annotations

import torch
from torch_ctf import calculate_relativistic_electron_wavelength


def to_zyx_matrices(
    rotation_matrices: torch.Tensor, zyx_matrices: bool
) -> torch.Tensor:
    """Rotation matrices acting on zyx vectors, as the kernels expect.

    xyz matrices (``zyx_matrices=False``) are converted by flipping the last two
    axes, as in the canonical layer.
    """
    if zyx_matrices:
        return rotation_matrices
    return torch.flip(rotation_matrices, dims=(-2, -1))


def to_zyx_vectors(vectors: torch.Tensor | None, zyx: bool) -> torch.Tensor | None:
    """Directions / shifts in zyx (or yx) component order, as the kernels expect.

    xyz (or xy) vectors (``zyx=False``) are converted by flipping the last axis.
    """
    if vectors is None or zyx:
        return vectors
    return torch.flip(vectors, dims=(-1,))


def ewald_coefficient(
    sidelength: int,
    apply_ewald_curvature: bool,
    ewald_voltage_kv: float,
    ewald_flip_sign: bool,
    ewald_px_size: float,
) -> float:
    """Signed Ewald z-offset coefficient for the kernels (0 = flat slice).

    The canonical layer bends the slice by ``dz = wavelength * |k|^2 / 2`` in
    physical units; in index units of a volume of side ``sidelength`` that is
    ``dz = wavelength / (2 * px_size * sidelength) * |k|^2``.
    """
    if not apply_ewald_curvature:
        return 0.0
    wavelength = 1e10 * float(  # Angstroms
        calculate_relativistic_electron_wavelength(energy=ewald_voltage_kv * 1e3)
    )
    coefficient = wavelength / (2.0 * ewald_px_size * sidelength)
    return -coefficient if ewald_flip_sign else coefficient
