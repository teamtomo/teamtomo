"""High-level multislice wrapper.

Wires up frequency grids, wavelength, and the per-slice loop on top of the
pure-math primitives in `_core`.
"""

import torch

from torch_scattering._core import (
    _prepare_propagation_parameters,
    chunk_slices,
    fresnel_propagator,
    multislice_step,
)


def multislice(
    potential: torch.Tensor,
    pixel_size: float | torch.Tensor,
    energy: float | torch.Tensor,
    n_slices: int | None = None,
) -> torch.Tensor:
    """
    Compute the 2D exit wave from a 3D scattering potential.

    Parameters
    ----------
    potential : torch.Tensor
        Complex-valued 3D scattering potential, shape (..., Z, H, W), where
        Z is the number of slices along the beam direction.
    pixel_size : float | torch.Tensor
        Pixel size in Angstroms. The slice thickness `dz` is assumed to
        equal `pixel_size`.
    energy : float | torch.Tensor
        Electron beam energy in kiloelectronvolts (e.g. 300 for 300 kV).
    n_slices : int | None
        Number of multislice steps to take. If `None` (default), every
        slice of `potential` is propagated individually (equivalent to
        `n_slices=Z`). If smaller than `Z`, slices are grouped into
        `n_slices` contiguous chunks via `chunk_slices` (all but the last
        chunk have the same size; the last absorbs the remainder), each
        chunk's potential is summed, and propagated as one thicker slab.
        Must satisfy ``0 < n_slices <= Z``.

    Returns
    -------
    torch.Tensor
        Complex-valued 2D exit wave, shape (..., H, W).

    Notes
    -----
    The incident wave entering the specimen is a uniform (unit-amplitude,
    zero-phase) plane wave. Each chunk's summed potential is transmitted
    with `dz=pixel_size` (the per-voxel Riemann-sum weight) and then
    propagated with `dz=chunk_size * pixel_size` (the chunk's physical
    thickness). `n_slices=None` propagates one true slice at a time and
    is the most accurate setting; grouping trades accuracy for speed.

    `torch_grid_utils.fftfreq_grid` only accepts a plain float for its
    `spacing` argument, so if `pixel_size` is a tensor, the frequency grid
    itself is built from a detached float copy; `pixel_size` still flows
    as a tensor into the propagator and transmission function, so gradients
    with respect to it are not lost, just not routed through the grid
    spacing.
    """
    wavelength, sigma, frequency_grid = _prepare_propagation_parameters(
        potential, pixel_size, energy
    )

    height, width = potential.shape[-2:]
    wave = torch.ones(
        (*potential.shape[:-3], height, width),
        dtype=potential.dtype,
        device=potential.device,
    )
    total_slices = potential.shape[-3]
    if n_slices is None:
        n_slices = total_slices
    sizes = chunk_slices(total_slices, n_slices)
    # At most 2 distinct chunk sizes occur; build one propagator per size.
    propagators = {
        size: fresnel_propagator(
            frequency_grid, wavelength=wavelength, dz=pixel_size * size
        )
        for size in set(sizes)
    }
    for chunk in torch.split(potential, sizes, dim=-3):
        chunk_size = chunk.shape[-3]
        potential_chunk = chunk.sum(dim=-3)
        wave = multislice_step(
            wave, potential_chunk, propagators[chunk_size], sigma, pixel_size
        )
    return wave
