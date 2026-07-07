"""High-level first Born approximation wrapper.

Wires up frequency grids, wavelength, and a per-slice Fresnel propagator
stack on top of the pure-math primitives in `_core`.
"""

import torch

from torch_scattering._core import (
    _prepare_propagation_parameters,
    chunk_slices,
    fresnel_propagator,
)


def firstborn(
    potential: torch.Tensor,
    pixel_size: float | torch.Tensor,
    energy: float | torch.Tensor,
    n_slices: int | None = None,
) -> torch.Tensor:
    """
    Compute the 2D exit wave from a 3D potential via the first Born approximation.

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
        Number of terms to sum in the Born series. If `None` (default),
        every slice of `potential` is treated individually (equivalent to
        `n_slices=Z`). If smaller than `Z`, slices are grouped into
        `n_slices` contiguous chunks via `chunk_slices` (all but the last
        chunk have the same size; the last absorbs the remainder), and each
        chunk's summed potential is propagated as one thicker slab. Must
        satisfy ``0 < n_slices <= Z``.

    Returns
    -------
    torch.Tensor
        Complex-valued 2D exit wave, shape (..., H, W).

    Notes
    -----
    Each chunk's scattered contribution is propagated directly to the exit
    plane and summed, rather than being propagated incrementally
    slice-by-slice as in `multislice`:
    ``psi = 1 + sum_c IFFT(FFT(i * sigma * dz * V_c) * H_c)``, where `V_c`
    is the summed potential of chunk `c` and `H_c` is the Fresnel
    propagator for the distance from chunk `c` to the exit plane. This is
    the linearised (first-order) term of the exponentiated Born series
    used by `rytov`, valid only for weak potentials (``sigma * dz * V <<
    1``). Faster than `multislice` (one FFT pass instead of one per chunk)
    but less accurate for thick or strongly scattering specimens, and less
    accurate than `rytov` for the same potential since it does not
    exponentiate the scattered sum.

    References
    ----------
    .. [1] E. J. Kirkland, Advanced Computing in Electron Microscopy,
       Springer US, Boston, MA, 2010.
    """
    wavelength, sigma, frequency_grid = _prepare_propagation_parameters(
        potential, pixel_size, energy
    )

    total_slices = potential.shape[-3]
    if n_slices is None:
        n_slices = total_slices
    sizes = chunk_slices(total_slices, n_slices)
    chunk_potential = torch.stack(
        [chunk.sum(dim=-3) for chunk in torch.split(potential, sizes, dim=-3)],
        dim=-3,
    )  # (..., n_slices, H, W)

    # Chunk c is (sum of sizes[c:]) slice thicknesses from the exit plane:
    # the wave still has to cross the remainder of chunk c itself plus
    # every chunk after it.
    size_tensor = torch.as_tensor(sizes, device=potential.device)
    remaining_slices = size_tensor.flip(0).cumsum(0).flip(0)
    distances = remaining_slices * pixel_size
    propagators = fresnel_propagator(
        frequency_grid, wavelength=wavelength, dz=distances[:, None, None]
    )  # (n_slices, H, W)

    scattered = torch.fft.ifft2(
        torch.fft.fft2(1j * sigma * pixel_size * chunk_potential) * propagators
    )
    exit_wave: torch.Tensor = 1 + scattered.sum(dim=-3)
    return exit_wave
