"""High-level projection-approximation wrapper.

Wires up the interaction parameter and slice summation on top of the
pure-math primitives in `_core`.
"""

import torch

from torch_scattering._core import interaction_parameter, transmission_function


def projection(
    potential: torch.Tensor,
    pixel_size: float | torch.Tensor,
    energy: float | torch.Tensor,
) -> torch.Tensor:
    """
    Compute the 2D exit wave from a 3D potential via the projection approximation.

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

    Returns
    -------
    torch.Tensor
        Complex-valued 2D exit wave, shape (..., H, W).

    Notes
    -----
    The projection approximation treats the specimen as infinitely thin,
    ignoring Fresnel propagation between slices: the potential is summed
    along the beam direction and transmitted through in a single step,
    ``psi = exp(i * sigma * dz * sum_z V(z))``. This is valid only when the
    specimen is thin enough that propagation effects within it are
    negligible.
    """
    sigma = interaction_parameter(energy=energy)
    projected_potential = potential.sum(dim=-3)
    return transmission_function(projected_potential, sigma=sigma, dz=pixel_size)
