"""High-level projection-approximation wrapper.

Wires up the interaction parameter and slice summation on top of the
pure-math primitives in `_core`.
"""

import torch

from torch_scattering._core import (
    _validate_propagation_inputs,
    interaction_parameter,
    transmission_function,
)


def projection(
    potential: torch.Tensor,
    pixel_size: float | torch.Tensor,
    voltage: float | torch.Tensor,
) -> torch.Tensor:
    """
    Compute the exit wave using the projection approximation.

    Parameters
    ----------
    potential : torch.Tensor
        Real-valued electrostatic potential in volts for a non-absorbing
        specimen, or complex-valued potential when modelling absorption.
        Shape (..., Z, H, W), where Z is the beam direction.
    pixel_size : float | torch.Tensor
        Positive finite isotropic voxel spacing in Angstroms. This is both
        the in-plane pixel spacing and slice thickness.
    voltage : float | torch.Tensor
        Electron beam acceleration voltage in kilovolts (e.g. 300 for 300 kV).

    Returns
    -------
    torch.Tensor
        Complex-valued 2D exit wave, shape (..., H, W).

    Notes
    -----
    This wave-propagation approximation numerically sums a sampled 3D
    potential along the beam direction, then treats the specimen as
    infinitely thin and ignores Fresnel propagation between slices:
    ``psi = exp(i * sigma * dz * sum_z V(z))``. This is valid only when the
    specimen is thin enough that propagation effects within it are
    negligible. It is distinct from an analytic projected-potential
    calculation and from projection alignment used when fitting structures
    into maps.
    """
    _validate_propagation_inputs(potential, pixel_size, voltage)
    sigma = interaction_parameter(voltage=voltage)
    projected_potential = potential.sum(dim=-3)
    return transmission_function(projected_potential, sigma=sigma, dz=pixel_size)
