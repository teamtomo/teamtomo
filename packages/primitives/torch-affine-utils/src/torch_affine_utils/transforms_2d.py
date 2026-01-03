"""3x3 matrices for rotations and translations in 2D.

Functions in this module generate matrices which left-multiply column vectors containing
`xyw` or `yxw` homogenous coordinates.
"""

import torch
import einops


def R(angles: torch.Tensor | list | tuple | float, 
      yx: bool = False, 
      device: torch.device | None = None) -> torch.Tensor:
    """3x3 matrices for a rotation of homogenous coordinates in 2D.

    Matrix structure (xyw):
    ┌          ┐
    │ c -s  0  │
    │ s  c  0  │
    │ 0  0  1  │
    └          ┘

    Matrix structure (yxw):
    ┌          ┐
    │ c  s  0  │
    │-s  c  0  │
    │ 0  0  1  │
    └          ┘
    where c=cos(θ), s=sin(θ)

    Parameters
    ----------
    angles: torch.Tensor | list | tuple | float
        `(..., )` array, list-like, or single float of angles in degrees.
    yx: bool
        Whether output should be compatible with `yxw` (`True`) or `xyw`
        (`False`) homogenous coordinates.
    device: torch.device, optional
        The device on which to place the resulting tensor. If None, uses the
        device of the input tensor or defaults to CPU.

    Returns
    -------
    matrices: `(..., 3, 3)` array of 3x3 rotation matrices.
    """
    angles = torch.as_tensor(angles, dtype=torch.float32)
    device = device or angles.device  # Use provided device or input tensor's device
    angles = angles.to(device)
    angles_packed, ps = einops.pack([angles], pattern='*')  # to 1d
    n = angles_packed.shape[0]

    # calculate useful values
    angles_radians = torch.deg2rad(angles_packed)
    c = torch.cos(angles_radians)
    s = torch.sin(angles_radians)

    # construct matrices
    matrices = einops.repeat(torch.eye(3, device=device), 'i j -> n i j', n=n).clone()
    matrices[:, 0, 0] = c
    matrices[:, 0, 1] = -s
    matrices[:, 1, 0] = s
    matrices[:, 1, 1] = c

    # operating on yx coordinates?
    if yx is True:
        matrices[:, :2, :2] = torch.flip(matrices[:, :2, :2], dims=(-2, -1))

    # shape (n, 3, 3) -> (..., 3, 3)
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices


def T(shifts: torch.Tensor | list | tuple, 
      device: torch.device = None) -> torch.Tensor:
    """3x3 matrices for translations in 2D.

    Matrix structure:
    ┌           ┐
    │ 1  0  t0  │
    │ 0  1  t1  │
    │ 0  0  1   │
    └           ┘
    where t0, t1 are translation components at indices 0, 1 in the last
    dimension of `shifts`.

    Parameters
    ----------
    shifts: torch.Tensor | list | tuple
        `(..., 2)` array or list-like of shifts.
    device: torch.device, optional
        The device on which to place the resulting tensor. If None, uses the
        device of the input tensor or defaults to CPU.

    Returns
    -------
    matrices: torch.Tensor
        `(..., 3, 3)` array of 3x3 shift matrices.
    """
    shifts = torch.as_tensor(shifts, dtype=torch.float32)
    if shifts.ndim > 0 and shifts.shape[-1] != 2:
        raise ValueError("Shifts must have the last dimension of size 2 for 2D transformations.")
    shifts = torch.atleast_1d(shifts)
    device = device or shifts.device  # Use provided device or input tensor's device
    shifts = shifts.to(device)
    shifts, ps = einops.pack([shifts], pattern='* coords')  # to 2d
    n = shifts.shape[0]

    # construct matrices
    matrices = einops.repeat(torch.eye(3, device=device), 'i j -> n i j', n=n).clone()
    matrices[:, :2, 2] = shifts

    # shape (n, 3, 3) -> (..., 3, 3)
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices


def S(scale_factors: torch.Tensor | list | tuple | float, 
      device: torch.device | None = None) -> torch.Tensor:
    """3x3 matrices for scaling in 2D.

     Matrix structure:
    ┌          ┐
    │ s1 0  0  │
    │ 0  s2 0  │
    │ 0  0  1  │
    └          ┘
    where s1, s2 are the scaling factors at indices 0, 1 in the last dimension
    of `scale_factors`.

    Parameters
    ----------
    scale_factors: torch.Tensor | list | tuple | float
        `(..., 2)` array, list-like, or single float of scale factors.
    device: torch.device, optional
        The device on which to place the resulting tensor. If None, uses the
        device of the input tensor or defaults to CPU.

    Returns
    -------
    matrices: torch.Tensor
        `(..., 3, 3)` array of 3x3 scaling matrices.
    """
    scale_factors = torch.as_tensor(scale_factors, dtype=torch.float32)
    if scale_factors.ndim > 0 and scale_factors.shape[-1] != 2:
        raise ValueError("Scale factors must have the last dimension of size 2 for 2D transformations.")
    scale_factors = torch.atleast_1d(scale_factors)
    device = device or scale_factors.device  # Use provided device or input tensor's device
    scale_factors = scale_factors.to(device)
    scale_factors, ps = einops.pack([scale_factors], pattern='* coords')  # to 2d
    n = scale_factors.shape[0]

    # construct matrices
    matrices = einops.repeat(torch.eye(3, device=device), 'i j -> n i j', n=n).clone()
    matrices[:, [0, 1], [0, 1]] = scale_factors

    # shape (n, 3, 3) -> (..., 3, 3)
    [matrices] = einops.unpack(matrices, packed_shapes=ps, pattern='* i j')
    return matrices
