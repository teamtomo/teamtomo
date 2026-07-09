"""Input validation and host-buffer preparation shared by the projectors.

Naming follows the kernels: ``bv`` is the batch of volumes, ``bp`` the batch of
projections; a volume's spatial axes are ``(d, h, w)`` with ``sidelength`` the
cube edge (``= h``) and ``sidelength_half = w`` the rfft width.
"""

from __future__ import annotations

import torch

_INTERP_CODES = {"linear": 0, "cubic": 1}


def interp_code(interpolation: str) -> int:
    """Map an interpolation name to the Mojo kernel's integer code."""
    if interpolation not in _INTERP_CODES:
        raise ValueError(
            f"interpolation must be one of {sorted(_INTERP_CODES)}, got "
            f"{interpolation!r}"
        )
    return _INTERP_CODES[interpolation]


def validate_reconstruction(
    reconstruction: torch.Tensor,
) -> tuple[torch.Tensor, int, int, int]:
    """Check/normalise rfft volume; return (4d, bv, sidelength, sidelength_half).

    Expects an rfft volume with DC at the origin (unshifted), shape
    ``(d, h, w)`` or ``(bv, d, h, w)``, cubic and even.
    """
    if not reconstruction.is_complex():
        raise ValueError("reconstruction must be a complex tensor")
    if reconstruction.dim() == 3:
        reconstruction = reconstruction.unsqueeze(0)
    if reconstruction.dim() != 4:
        raise ValueError("reconstruction must be (d, h, w) or (bv, d, h, w)")
    bv, _d, sidelength, sidelength_half = reconstruction.shape
    if sidelength % 2 != 0:
        raise ValueError(f"side length ({sidelength}) must be even")
    if sidelength != (sidelength_half - 1) * 2:
        raise ValueError(
            f"shape mismatch: h={sidelength} must equal 2*(w - 1)="
            f"{(sidelength_half - 1) * 2}"
        )
    return reconstruction, bv, sidelength, sidelength_half


def prep_rotations(
    rotations: torch.Tensor, bv: int, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, int, int]:
    """Normalise rotations to a contiguous float32 ``(bv_rot, bp, 3, 3)`` on ``device``.

    ``device`` is the CPU for the CPU kernel and the compute GPU for the
    zero-copy GPU kernel (which reads this buffer's device memory directly).
    """
    if rotations.dim() == 2:
        rotations = rotations.unsqueeze(0)
    if rotations.dim() == 3:
        rotations = rotations.unsqueeze(0)
    if rotations.dim() != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must broadcast to (bv_rot, bp, 3, 3)")
    bv_rot = rotations.shape[0]
    bp = rotations.shape[1]
    if bv_rot not in (1, bv):
        raise ValueError("rotations batch must be 1 or match batch dimension")
    rot = rotations.to(device=device, dtype=torch.float32).contiguous()
    return rot, bv_rot, bp


def prep_shifts_2d(
    shifts_2d: torch.Tensor | None,
    bv: int,
    bp: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, int]:
    """Normalise 2D shifts to float32 ``(bv_shift_2d, bp, 2)`` on ``device``; + flag."""
    if shifts_2d is None:
        return torch.zeros(1, bp, 2, dtype=torch.float32, device=device), 0
    shifts_2d_t = shifts_2d
    if shifts_2d_t.dim() == 2:
        shifts_2d_t = shifts_2d_t.unsqueeze(0)
    if shifts_2d_t.dim() != 3 or shifts_2d_t.shape[-1] != 2:
        raise ValueError("shifts_2d must broadcast to (bv_shift_2d, bp, 2)")
    if shifts_2d_t.shape[0] not in (1, bv):
        raise ValueError("shifts_2d batch must be 1 or match batch dimension")
    if shifts_2d_t.shape[1] != bp:
        raise ValueError("shifts_2d pose count must match rotations")
    return shifts_2d_t.to(device=device, dtype=torch.float32).contiguous(), 1


def prep_shifts_3d(
    shifts_3d: torch.Tensor | None,
    bv: int,
    bp: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, int]:
    """Normalise 3D shifts to float32 ``(bv_shift_3d, bp, 3)`` zyx on ``device``; + flag."""
    if shifts_3d is None:
        return torch.zeros(1, bp, 3, dtype=torch.float32, device=device), 0
    s = shifts_3d
    if s.dim() == 2:
        s = s.unsqueeze(0)
    if s.dim() != 3 or s.shape[-1] != 3:
        raise ValueError("shifts_3d must broadcast to (bv_shift_3d, bp, 3)")
    if s.shape[0] not in (1, bv):
        raise ValueError("shifts_3d batch must be 1 or match batch dimension")
    if s.shape[1] != bp:
        raise ValueError("shifts_3d pose count must match rotations")
    return s.to(device=device, dtype=torch.float32).contiguous(), 1


def prep_poses(
    bv: int,
    sidelength: int,
    rotations: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    output_shape: tuple[int, int] | None,
    oversampling: float,
    fourier_radius_cutoff: float | None,
    interpolation: str,
    ewald_curvature: float = 0.0,
    shifts_3d: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    """Validate poses and build (rot, shifts, shifts_3d, output, params) buffers.

    All buffers land on ``device``: the CPU for the CPU kernel, or the compute
    GPU for the zero-copy GPU kernel. ``proj_r`` is the zeroed output the kernel
    writes into (radius-cut pixels stay 0, so pre-zeroing is load-bearing).
    """
    rot, _bv_rot, bp = prep_rotations(rotations, bv, device)
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, device)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, device)

    if output_shape is None:
        proj_sidelength = sidelength
    else:
        if len(output_shape) != 2 or output_shape[0] != output_shape[1]:
            raise ValueError(f"output_shape {output_shape} must be square")
        if output_shape[0] % 2 != 0:
            raise ValueError(f"output side length {output_shape[0]} must be even")
        proj_sidelength = int(output_shape[0])
    proj_sidelength_half = proj_sidelength // 2 + 1

    radius = (
        proj_sidelength / 2.0
        if fourier_radius_cutoff is None
        else float(fourier_radius_cutoff)
    )
    proj_r = torch.zeros(
        bv, bp, proj_sidelength, proj_sidelength_half, 2, dtype=torch.float32, device=device
    )
    params = (
        float(oversampling),
        float(radius * radius),
        int(has_shifts_2d),
        interp_code(interpolation),
        float(ewald_curvature),
        int(has_shifts_3d),
    )
    return rot, shifts_2d_t, shifts_3d_t, proj_r, params
