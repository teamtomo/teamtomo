"""Projection-based alignment using Fourier-slice projections."""

from __future__ import annotations

import torch
from torch_fourier_slice import project_3d_to_2d
from torch_so3 import get_uniform_euler_angles

from ._config import ProjectionAlignmentConfig
from ._exhaustive import _euler_zyz_to_4x4_zyx
from ._preprocess import _normalise_volume
from ._result import AlignmentResult


def _find_2d_peak_shift(
    cc_map: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Return the centred shift and score from a 2-D cross-correlation map."""
    h, w = cc_map.shape[-2:]
    flat_idx = cc_map.argmax()
    score = float(cc_map.view(-1)[flat_idx].item())
    iy, ix = divmod(int(flat_idx.item()), w)
    dy = float(iy if iy <= h // 2 else iy - h)
    dx = float(ix if ix <= w // 2 else ix - w)
    return torch.tensor([dy, dx], dtype=torch.float32, device=cc_map.device), score


def projection_align(
    reference: torch.Tensor,
    mobile: torch.Tensor,
    config: ProjectionAlignmentConfig | None = None,
    mask: torch.Tensor | None = None,
) -> AlignmentResult:
    """Projection-based alignment via Fourier-slice 2-D projections.

    For each candidate orientation of the mobile, a 2-D projection is
    computed using :func:`torch_fourier_slice.project_3d_to_2d` and compared
    against the reference projection at the same viewing direction via 2-D
    normalised cross-correlation.  The orientation with the highest 2-D NCC
    score provides the rotation, and the CC-map peak position provides the
    in-plane translation.

    .. note::
        Only the in-plane translation components (y and x) are recovered.
        The depth shift (along the projection axis z) is set to zero and
        should be refined with :func:`gradient_refine`.

    Parameters
    ----------
    reference : torch.Tensor
        ``(d, h, w)`` reference volume.  Must be cubic (``d == h == w``).
    mobile : torch.Tensor
        ``(d, h, w)`` mobile volume at the same voxel size.
    config : ProjectionAlignmentConfig or None
        Alignment parameters.
    mask : torch.Tensor or None
        Optional ``(d, h, w)`` soft mask applied before projection.

    Returns
    -------
    AlignmentResult
        Best rotation matrix (3x3, zyx), translation ``[0, dy, dx]`` in
        pixels, and peak 2-D NCC score.
    """
    if config is None:
        config = ProjectionAlignmentConfig()

    device = reference.device
    ref_float = _normalise_volume(reference.float(), mask)
    mob_float = _normalise_volume(mobile.float(), mask)

    if mask is not None:
        ref_float = ref_float * mask
        mob_float = mob_float * mask

    # Sample orientations (N, 3)
    euler_angles = get_uniform_euler_angles(
        psi_step=config.angular_step_degrees,
        theta_step=config.angular_step_degrees,
    ).to(device)
    # Build xyz (3, 3) rotation matrices for project_3d_to_2d (xyz convention)
    # _euler_zyz_to_4x4_zyx → (N, 4, 4) zyx; extract upper-left, flip to xyz
    R_4x4_zyx = _euler_zyz_to_4x4_zyx(euler_angles)  # (N, 4, 4)
    R3_zyx = R_4x4_zyx[:, :3, :3]  # (N, 3, 3)
    R3_xyz = torch.flip(R3_zyx, dims=(-2, -1))  # (N, 3, 3) xyz

    n = R3_xyz.shape[0]

    # project_3d_to_2d: zyx_matrices=False → xyz convention input
    # Batch-project reference and mobile (N, d, d) each
    ref_projs = project_3d_to_2d(
        ref_float, R3_xyz, fftfreq_max=config.fftfreq_max
    )  # (N, d, d)
    mob_projs = project_3d_to_2d(
        mob_float, R3_xyz, fftfreq_max=config.fftfreq_max
    )  # (N, d, d)

    # 2-D NCC map for each orientation
    ref_rfft = torch.fft.rfft2(ref_projs, dim=(-2, -1))  # (N, d, d//2+1)
    mob_rfft = torch.fft.rfft2(mob_projs, dim=(-2, -1))
    cc_2d = torch.fft.irfft2(ref_rfft * torch.conj(mob_rfft), dim=(-2, -1))  # (N, d, d)

    # Best orientation: argmax of per-orientation peak score
    peak_per_orient = cc_2d.view(n, -1).max(dim=-1).values  # (N,)
    best_idx = int(peak_per_orient.argmax().item())

    best_R3_xyz = R3_xyz[best_idx]  # (3, 3) xyz
    best_R3_zyx = torch.flip(best_R3_xyz, dims=(-2, -1))  # (3, 3) zyx
    t_2d, score = _find_2d_peak_shift(cc_2d[best_idx])
    translation_zyx = torch.zeros(3, dtype=torch.float32, device=device)
    translation_zyx[1] = t_2d[0]
    translation_zyx[2] = t_2d[1]

    t_angstroms: torch.Tensor | None = None
    if config.pixel_size_angstroms is not None:
        t_angstroms = translation_zyx * config.pixel_size_angstroms

    return AlignmentResult(
        rotation_matrix=best_R3_zyx,
        translation_pixels=translation_zyx,
        score=score,
        translation_angstroms=t_angstroms,
    )
