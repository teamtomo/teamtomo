"""Projection-based alignment using Fourier-slice projections."""

from __future__ import annotations

import torch
from torch_fourier_slice import project_3d_to_2d
from torch_so3 import get_uniform_euler_angles

from ._config import ProjectionAlignmentConfig
from ._exhaustive import _euler_zyz_to_4x4_zyx
from ._preprocess import _normalise_volume
from ._result import AlignmentResult

# Three orthogonal views of the reference as ZYZ Euler angles (degrees): looking
# along z, x and y, each tilted by the same generic rotation.  Views along a box
# axis are avoided because rotation matrices with zero entries put whole lines
# of the slice on the kx = 0 plane of the rfft, which torch-fourier-slice
# samples incorrectly.
_AXIS_VIEW_EULER_ANGLES = torch.tensor(
    [[0.0, 0.0, 0.0], [0.0, 90.0, 0.0], [0.0, 90.0, 90.0]], dtype=torch.float64
)
_VIEW_TILT_EULER_ANGLES = torch.tensor([[20.0, 50.0, 10.0]], dtype=torch.float64)


def _view_matrices() -> torch.Tensor:
    """Return the ``(3, 3, 3)`` zyx active rotations of the fixed views."""
    axes = _euler_zyz_to_4x4_zyx(_AXIS_VIEW_EULER_ANGLES)[:, :3, :3]
    tilt = _euler_zyz_to_4x4_zyx(_VIEW_TILT_EULER_ANGLES)[:, :3, :3]
    return (axes @ tilt).float()


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


def _normalise_projections(projections: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-norm images, so their cross-correlation is an NCC."""
    centred = projections - projections.mean(dim=(-2, -1), keepdim=True)
    norm = centred.square().sum(dim=(-2, -1), keepdim=True).sqrt()
    return centred / norm.clamp(min=1e-8)


def projection_align(
    reference: torch.Tensor,
    mobile: torch.Tensor,
    config: ProjectionAlignmentConfig | None = None,
    mask: torch.Tensor | None = None,
) -> AlignmentResult:
    """Projection-based alignment via Fourier-slice 2-D projections.

    The reference is projected along three fixed, mutually orthogonal views.
    For each candidate rotation ``R`` the mobile is rotated back by ``R`` and
    projected along the same three views with
    :func:`torch_fourier_slice.project_3d_to_2d`, then compared with the
    matching reference projection via 2-D normalised cross-correlation (NCC).
    The candidate with the highest NCC, averaged over the views, gives the
    rotation.  The CC-map peaks of the three views each give two components
    of the 3-D shift, which are combined by least squares into the
    translation.

    .. note::
        The translation comes from whole-pixel 2-D CC peaks, so it is accurate
        to about a pixel and should be refined with :func:`gradient_refine`.

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
        ``rotation_matrix`` is the ``(3, 3)`` zyx matrix that brings the mobile
        onto the reference, with the same meaning as for
        :func:`exhaustive_search` and :func:`gradient_refine`: if *mobile* is
        *reference* actively rotated by ``Q`` about the box centre, it
        approximates ``Q``, and :func:`apply_alignment` rotates the mobile
        back.  ``translation_pixels`` is the ``(3,)`` zyx shift in the same
        convention, and ``score`` is the peak 2-D NCC averaged over the views.
    """
    if config is None:
        config = ProjectionAlignmentConfig()

    device = reference.device
    d = reference.shape[-1]
    ref_float = _normalise_volume(reference.float(), mask)
    mob_float = _normalise_volume(mobile.float(), mask)

    if mask is not None:
        ref_float = ref_float * mask
        mob_float = mob_float * mask

    # Candidate rotations (N, 3, 3) zyx, read as AlignmentResult.rotation_matrix:
    # the aligned mobile is the mobile actively rotated by R^T.
    euler_angles = get_uniform_euler_angles(
        psi_step=config.angular_step_degrees,
        theta_step=config.angular_step_degrees,
    )
    candidates = _euler_zyz_to_4x4_zyx(euler_angles)[:, :3, :3].float().to(device)
    views = _view_matrices().to(device)

    # project_3d_to_2d takes active rotations; views S_j: (V, d, d)
    ref_projs = project_3d_to_2d(
        ref_float, views, fftfreq_max=config.fftfreq_max, zyx_matrices=True
    )
    ref_rfft = torch.fft.rfft2(_normalise_projections(ref_projs), dim=(-2, -1))

    best_score = float("-inf")
    best_R = candidates[0]
    best_cc = torch.zeros(views.shape[0], d, d, device=device)
    for start in range(0, candidates.shape[0], config.rotation_batch_size):
        R = candidates[start : start + config.rotation_batch_size]  # (B, 3, 3)
        # Viewing the aligned mobile from S_j means projecting the mobile
        # actively rotated by S_j @ R^T: (B, V, 3, 3) -> (B, V, d, d)
        mob_projs = project_3d_to_2d(
            mob_float,
            views @ R.transpose(-1, -2).unsqueeze(1),
            fftfreq_max=config.fftfreq_max,
            zyx_matrices=True,
        )
        mob_rfft = torch.fft.rfft2(_normalise_projections(mob_projs), dim=(-2, -1))
        cc_2d = torch.fft.irfft2(
            ref_rfft * torch.conj(mob_rfft), s=(d, d), dim=(-2, -1)
        )  # (B, V, d, d)

        # Score each candidate by its per-view NCC peak, averaged over views
        scores = cc_2d.flatten(start_dim=-2).max(dim=-1).values.mean(dim=-1)  # (B,)
        i = int(scores.argmax().item())
        if scores[i].item() > best_score:
            best_score = float(scores[i].item())
            best_R = R[i]
            best_cc = cc_2d[i]

    # View S_j sees the 3-D shift t as the yx part of S_j @ t: (2V, 3) @ t = (2V,)
    shifts_2d = torch.stack([_find_2d_peak_shift(cc)[0] for cc in best_cc])
    translation_zyx = torch.linalg.lstsq(
        views[:, 1:, :].reshape(-1, 3), shifts_2d.reshape(-1, 1)
    ).solution.squeeze(-1)
    # Fourier-slice projection rotates about voxel d // 2, AlignmentResult
    # about (d - 1) / 2; for even boxes this offsets the shift by
    # (I - R^T)(c_fft - c).
    c_offset = torch.full((3,), d // 2 - (d - 1) / 2.0, device=device)
    eye = torch.eye(3, device=device)
    translation_zyx = translation_zyx + (eye - best_R.T) @ c_offset

    t_angstroms: torch.Tensor | None = None
    if config.pixel_size_angstroms is not None:
        t_angstroms = translation_zyx * config.pixel_size_angstroms

    return AlignmentResult(
        rotation_matrix=best_R,
        translation_pixels=translation_zyx,
        score=best_score,
        translation_angstroms=t_angstroms,
    )
