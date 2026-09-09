"""Gradient-based local refinement via autograd (axis-angle parameterisation)."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch_grid_utils import coordinate_grid
from torch_image_interpolation import sample_image_3d
from tqdm import tqdm

from ._config import GradientRefinementConfig
from ._preprocess import _normalise_volume
from ._result import AlignmentResult

# ---------------------------------------------------------------------------
# Rodrigues rotation helpers
# ---------------------------------------------------------------------------


def _axis_angle_to_rotation_matrix_xyz(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula.  v: ``(3,)`` axis-angle vector in xyz (radians).

    Numerically stable: when ``|v| → 0`` the result approaches the identity.
    """
    theta = v.norm(p=2).clamp(min=1e-7)
    k = v / theta  # unit axis
    # Skew-symmetric cross-product matrix
    K = torch.stack(
        [
            torch.stack([torch.zeros_like(k[0]), -k[2], k[1]]),
            torch.stack([k[2], torch.zeros_like(k[1]), -k[0]]),
            torch.stack([-k[1], k[0], torch.zeros_like(k[2])]),
        ]
    )  # (3, 3)
    identity = torch.eye(3, dtype=v.dtype, device=v.device)
    return identity + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def _rotation_matrix_xyz_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """Inverse Rodrigues.  R: ``(3, 3)`` xyz rotation → ``(3,)`` axis-angle."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    if theta.abs() < 1e-6:
        return torch.zeros(3, dtype=R.dtype, device=R.device)
    axis = torch.stack([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
        2.0 * torch.sin(theta)
    )
    return axis * theta


def _flip_3x3(M: torch.Tensor) -> torch.Tensor:
    """Convert between xyz and zyx 3 x 3 rotation matrices by flipping."""
    return torch.flip(M, dims=(-2, -1))


# ---------------------------------------------------------------------------
# Differentiable transform
# ---------------------------------------------------------------------------


def _transform_volume(
    volume: torch.Tensor,
    R_xyz: torch.Tensor,
    t_zyx: torch.Tensor,
    centre_zyx: torch.Tensor,
) -> torch.Tensor:
    """Apply rotation + translation to *volume* in a differentiable way.

    Implements: ``output[p] = volume[R_zyx @ (p - t - c) + c]``, i.e. first
    shift the sampling grid by *t*, then rotate around box-centre *c*.  This
    matches the convention used by the exhaustive search and
    :func:`~torch_fit_in_map.apply_alignment`.

    Parameters
    ----------
    volume : torch.Tensor
        ``(d, h, w)`` volume to transform.
    R_xyz : torch.Tensor
        ``(3, 3)`` rotation matrix in xyz convention (differentiable).
    t_zyx : torch.Tensor
        ``(3,)`` translation in zyx pixels (differentiable).
    centre_zyx : torch.Tensor
        ``(3,)`` fixed box-centre in zyx pixels.

    Returns
    -------
    transformed : torch.Tensor
        ``(d, h, w)`` transformed volume.
    """
    R_zyx = _flip_3x3(R_xyz)  # (3, 3) zyx, differentiable

    d, h, w = volume.shape
    with torch.no_grad():
        grid = coordinate_grid(
            image_shape=(d, h, w), device=volume.device
        )  # (d, h, w, 3)

    # Convention: output[p] = volume[R_zyx @ (p - t - c) + c]
    # Matches exhaustive search and apply_alignment: rotate around c, then shift by t.
    # The previous form R_zyx @ (p - c) + c - t differs by (R@t - t) and broke
    # gradient-refined results whenever t != 0 and R != I.
    p_minus_t_minus_c = grid - t_zyx - centre_zyx  # (d, h, w, 3)
    rotated = torch.einsum("ij,...j->...i", R_zyx, p_minus_t_minus_c)  # (d, h, w, 3)
    input_coords = rotated + centre_zyx  # (d, h, w, 3), differentiable

    transformed = sample_image_3d(volume, input_coords, interpolation="trilinear")
    return cast("torch.Tensor", transformed)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def _ncc_loss(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Negative normalised cross-correlation (minimise to maximise NCC)."""
    if mask is not None:
        a = a * mask
        b = b * mask
        n = mask.sum().clamp(min=1.0)
    else:
        n = torch.tensor(float(a.numel()), dtype=a.dtype, device=a.device)

    a_mean = a.sum() / n
    b_mean = b.sum() / n
    ac = a - a_mean
    bc = b - b_mean
    ncc = (ac * bc).sum() / (torch.sqrt((ac**2).sum() * (bc**2).sum() + 1e-8))
    return -ncc


def _mse_loss(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Mean squared error."""
    if mask is not None:
        diff = (a - b) * mask
        return (diff**2).sum() / mask.sum().clamp(min=1.0)
    return torch.nn.functional.mse_loss(a, b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def gradient_refine(
    reference: torch.Tensor,
    mobile: torch.Tensor,
    initial_rotation: torch.Tensor,
    initial_translation: torch.Tensor,
    config: GradientRefinementConfig | None = None,
    mask: torch.Tensor | None = None,
    verbose: bool = True,
) -> AlignmentResult:
    """Gradient-based local refinement of rotation and translation.

    Refines an initial alignment (e.g. from :func:`exhaustive_search`) using
    PyTorch autograd.  The rotation is parameterised as an axis-angle vector
    (Rodrigues formula), which is unconstrained and avoids gimbal lock.

    Parameters
    ----------
    reference : torch.Tensor
        ``(d, h, w)`` reference volume.
    mobile : torch.Tensor
        ``(d, h, w)`` mobile volume at the same voxel size.
    initial_rotation : torch.Tensor
        ``(3, 3)`` zyx rotation matrix from a coarse search.
    initial_translation : torch.Tensor
        ``(3,)`` zyx translation in pixels from a coarse search.
    config : GradientRefinementConfig or None
        Optimisation parameters.
    mask : torch.Tensor or None
        Optional ``(d, h, w)`` soft mask.
    verbose : bool
        Show refinement progress.

    Returns
    -------
    AlignmentResult
        Refined rotation matrix (3 x 3, zyx), translation in pixels (3,), and
        final NCC score.
    """
    if config is None:
        config = GradientRefinementConfig()

    device = reference.device
    d, h, w = reference.shape[-3:]

    ref_norm = _normalise_volume(reference.float(), mask)
    mob_norm = _normalise_volume(mobile.float(), mask)

    centre = torch.tensor(
        [(d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0],
        dtype=torch.float32,
        device=device,
    )

    # Initialise from the coarse-search result
    R_zyx_init = initial_rotation.to(device=device, dtype=torch.float32)
    R_xyz_init = _flip_3x3(R_zyx_init)
    v0 = _rotation_matrix_xyz_to_axis_angle(R_xyz_init)

    v_param = nn.Parameter(v0.clone().to(device=device, dtype=torch.float32))
    t_param = nn.Parameter(
        initial_translation.clone().to(device=device, dtype=torch.float32)
    )

    optimizer: torch.optim.Optimizer
    if config.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [v_param, t_param],
            lr=config.learning_rate,
            max_iter=config.n_iterations,
            line_search_fn="strong_wolfe",
        )
    else:
        # Adam's usual learning rate is 1e-2; the LBFGS default may be too high.
        lr = config.learning_rate
        if config.learning_rate == 1.0 and config.optimizer == "adam":
            lr = 1e-2
        optimizer = torch.optim.Adam([v_param, t_param], lr=lr)

    loss_fn = _ncc_loss if config.loss == "ncc" else _mse_loss

    pbar = tqdm(
        total=config.n_iterations,
        desc=f"Gradient refinement ({config.optimizer.upper()})",
        unit="iter",
        dynamic_ncols=True,
        disable=not verbose,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        R_xyz = _axis_angle_to_rotation_matrix_xyz(v_param)
        transformed = _transform_volume(mob_norm, R_xyz, t_param, centre)
        loss = loss_fn(ref_norm, transformed, mask)
        if loss.requires_grad:
            loss.backward()  # type: ignore[no-untyped-call]
        # L-BFGS may call closure multiple times per iteration (line search)
        # but we use it as a heartbeat
        if config.optimizer == "adam":
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        else:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return loss

    if config.optimizer == "lbfgs":
        # L-BFGS runs all iterations inside one .step(closure)
        optimizer.step(closure)  # type: ignore[no-untyped-call]
    else:
        # Adam runs explicitly in a loop
        for _ in range(config.n_iterations):
            optimizer.step(closure)  # type: ignore[no-untyped-call]
            pbar.update(1)

    pbar.close()

    # Final evaluation
    with torch.no_grad():
        R_xyz_final = _axis_angle_to_rotation_matrix_xyz(v_param)
        transformed_final = _transform_volume(mob_norm, R_xyz_final, t_param, centre)
        final_loss = float(loss_fn(ref_norm, transformed_final, mask).item())
        final_score = -final_loss

        R_zyx_final = _flip_3x3(R_xyz_final)
        t_final = t_param.data.clone()

    t_angstroms: torch.Tensor | None = None
    if config.pixel_size_angstroms is not None:
        t_angstroms = t_final * config.pixel_size_angstroms

    return AlignmentResult(
        rotation_matrix=R_zyx_final,
        translation_pixels=t_final,
        score=final_score,
        translation_angstroms=t_angstroms,
    )
