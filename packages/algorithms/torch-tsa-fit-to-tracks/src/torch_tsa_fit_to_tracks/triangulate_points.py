"""Least-squares 3D positions of points from their tracked 2D positions."""

import torch
from torch_tilt_series import TiltSeries


def affine_projection(tilt_series: TiltSeries) -> tuple[torch.Tensor, torch.Tensor]:
    """Affine part of the projection: `project(p) = matrices @ p + offsets`.

    Evaluated by projecting the origin and three unit vectors, so it follows
    whatever geometry `tilt_series` holds (tomogram -> sample transform,
    x-axis tilt, per-tilt rotation and shift). Non-affine local shift models
    (`local_shifts`, `local_shifts_2d`) are included only through their value
    at these four points.

    Parameters
    ----------
    tilt_series : TiltSeries
        Projection geometry.

    Returns
    -------
    matrices : torch.Tensor
        `(n_tilts, 2, 3)` maps from tomogram zyx (Angstroms) to detector yx
        (Angstroms).
    offsets : torch.Tensor
        `(n_tilts, 2)` detector yx (Angstroms) of the tomogram origin.
    """
    scale = 1000.0  # Angstroms; the map is affine, so any scale is exact
    points = torch.cat(
        [
            torch.zeros(1, 3, device=tilt_series.device),
            scale * torch.eye(3, device=tilt_series.device),
        ]
    )
    projected = tilt_series.project_points(points)  # (4, n_tilts, 2)
    offsets = projected[0]
    matrices = (projected[1:] - offsets) / scale  # (3, n_tilts, 2)
    return matrices.permute(1, 2, 0), offsets


def triangulate_points(
    tilt_series: TiltSeries,
    tracks_yx: torch.Tensor,
    observed: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D positions of points that best explain their tracked 2D positions.

    Solves, for each point independently, the weighted linear least-squares
    problem `min_p sum_i w_i |A_i p + c_i - y_i|^2` over the tilts `i` where
    the point was observed, with `A_i`, `c_i` from `affine_projection`.

    Parameters
    ----------
    tilt_series : TiltSeries
        Projection geometry.
    tracks_yx : torch.Tensor
        `(n_points, n_tilts, 2)` tracked detector positions, yx, in Angstroms
        relative to the detector center (the convention of
        `TiltSeries.project_points`).
    observed : torch.Tensor
        `(n_points, n_tilts)` boolean, True where the point was tracked.
    weights : torch.Tensor | None
        Optional `(n_points, n_tilts)` weights; defaults to 1 where observed.

    Returns
    -------
    points_zyx : torch.Tensor
        `(n_points, 3)` zyx positions in Angstroms, relative to the tomogram
        center. Points seen in fewer than two tilts are underdetermined; a
        small ridge term keeps the solve finite and their position along the
        undetermined direction stays near 0.
    """
    matrices, offsets = affine_projection(tilt_series)
    w = observed.float() if weights is None else weights * observed.float()
    tracks = torch.where(observed[..., None], tracks_yx, 0.0)
    rhs = tracks - offsets  # (n_points, n_tilts, 2)
    ata = torch.einsum("pt,tki,tkj->pij", w, matrices, matrices)  # (p, 3, 3)
    atb = torch.einsum("pt,tki,ptk->pi", w, matrices, rhs)  # (p, 3)
    ridge = 1e-6 * ata.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-6)
    ata = ata + ridge[:, None, None] * torch.eye(3, device=ata.device)
    return torch.linalg.solve(ata, atb)
