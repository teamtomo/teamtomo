"""Utilities for deformation field operations."""

import einops
import torch
import torch.nn.functional as F
from torch_cubic_spline_grids import CubicBSplineGrid3d, CubicCatmullRomGrid3d


def evaluate_deformation_field(
    deformation_field: torch.Tensor,  # (yx, nt, nh, nw)
    tyx: torch.Tensor,  # (..., 3)
    grid_type: str = "catmull_rom",
) -> torch.Tensor:  # (
    """Evaluate shifts from deformation field data.

    Parameters
    ----------
    deformation_field: torch.Tensor
        (yx, nt, nh, nw) deformation field data
    tyx: torch.Tensor
        (..., 3) coordinate grid
    grid_type: str
        Type of grid to use ('catmull_rom' or 'bspline'). Default is 'catmull_rom'.

    Returns
    -------
    predicted_shifts: torch.Tensor
        (..., 2) predicted shifts
    """
    if grid_type == "catmull_rom":
        deformation_field = CubicCatmullRomGrid3d.from_grid_data(deformation_field).to(
            deformation_field.device
        )
    elif grid_type == "bspline":
        deformation_field = CubicBSplineGrid3d.from_grid_data(deformation_field).to(
            deformation_field.device
        )
    predicted_shifts = deformation_field(tyx)
    return predicted_shifts


def evaluate_deformation_field_at_t(
    deformation_field: torch.Tensor | CubicCatmullRomGrid3d | CubicBSplineGrid3d,
    t: float,  # [0, 1]
    grid_shape: tuple[int, int],  # (h, w)
    grid_type: str = "catmull_rom",
) -> torch.Tensor:
    """Evaluate a grid of shifts at a specific timepoint from deformation field data.

    Parameters
    ----------
    deformation_field: torch.Tensor | CubicCatmullRomGrid3d | CubicBSplineGrid3d
        (yx, nt, nh, nw) deformation field data
    t: float
        Timepoint to evaluate at [0, 1]
    grid_shape: tuple[int, int]
        (h, w) shape of the grid to evaluate at
    grid_type: str
        Type of grid to use ('catmull_rom' or 'bspline'). Default is 'catmull_rom'.

    Returns
    -------
    shifts: torch.Tensor
        (tyx, h, w) predicted shifts
    """
    if isinstance(deformation_field, CubicCatmullRomGrid3d) or isinstance(
        deformation_field, CubicBSplineGrid3d
    ):
        device = deformation_field.data.device
    else:
        device = deformation_field.device
    h, w = grid_shape
    y = torch.linspace(0, 1, steps=h, device=device)
    x = torch.linspace(0, 1, steps=w, device=device)

    # Create meshgrid and flatten to get (h*w, 2) array of yx coordinates
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yx_grid = einops.rearrange([yy, xx], "yx h w -> (h w) yx")

    # Add t coordinate using F.pad to create tyx triplets
    tyx_grid = F.pad(yx_grid, (1, 0), value=t)  # (h*w, 3)

    # Evaluate deformation grid and return as (tyx, h, w)
    if isinstance(deformation_field, CubicCatmullRomGrid3d) or isinstance(
        deformation_field, CubicBSplineGrid3d
    ):
        shifts = deformation_field(tyx_grid)
    else:
        shifts = evaluate_deformation_field(
            deformation_field, tyx=tyx_grid, grid_type=grid_type
        )  # (h*w, c)
    shifts = einops.rearrange(shifts, "(h w) tyx -> tyx h w", h=h, w=w)
    return shifts


def resample_deformation_field(
    deformation_field: torch.Tensor,
    target_resolution: tuple[int, int, int],
) -> torch.Tensor:
    """Resample a deformation field to a new resolution.

    Parameters
    ----------
    deformation_field: torch.Tensor
        (yx, nt, nh, nw) deformation field data
    target_resolution: tuple[int, int, int]
        (nt, nh, nw) target resolution
    """
    nt, nh, nw = target_resolution

    # setup grid of points to evaluate over existing deformation grid
    t = torch.linspace(0, 1, steps=nt)
    y = torch.linspace(0, 1, steps=nh)
    x = torch.linspace(0, 1, steps=nw)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    tyx = einops.rearrange([tt, yy, xx], "tyx nt nh nw -> nt nh nw tyx")

    # evaluate existing field at new field grid points
    new_deformation_field = evaluate_deformation_field(
        deformation_field, tyx=tyx.to(deformation_field.device)
    )

    new_deformation_field = einops.rearrange(
        new_deformation_field, "nt nh nw tyx -> tyx nt nh nw"
    )
    return new_deformation_field


def image_shifts_to_deformation_field(
    shifts: torch.Tensor,  # (t, 2) shifts in pixels
    pixel_spacing: float,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Convert whole image shifts to a deformation field for compatibility.

    Parameters
    ----------
    shifts: torch.Tensor
        (t, 2) array of shifts for each frame in pixels (y, x)
    pixel_spacing: float
        Pixel spacing in Angstroms
    device: torch.device, optional
        Device for computation

    Returns
    -------
    deformation_field: torch.Tensor
        (2, t, 1, 1) deformation field with constant shifts per frame
    """
    if device is None:
        device = shifts.device
    else:
        shifts = shifts.to(device)

    # Rescale shifts in pixels to angstroms
    shifts = shifts * pixel_spacing

    # Create deformation field with one yx shift per frame
    # deformation_field = -1 * einops.rearrange(shifts, 't c -> c t 1 1')
    deformation_field = einops.rearrange(shifts, "t c -> c t 1 1")
    return deformation_field
