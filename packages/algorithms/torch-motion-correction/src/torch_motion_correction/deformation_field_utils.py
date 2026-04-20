"""Compatibility shims — logic now lives on DeformationField in types.py."""

import einops
import torch
import torch.nn.functional as F
from torch_cubic_spline_grids import CubicBSplineGrid3d, CubicCatmullRomGrid3d

from torch_motion_correction.deformation_field import DeformationField


def evaluate_deformation_field(
    deformation_field: DeformationField | torch.Tensor,
    tyx: torch.Tensor,
    grid_type: str = "catmull_rom",
) -> torch.Tensor:
    """Evaluate shifts from a deformation field at arbitrary (t, y, x) coordinates."""
    if isinstance(deformation_field, torch.Tensor):
        deformation_field = DeformationField(
            data=deformation_field, grid_type=grid_type
        )
    return deformation_field.evaluate_at(tyx)


def evaluate_deformation_field_at_t(
    deformation_field: DeformationField
    | torch.Tensor
    | CubicCatmullRomGrid3d
    | CubicBSplineGrid3d,
    t: float,
    grid_shape: tuple[int, int],
    grid_type: str = "catmull_rom",
) -> torch.Tensor:
    """Evaluate a dense shift grid at a single normalized timepoint."""
    if isinstance(deformation_field, (CubicCatmullRomGrid3d, CubicBSplineGrid3d)):
        device = deformation_field.data.device
        h, w = grid_shape
        y = torch.linspace(0, 1, steps=h, device=device)
        x = torch.linspace(0, 1, steps=w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        yx_grid = einops.rearrange([yy, xx], "yx h w -> (h w) yx")
        tyx_grid = F.pad(yx_grid, (1, 0), value=t)
        shifts = deformation_field(tyx_grid)
        return einops.rearrange(shifts, "(h w) tyx -> tyx h w", h=h, w=w)

    if isinstance(deformation_field, torch.Tensor):
        deformation_field = DeformationField(
            data=deformation_field, grid_type=grid_type
        )
    return deformation_field.evaluate_at_t(t=t, grid_shape=grid_shape)


def resample_deformation_field(
    deformation_field: DeformationField | torch.Tensor,
    target_resolution: tuple[int, int, int],
) -> torch.Tensor:
    """Resample a deformation field to a new resolution. Returns raw tensor."""
    if isinstance(deformation_field, torch.Tensor):
        deformation_field = DeformationField(data=deformation_field)
    return deformation_field.resample(target_resolution).data


def image_shifts_to_deformation_field(
    shifts: torch.Tensor,
    pixel_spacing: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert per-frame whole-image shifts to a deformation field tensor."""
    return DeformationField.from_frame_shifts(
        shifts=shifts, pixel_spacing=pixel_spacing, device=device
    ).data
