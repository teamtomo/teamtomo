"""Tests for DeformationField resampling, construction and devices."""

import pytest
import torch

from torch_motion_correction.deformation_field import DeformationField


def _sample_points(n: int = 200) -> torch.Tensor:
    return torch.rand(n, 3, generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("grid_type", ["catmull_rom", "bspline"])
def test_resample_to_same_resolution_reproduces_field(grid_type):
    torch.manual_seed(0)
    field = DeformationField(torch.randn(2, 5, 4, 3), grid_type=grid_type)
    resampled = field.resample((5, 4, 3))
    points = _sample_points()
    with torch.no_grad():
        torch.testing.assert_close(resampled(points), field(points), atol=1e-4, rtol=0)


def test_resample_to_bspline_passes_through_knot_values():
    torch.manual_seed(0)
    field = DeformationField(torch.randn(2, 5, 4, 3), grid_type="catmull_rom")
    bspline = field.resample((5, 4, 3), grid_type="bspline")
    assert bspline.grid_type == "bspline"
    t, y, x = (torch.linspace(0, 1, n) for n in (5, 4, 3))
    knots = torch.stack(torch.meshgrid(t, y, x, indexing="ij"), dim=-1)
    with torch.no_grad():
        torch.testing.assert_close(bspline(knots), field(knots), atol=1e-4, rtol=0)


@pytest.mark.parametrize("grid_type", ["catmull_rom", "bspline"])
def test_from_frame_shifts_passes_through_shifts(grid_type):
    torch.manual_seed(0)
    shifts = torch.randn(7, 2)
    field = DeformationField.from_frame_shifts(
        shifts, pixel_spacing=2.0, grid_type=grid_type
    )
    frames = torch.stack(
        [torch.linspace(0, 1, 7), torch.zeros(7), torch.zeros(7)], dim=-1
    )
    with torch.no_grad():
        torch.testing.assert_close(field(frames), 2.0 * shifts, atol=1e-4, rtol=0)


@pytest.mark.parametrize("grid_type", ["catmull_rom", "bspline"])
def test_from_initial_field_base_reproduces_initial_field(grid_type):
    torch.manual_seed(0)
    initial = DeformationField(torch.randn(2, 5, 4, 3))
    new_field, base_field = DeformationField.from_initial_field(
        resolution=(5, 4, 3),
        initial_field=initial,
        grid_type=grid_type,
        device=torch.device("cpu"),
    )
    assert base_field.data.grad_fn is None
    assert not base_field.data.requires_grad
    # The base passes through the initial field at the control points, up to the
    # constant per axis that the mean removal takes out
    t, y, x = (torch.linspace(0, 1, n) for n in (5, 4, 3))
    knots = torch.stack(torch.meshgrid(t, y, x, indexing="ij"), dim=-1)
    with torch.no_grad():
        difference = base_field(knots) - initial(knots)
        torch.testing.assert_close(
            difference,
            difference.mean(dim=(0, 1, 2)).expand_as(difference),
            atol=1e-4,
            rtol=0,
        )
        torch.testing.assert_close(new_field(knots), torch.zeros_like(difference))


def test_to_same_device_returns_same_field():
    field = DeformationField(torch.zeros(2, 3, 2, 2))
    assert field.to("cpu") is field
    assert field.to(torch.device("cpu")) is field
