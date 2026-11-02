"""Tilt axis and tilt magnitude derived from a linear defocus model."""

import math

import pytest
import torch

from torch_ctf_estimation.estimate_ctf import estimate_ctf
from torch_ctf_estimation.models import (
    CTFFittingParams,
    Defocus2DResults,
    LinearDefocusModel,
    OpticalParams,
    linear_tilt_axis_and_magnitude_deg,
)


def _linear_result(angle_deg: float, magnitude: float) -> Defocus2DResults:
    return Defocus2DResults(
        defocus_model_type="linear",
        defocus_model=LinearDefocusModel(
            defocus_0=2.0,
            defocus_gradient_magnitude=magnitude,
            defocus_gradient_angle=angle_deg,
        ),
    )


# 1001 px at 1 A: one normalised unit is 1000 A = 0.1 um, so a 0.1 um gradient is a
# slope of 1 (45 degrees). The tilt axis follows the TeamTomo convention of
# torch-tilt-series (Rz(axis) @ Ry(tilt)), torch-tiltxcorr and
# torch-refine-tilt-axis-angle: the axis points along (-sin a, cos a) in (x, y) =
# (column, row), so a = 0 is the y axis. Tilting about it moves the specimen up and
# down along (cos a, sin a), which is where the defocus gradient points.
@pytest.mark.parametrize(
    ("gradient_angle", "magnitude", "expected_axis"),
    [
        (90.0, 0.1, 0.0),  # defocus rises along columns (+x): axis along y
        (0.0, 0.1, 90.0),  # defocus rises along rows (+y): axis along x
        (45.0, 0.1, 45.0),  # rises towards +x+y: axis along (-1, 1)
        (135.0, -0.1, 135.0),  # rises towards -x+y; the sign does not move the axis
        (90.0, -0.1, 0.0),  # rises towards -x: same axis as rising towards +x
    ],
)
def test_tilt_axis_is_perpendicular_to_the_defocus_gradient(
    gradient_angle, magnitude, expected_axis
):
    """The reported axis is the line of constant defocus, in TeamTomo's convention."""
    axis, tilt = linear_tilt_axis_and_magnitude_deg(
        _linear_result(gradient_angle, magnitude), 1.0, (1001, 1001)
    )
    assert axis == pytest.approx(expected_axis, abs=1e-6)
    assert tilt == pytest.approx(45.0, abs=1e-6)


@pytest.mark.parametrize(
    ("gradient_angle", "expected_tilt"),
    [
        (90.0, 45.0),  # along columns: 1000 A per normalised unit
        (0.0, math.degrees(math.atan(2.0))),  # along rows: 500 A per normalised unit
    ],
)
def test_tilt_magnitude_uses_the_image_length_along_the_gradient(
    gradient_angle, expected_tilt
):
    """Normalised positions span (h - 1) rows and (w - 1) columns."""
    _axis, tilt = linear_tilt_axis_and_magnitude_deg(
        _linear_result(gradient_angle, 0.1), 1.0, (501, 1001)
    )
    assert tilt == pytest.approx(expected_tilt, abs=1e-6)


def test_integer_image_size_means_a_square_image():
    result = _linear_result(30.0, 0.07)
    assert linear_tilt_axis_and_magnitude_deg(result, 1.5, 801) == pytest.approx(
        linear_tilt_axis_and_magnitude_deg(result, 1.5, (801, 801))
    )


def test_estimate_ctf_reports_axis_perpendicular_to_gradient():
    """estimate_ctf fills tilt_axis_angle_deg from the fitted gradient direction."""
    image = torch.randn(1024, 1024, generator=torch.Generator().manual_seed(0))
    optical = OpticalParams(
        pixel_spacing_angstroms=1.0, amplitude_contrast_fraction=0.1
    )
    fitting = CTFFittingParams(
        defocus_grid_resolution=(1, 2, 2),
        frequency_fit_range_angstroms=(30.0, 5.0),
        defocus_range_microns=(0.5, 3.0),
        patch_sidelength=128,
        defocus_model="linear",
    )
    _mean_ps, _result1d, result2d = estimate_ctf(
        image, optical, fitting, device=torch.device("cpu")
    )
    gradient_angle = result2d.defocus_model.defocus_gradient_angle
    # gradient (row, col) direction (cos g, sin g) -> (x, y) = (sin g, cos g), at
    # 90 - g from +x; for a square image that is the axis angle
    assert result2d.tilt_axis_angle_deg == pytest.approx(
        (90.0 - gradient_angle) % 180.0, abs=1e-4
    )
    assert result2d.tilt_magnitude_deg >= 0.0
