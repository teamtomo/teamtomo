"""The laser-axis mask covers the strip the LPP model puts the laser on."""

import pytest
import torch
from torch_ctf import calc_LPP_phase
from torch_grid_utils import fftfreq_grid

from torch_ctf_estimation.models import LaserParams
from torch_ctf_estimation.utils.laser_axis_mask import build_laser_axis_mask


def _lpp_phase(sidelength: int, laser_xy_angle_deg: float) -> torch.Tensor:
    lp = LaserParams(model_laser=True)
    grid = fftfreq_grid((sidelength, sidelength), rfft=True)  # 1 A/px: cycles/A
    return calc_LPP_phase(
        grid,
        NA=lp.NA,
        laser_wavelength_angstrom=lp.laser_wavelength_angstrom,
        focal_length_angstrom=lp.focal_length_angstrom,
        laser_xy_angle_deg=laser_xy_angle_deg,
        laser_xz_angle_deg=lp.laser_xz_angle_deg,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=lp.laser_polarization_angle_deg,
        peak_phase_deg=lp.peak_phase_deg,
        voltage=300.0,
    )


@pytest.mark.parametrize("sidelength", [64, 65])
@pytest.mark.parametrize("laser_xy_angle_deg", [0.0, 30.0, 90.0])
def test_single_laser_mask_covers_lpp_phase(sidelength, laser_xy_angle_deg):
    """Every pixel with more than half the peak laser phase is masked."""
    phase = _lpp_phase(sidelength, laser_xy_angle_deg)
    mask = build_laser_axis_mask(
        image_shape=(sidelength, sidelength),
        laser_xy_angle_deg=laser_xy_angle_deg,
        dual_laser=False,
        mask_width=0.1,
    )
    high_phase = phase > 0.5 * phase.max()
    assert high_phase.sum() > 0
    assert torch.all(mask[high_phase] == 0)


@pytest.mark.parametrize("laser_xy_angle_deg", [0.0, 90.0])
def test_single_laser_mask_axis_direction(laser_xy_angle_deg):
    """0 degrees masks the ky axis (column kx = 0), 90 degrees the kx axis (row 0)."""
    mask = build_laser_axis_mask(
        image_shape=(32, 32),
        laser_xy_angle_deg=laser_xy_angle_deg,
        dual_laser=False,
        mask_width=0.1,
    )
    if laser_xy_angle_deg == 0.0:
        assert torch.all(mask[:, 0] == 0)
        assert mask[0, 1:].sum() > 0
    else:
        assert torch.all(mask[0, :] == 0)
        assert mask[1:, 0].sum() > 0
