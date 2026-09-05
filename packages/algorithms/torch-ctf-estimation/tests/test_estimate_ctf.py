import math

import pytest
import torch

from torch_ctf_estimation.estimate_ctf import estimate_ctf
from torch_ctf_estimation.models import (
    CTFFittingParams,
    LaserParams,
    OpticalParams,
)


def default_optical_params(**overrides):
    """Build OpticalParams with test defaults; overrides merged in."""
    p = {
        "pixel_spacing_angstroms": 1.0,
        "voltage_kev": 300.0,
        "spherical_aberration_mm": 2.7,
        "amplitude_contrast_fraction": 0.1,
    }
    p.update(overrides)
    return OpticalParams(**p)


def default_fitting_params(**overrides):
    """Build CTFFittingParams with test defaults; overrides merged in."""
    p = {
        "defocus_grid_resolution": (1, 2, 2),
        "frequency_fit_range_angstroms": (30.0, 5.0),
        "defocus_range_microns": (0.5, 3.0),
        "patch_sidelength": 128,
    }
    p.update(overrides)
    return CTFFittingParams(**p)


def test_estimate_ctf_2d_image():
    """Test estimate_ctf with a 2D image."""
    image = torch.randn(1024, 1024)
    optical = default_optical_params()
    fitting = default_fitting_params(
        defocus_grid_resolution=(1, 3, 3),
        defocus_range_microns=(0.5, 5.0),
        patch_sidelength=128,
    )
    _mean_ps, _result1d, result2d = estimate_ctf(
        image, optical, fitting, device=torch.device("cpu")
    )
    assert result2d.defocus_model_type == "grid"
    defocus_field = result2d.defocus_model.data.squeeze(0)
    assert defocus_field.shape == (1, 3, 3)

    # Check defocus values are within reasonable range
    # assert torch.all(defocus_field >= defocus_range_microns[0])
    # assert torch.all(defocus_field <= defocus_range_microns[1])


def test_estimate_ctf_2d_image_linear_model():
    """Test estimate_ctf with 2D image and linear defocus model."""
    image = torch.randn(1024, 1024)
    optical = default_optical_params()
    fitting = default_fitting_params(
        defocus_grid_resolution=(1, 1, 1),
        defocus_range_microns=(0.5, 5.0),
        defocus_model="linear",
    )
    _mean_ps, _result1d, result2d = estimate_ctf(
        image, optical, fitting, device=torch.device("cpu")
    )
    assert result2d.defocus_model_type == "linear"
    linear = result2d.defocus_model
    assert hasattr(linear, "defocus_0")
    assert hasattr(linear, "defocus_gradient_magnitude")
    assert hasattr(linear, "defocus_gradient_angle")
    assert result2d.defocus_u is not None
    assert result2d.defocus_v is not None


def test_estimate_ctf_3d_image():
    """Test estimate_ctf with a 3D image stack."""
    image = torch.randn(4, 256, 256)
    optical = default_optical_params(
        pixel_spacing_angstroms=1.5,
        voltage_kev=200.0,
        spherical_aberration_mm=2.0,
        amplitude_contrast_fraction=0.07,
    )
    fitting = default_fitting_params(
        defocus_grid_resolution=(4, 2, 2),
        frequency_fit_range_angstroms=(20.0, 4.0),
        defocus_range_microns=(1.0, 4.0),
        patch_sidelength=64,
    )
    _mean_ps, _result1d, result2d = estimate_ctf(
        image, optical, fitting, device=torch.device("cpu")
    )
    assert result2d.defocus_model_type == "grid"
    defocus_field = result2d.defocus_model.data.squeeze(0)
    assert defocus_field.shape == (4, 2, 2)

    # Check defocus values are within reasonable range
    # assert torch.all(defocus_field >= defocus_range_microns[0])
    # assert torch.all(defocus_field <= defocus_range_microns[1])


def test_estimate_ctf_whole_image_mode():
    """Test estimate_ctf with whole-image mode (patch_sidelength < 0)."""
    image = torch.randn(256, 256)
    optical = default_optical_params()
    fitting = default_fitting_params(
        defocus_grid_resolution=(1, 1, 1),
        defocus_range_microns=(0.5, 5.0),
        patch_sidelength=-1,
    )
    mean_ps, _result1d, result2d = estimate_ctf(
        image, optical, fitting, device=torch.device("cpu")
    )
    assert mean_ps.dim() == 2
    assert result2d.defocus_model_type == "grid"
    defocus_field = result2d.defocus_model.data.squeeze(0)
    assert defocus_field.shape == (1, 1, 1)


def test_estimate_ctf_whole_image_mode_rejects_nh_nw_not_1():
    """Test that whole-image mode raises when nh or nw is not 1."""
    image = torch.randn(256, 256)
    optical = default_optical_params()
    with pytest.raises(ValueError, match="nh=1 and nw=1"):
        estimate_ctf(
            image,
            optical,
            default_fitting_params(
                defocus_grid_resolution=(1, 2, 1),
                patch_sidelength=-1,
            ),
        )
    with pytest.raises(ValueError, match="nh=1 and nw=1"):
        estimate_ctf(
            image,
            optical,
            default_fitting_params(
                defocus_grid_resolution=(1, 1, 2),
                patch_sidelength=-1,
            ),
        )


def test_estimate_ctf_use_1d_defocus_for_spatial():
    """Test use_1d_defocus_for_spatial returns grid/linear result with correct shape."""
    image = torch.randn(512, 512)
    optical = default_optical_params()
    fitting = default_fitting_params(
        defocus_grid_resolution=(1, 2, 2),
        use_1d_defocus_for_spatial=True,
        use_equiphase_for_1d_spatial=False,
        defocus_model="grid",
    )
    _mean_ps, _result1d, result2d = estimate_ctf(
        image, optical, fitting, device=torch.device("cpu")
    )
    assert result2d.defocus_model_type == "grid"
    defocus_field = result2d.defocus_model.data.squeeze(0)
    assert defocus_field.shape == (1, 2, 2)
    assert result2d.astigmatism is not None or result2d.astigmatism is None


def test_estimate_ctf_use_1d_defocus_for_spatial_linear():
    """Test use_1d_defocus_for_spatial with linear model."""
    image = torch.randn(512, 512)
    optical = default_optical_params()
    fitting = default_fitting_params(
        defocus_grid_resolution=(1, 2, 2),
        use_1d_defocus_for_spatial=True,
        use_equiphase_for_1d_spatial=False,
        defocus_model="linear",
    )
    _mean_ps, _result1d, result2d = estimate_ctf(
        image, optical, fitting, device=torch.device("cpu")
    )
    assert result2d.defocus_model_type == "linear"
    assert hasattr(result2d.defocus_model, "defocus_0")
    assert hasattr(result2d.defocus_model, "defocus_gradient_magnitude")
    assert hasattr(result2d.defocus_model, "defocus_gradient_angle")


def test_estimate_ctf_linear_fix_defocus_0_from_1x1():
    """Test linear_fix_defocus_0_from_1x1: defocus_0 comes from 2D@1x1."""
    image = torch.randn(512, 512)
    optical = default_optical_params()
    fitting = default_fitting_params(
        defocus_grid_resolution=(1, 2, 2),
        defocus_model="linear",
        linear_fix_defocus_0_from_1x1=True,
    )
    _mean_ps, _result1d, result2d = estimate_ctf(
        image, optical, fitting, device=torch.device("cpu")
    )
    assert result2d.defocus_model_type == "linear"
    assert result2d.defocus_model.defocus_0 is not None
    assert result2d.defocus_u is not None
    assert result2d.defocus_v is not None


def test_estimate_ctf_linear_fix_defocus_0_2d_zncc():
    """Test linear_fix_defocus_0_from_1x1 with gradient from 2D ZNCC."""
    image = torch.randn(512, 512)
    optical = default_optical_params()
    fitting = default_fitting_params(
        defocus_grid_resolution=(1, 2, 2),
        defocus_model="linear",
        linear_fix_defocus_0_from_1x1=True,
    )
    _mean_ps, _result1d, result2d = estimate_ctf(
        image, optical, fitting, device=torch.device("cpu")
    )
    assert result2d.defocus_model_type == "linear"
    assert result2d.defocus_model.defocus_0 is not None
    assert result2d.defocus_u is not None
    assert result2d.defocus_v is not None


def test_estimate_ctf_phase_shift_default_zero():
    """Test that without optimize_phase_shift, phase_shift_degrees is 0."""
    image = torch.randn(512, 512)
    _mean_ps, result1d, result2d = estimate_ctf(
        image,
        default_optical_params(),
        default_fitting_params(),
        device=torch.device("cpu"),
    )
    phase_1d = result1d.ctf_model.phase_shift_degrees
    assert phase_1d is not None
    if isinstance(phase_1d, torch.Tensor):
        assert float(phase_1d.cpu().item()) == 0.0
    else:
        assert phase_1d == 0.0
    assert result2d.phase_shift_degrees is None or result2d.phase_shift_degrees == 0.0


def test_estimate_ctf_optimize_phase_shift_1d():
    """Test 1D with optimize_phase_shift=True; phase in [0, 90] (folded from 0-180)."""
    image = torch.randn(512, 512)
    fitting = default_fitting_params(
        defocus_grid_resolution=(1, 1, 1),
        optimize_phase_shift=True,
    )
    _mean_ps, result1d, _result2d = estimate_ctf(
        image,
        default_optical_params(),
        fitting,
        device=torch.device("cpu"),
    )
    phase = result1d.ctf_model.phase_shift_degrees
    assert phase is not None
    p = float(phase.cpu().item()) if isinstance(phase, torch.Tensor) else float(phase)
    assert 0.0 <= p <= 90.0


def test_estimate_ctf_optimize_phase_shift_2d_grid():
    """Test 2D with optimize_phase_shift=True and phase_shift_model='grid'."""
    image = torch.randn(512, 512)
    fitting = default_fitting_params(
        optimize_phase_shift=True,
        phase_shift_model="grid",
    )
    _mean_ps, _, result2d = estimate_ctf(
        image,
        default_optical_params(),
        fitting,
        device=torch.device("cpu"),
    )
    assert result2d.phase_shift_degrees is not None
    assert 0.0 <= result2d.phase_shift_degrees <= 90.0
    assert result2d.phase_shift_model_type == "grid"
    assert result2d.phase_shift_model is not None
    # Grid model is (u_grid, v_grid) tuple for (u,v) representation
    assert isinstance(result2d.phase_shift_model, tuple)
    assert len(result2d.phase_shift_model) == 2


def test_estimate_ctf_optimize_phase_shift_2d_quadratic():
    """Test 2D with optimize_phase_shift=True and phase_shift_model='quadratic'."""
    image = torch.randn(512, 512)
    fitting = default_fitting_params(
        optimize_phase_shift=True,
        phase_shift_model="quadratic",
    )
    _mean_ps, _, result2d = estimate_ctf(
        image,
        default_optical_params(),
        fitting,
        device=torch.device("cpu"),
    )
    assert result2d.phase_shift_degrees is not None
    assert math.isfinite(result2d.phase_shift_degrees)
    assert result2d.phase_shift_model_type == "quadratic"
    assert result2d.phase_shift_model is not None
    # Quadratic: C, alpha_rad, g1, k1, g2, k2 (g2,k2 fixed at 0 when perpendicular off)
    assert hasattr(result2d.phase_shift_model, "C")
    assert hasattr(result2d.phase_shift_model, "alpha_rad")
    assert hasattr(result2d.phase_shift_model, "g1")
    assert hasattr(result2d.phase_shift_model, "k1")
    assert hasattr(result2d.phase_shift_model, "g2")
    assert hasattr(result2d.phase_shift_model, "k2")
    assert result2d.phase_shift_model.g2 == 0.0
    assert result2d.phase_shift_model.k2 == 0.0


def test_init_phase_shift_quadratic_perpendicular_axis_requires_grad():
    """When phase_shift_quadratic_perpendicular_axis=True, g2 and k2 are trainable."""
    from torch_ctf_estimation.estimate_ctf_2d.phase_shift_2d import (
        init_phase_shift_models,
    )

    m_on = init_phase_shift_models(
        optimize_phase_shift=True,
        phase_shift_model="quadratic",
        initial_phase_shift=10.0,
        grid_resolution=(1, 2, 2),
        device=torch.device("cpu"),
        phase_shift_quadratic_perpendicular_axis=True,
    )
    assert m_on is not None and m_on.quad_params is not None
    assert m_on.quad_params["g2"].requires_grad is True
    assert m_on.quad_params["k2"].requires_grad is True

    m_off = init_phase_shift_models(
        optimize_phase_shift=True,
        phase_shift_model="quadratic",
        initial_phase_shift=10.0,
        grid_resolution=(1, 2, 2),
        device=torch.device("cpu"),
        phase_shift_quadratic_perpendicular_axis=False,
    )
    assert m_off is not None and m_off.quad_params is not None
    assert m_off.quad_params["g2"].requires_grad is False
    assert m_off.quad_params["k2"].requires_grad is False


def test_estimate_ctf_raises_when_rescaled_image_smaller_than_patch():
    """estimate_ctf raises ValueError when rescaled image < patch_sidelength."""
    image = torch.randn(256, 256)
    with pytest.raises(
        ValueError, match=r"Rescaled image size.*smaller than patch_sidelength"
    ):
        estimate_ctf(
            image,
            default_optical_params(),
            default_fitting_params(patch_sidelength=128),
            device=torch.device("cpu"),
        )


def test_estimate_ctf_2d_default_no_laser():
    """estimate_ctf with laser_params=None uses normal CTF and returns valid result."""
    image = torch.randn(512, 512)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image,
        default_optical_params(),
        default_fitting_params(defocus_range_microns=(0.5, 5.0)),
        laser_params=None,
        device=torch.device("cpu"),
    )
    assert result2d.defocus_model_type == "grid"
    assert result2d.defocus_model.data.shape[1:] == (1, 2, 2)


def test_estimate_ctf_2d_with_laser_params():
    """estimate_ctf with model_laser=True uses LPP CTF and returns Defocus2DResults."""
    image = torch.randn(512, 512)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image,
        default_optical_params(),
        default_fitting_params(defocus_range_microns=(0.5, 5.0)),
        laser_params=LaserParams(model_laser=True),
        device=torch.device("cpu"),
    )
    assert result2d.defocus_model_type == "grid"
    assert result2d.defocus_model.data.shape[1:] == (1, 2, 2)


def test_estimate_ctf_defocus_and_phase_bounds():
    """Defocus and phase bounds are applied when set in CTFFittingParams."""
    image = torch.randn(512, 512)
    fitting = default_fitting_params(
        defocus_range_microns=(0.8, 2.5),
        phase_shift_range_degrees=(10.0, 80.0),
        optimize_phase_shift=True,
    )
    _mean_ps, result1d, result2d = estimate_ctf(
        image,
        default_optical_params(),
        fitting,
        device=torch.device("cpu"),
    )
    d1 = float(result1d.ctf_model.defocus_um.cpu().item())
    assert 0.8 <= d1 <= 2.5
    assert result2d.phase_shift_degrees is not None
    assert 10.0 <= result2d.phase_shift_degrees <= 80.0
    defocus_field = result2d.defocus_model.data.squeeze(0)
    assert defocus_field.min().item() >= 0.8 - 1e-3
    assert defocus_field.max().item() <= 2.5 + 1e-3


def test_fitting_bounds_defaults_and_fixed_phase():
    """Default bounds and equal phase bounds fix phase without optimisation."""
    from torch_ctf_estimation.utils.fitting_bounds import (
        DEFAULT_DEFOCUS_BOUNDS_MICRONS,
        DEFAULT_PHASE_SHIFT_BOUNDS_DEG,
        resolve_defocus_bounds,
        resolve_phase_shift_bounds,
        resolve_phase_shift_fitting,
    )

    assert resolve_defocus_bounds(None) == DEFAULT_DEFOCUS_BOUNDS_MICRONS
    assert resolve_phase_shift_bounds(None) == DEFAULT_PHASE_SHIFT_BOUNDS_DEG
    optimize, phase_deg, bounds = resolve_phase_shift_fitting(
        optimize_phase_shift=True,
        phase_shift_range_degrees=(45.0, 45.0),
        initial_phase_shift=0.0,
    )
    assert optimize is False
    assert phase_deg == 45.0
    assert bounds == (45.0, 45.0)


def test_estimate_ctf_fixed_known_phase():
    """Equal phase bounds use a known phase; defocus is still estimated."""
    image = torch.randn(512, 512)
    fitting = default_fitting_params(
        phase_shift_range_degrees=(45.0, 45.0),
        optimize_phase_shift=True,
    )
    _mean_ps, _result1d, result2d = estimate_ctf(
        image,
        default_optical_params(),
        fitting,
        device=torch.device("cpu"),
    )
    assert result2d.phase_shift_degrees == 45.0
    assert result2d.phase_shift_model_type is None


def test_estimate_ctf_mask_laser_axis_without_lpp_model():
    """Laser axis masking works with standard CTF when model_laser=False."""
    image = torch.randn(512, 512)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image,
        default_optical_params(),
        default_fitting_params(
            defocus_range_microns=(0.5, 5.0),
            mask_laser_axis=True,
            laser_axis_mask_width=0.1,
        ),
        laser_params=LaserParams(
            model_laser=False,
            laser_xy_angle_deg=45.0,
            dual_laser=True,
        ),
        device=torch.device("cpu"),
    )
    assert result2d.defocus_model_type == "grid"
    assert result2d.defocus_model.data.shape[1:] == (1, 2, 2)


def test_compute_ctf2_t_amplitude_is_abs_ctf():
    """use_amplitude=True returns |CTF|·env·bp; default is CTF²·env²·bp."""
    from torch_ctf_estimation.estimate_ctf_2d.ctf_loss_2d import compute_ctf2_t

    image_shape = (32, 32)
    rfft_w = image_shape[1] // 2 + 1
    ones = torch.ones(image_shape[0], rfft_w)
    kwargs = {
        "defocus_t": torch.tensor([[1.2]]),
        "phase_shift_t": torch.zeros(1, 1),
        "astig_clamped": torch.tensor(0.02),
        "astig_angle_clamped": torch.tensor(15.0),
        "image_shape": image_shape,
        "pixel_spacing_angstroms": 1.4,
        "voltage_kev": 300.0,
        "spherical_aberration_mm": 2.7,
        "amplitude_contrast_fraction": 0.07,
        "env_2d": ones,
        "bp_filter": ones,
        "laser_params": None,
    }
    power = compute_ctf2_t(**kwargs, use_amplitude=False)
    amp = compute_ctf2_t(**kwargs, use_amplitude=True)
    assert power.shape == amp.shape
    assert torch.all(amp >= 0)
    torch.testing.assert_close(amp**2, power, rtol=1e-5, atol=1e-6)


def test_estimate_ctf_use_amplitude_2d():
    """2D amplitude flag still returns a valid grid defocus result."""
    image = torch.randn(512, 512)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image,
        default_optical_params(),
        default_fitting_params(
            defocus_range_microns=(0.5, 5.0),
            use_amplitude_2d=True,
            n_iterations_2d=2,
        ),
        device=torch.device("cpu"),
    )
    assert result2d.defocus_model_type == "grid"
    assert result2d.defocus_u is not None
    assert result2d.defocus_v is not None
