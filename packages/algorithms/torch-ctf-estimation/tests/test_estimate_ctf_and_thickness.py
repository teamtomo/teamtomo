import torch

from torch_ctf_estimation.estimate_ctf import (
    CTFEstimationResult,
    estimate_ctf_and_thickness,
)
from torch_ctf_estimation.models import (
    CTFFittingParams,
    OpticalParams,
    ThicknessParams,
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


def default_thickness_params(**overrides):
    """Build ThicknessParams with fast test defaults; overrides merged in."""
    p = {
        "refine_dim": "none",
        "thickness_range_angstroms": (300.0, 1000.0),
        "thickness_step_angstroms": 350.0,
        "n_iterations": 3,
    }
    p.update(overrides)
    return ThicknessParams(**p)


def test_estimate_ctf_and_thickness_grid_search_only():
    """refine_dim='none' runs the 1D thickness grid search but no joint refine."""
    image = torch.randn(512, 512)
    optical = default_optical_params()
    fitting = default_fitting_params()
    thickness = default_thickness_params(refine_dim="none")

    result = estimate_ctf_and_thickness(
        image, optical, fitting, thickness, device=torch.device("cpu")
    )
    assert isinstance(result, CTFEstimationResult)
    assert result.thickness1d is not None
    assert 300.0 <= result.thickness1d.thickness_angstroms <= 1000.0
    assert result.thickness_joint is None


def test_estimate_ctf_and_thickness_refine_thickness_only():
    """refine_dim='thickness' refines thickness only, defocus stays at the 2D mean."""
    image = torch.randn(512, 512)
    optical = default_optical_params()
    fitting = default_fitting_params()
    thickness = default_thickness_params(refine_dim="thickness")

    result = estimate_ctf_and_thickness(
        image, optical, fitting, thickness, device=torch.device("cpu")
    )
    assert result.thickness1d is not None
    assert result.thickness_joint is not None
    assert 300.0 <= result.thickness_joint.thickness_angstroms <= 1000.0


def test_estimate_ctf_and_thickness_refine_1d():
    """refine_dim='1d' jointly refines scalar defocus and thickness."""
    image = torch.randn(512, 512)
    optical = default_optical_params()
    fitting = default_fitting_params()
    thickness = default_thickness_params(refine_dim="1d")

    result = estimate_ctf_and_thickness(
        image, optical, fitting, thickness, device=torch.device("cpu")
    )
    assert result.thickness_joint is not None
    assert result.result2d.defocus_u is not None
    assert result.result2d.defocus_v is not None


def test_estimate_ctf_and_thickness_refine_2d():
    """refine_dim='2d' jointly refines the 2D defocus field and thickness."""
    image = torch.randn(512, 512)
    optical = default_optical_params()
    fitting = default_fitting_params()
    thickness = default_thickness_params(refine_dim="2d")

    result = estimate_ctf_and_thickness(
        image, optical, fitting, thickness, device=torch.device("cpu")
    )
    assert result.thickness_joint is not None
    assert result.thickness_joint.mean_thickness is not None


def test_estimate_ctf_and_thickness_use_equiphase():
    """use_equiphase runs the equiphase-averaged 1D thickness path without error."""
    image = torch.randn(512, 512)
    optical = default_optical_params()
    fitting = default_fitting_params()
    thickness = default_thickness_params(refine_dim="none", use_equiphase=True)

    result = estimate_ctf_and_thickness(
        image, optical, fitting, thickness, device=torch.device("cpu")
    )
    assert result.thickness1d is not None


def test_estimate_ctf_and_thickness_use_tilt_corrected_ps():
    """use_tilt_corrected_ps runs the tilt-corrected mean PS path without error."""
    image = torch.randn(4, 256, 256)
    optical = default_optical_params()
    fitting = default_fitting_params(
        defocus_grid_resolution=(4, 1, 1),
        patch_sidelength=64,
    )
    thickness = default_thickness_params(refine_dim="none", use_tilt_corrected_ps=True)

    result = estimate_ctf_and_thickness(
        image, optical, fitting, thickness, device=torch.device("cpu")
    )
    assert result.thickness1d is not None
