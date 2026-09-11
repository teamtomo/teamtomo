"""Estimate sample thickness in 1D from a power spectrum."""

from typing import Optional

import einops
import torch
from torch_ctf.ctf_thickness import calculate_ctf_thickness_1d
from torch_grid_utils.fftfreq_grid import fftfreq_to_spatial_frequency

from torch_ctf_estimation.estimate_ctf_1d.estimate_ctf_1d_utils import (
    get_background_result,
)
from torch_ctf_estimation.models import LaserParams
from torch_ctf_estimation.models.results_models import (
    Thickness1DResults,
    _Background1DResult,
)


def estimate_thickness_1d(
    power_spectrum: torch.Tensor,
    image_sidelength: int,
    frequency_fit_range_angstroms: tuple[float, float],
    defocus_um: float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    pixel_spacing_angstroms: float,
    phase_shift_deg: float = 0.0,
    thickness_range_angstroms: tuple[float, float] = (300.0, 4000.0),
    thickness_step_angstroms: float = 100.0,
    background_result: Optional[_Background1DResult] = None,
    use_equiphase: bool = False,
    equiphase_defocus_um: Optional[float] = None,
    equiphase_astigmatism_um: Optional[float] = None,
    equiphase_astigmatism_angle_deg: Optional[float] = None,
    equiphase_phase_shift_deg: Optional[float] = None,
    laser_params: Optional[LaserParams] = None,
    equiphase_n_theta: int = 64,
) -> Thickness1DResults:
    """
    Estimate sample thickness in 1D from a power spectrum.

    Fits a log-spline background, then runs a grid search over sample thickness
    (300-4000 Å in 100 Å steps by default) using L2-normalised cross-correlation
    against the prepared 1D power spectrum. The thickness-modulated power
    spectrum form (Thon-ring model) is used throughout.

    Parameters
    ----------
    power_spectrum : torch.Tensor
        (h, w) array containing 2D rfft (no fftshift applied).
    image_sidelength : int
        Sidelength of 2D images prior to rfft calculation.
    frequency_fit_range_angstroms : tuple[float, float]
        (low, high) spatial frequency cutoffs for fitting in Angstroms.
    defocus_um : float
        Fixed defocus in micrometers (positive = underfocused).
    voltage_kev : float
        Acceleration voltage in keV.
    spherical_aberration_mm : float
        Spherical aberration in mm.
    amplitude_contrast : float
        Amplitude contrast fraction.
    pixel_spacing_angstroms : float
        Isotropic pixel spacing in Angstroms.
    phase_shift_deg : float
        Fixed phase shift in degrees. Default 0.0.
    thickness_range_angstroms : tuple[float, float]
        (low, high) thickness range for the grid search in Angstroms.
        Default (300.0, 4000.0).
    thickness_step_angstroms : float
        Grid step size in Angstroms. Default 100.0.
    background_result : _Background1DResult, optional
        Pre-fitted background; if provided, skips background fitting.
    use_equiphase : bool
        If True, use equiphase shell average for the 1D spectrum. Default False.
    equiphase_defocus_um : float, optional
        Mean defocus (µm) for equiphase when ``use_equiphase`` is True.
    equiphase_astigmatism_um : float, optional
        Astigmatism (µm) for equiphase.
    equiphase_astigmatism_angle_deg : float, optional
        Astigmatism angle (degrees) for equiphase.
    equiphase_phase_shift_deg : float, optional
        Phase shift (degrees) for equiphase.
    laser_params : LaserParams, optional
        Optional laser preset for LPP phase in equiphase chi.
    equiphase_n_theta : int
        Azimuth samples per shell for equiphase. Default 64.

    Returns
    -------
    Thickness1DResults
        Best-fit thickness, ZNCC cross-correlations for all test thicknesses,
        and the 1D power spectrum / background model used.
    """
    low_ang, high_ang = frequency_fit_range_angstroms

    bg_result = get_background_result(
        power_spectrum=power_spectrum,
        image_sidelength=image_sidelength,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        background_result=background_result,
        use_equiphase=use_equiphase,
        equiphase_defocus_um=equiphase_defocus_um,
        equiphase_astigmatism_um=equiphase_astigmatism_um,
        equiphase_astigmatism_angle_deg=equiphase_astigmatism_angle_deg,
        equiphase_phase_shift_deg=equiphase_phase_shift_deg,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        laser_params=laser_params,
        equiphase_n_theta=equiphase_n_theta,
    )

    device = power_spectrum.device
    dtype = bg_result.raps_in_fit_range.dtype

    t_low, t_high = thickness_range_angstroms
    test_thicknesses = torch.arange(
        t_low,
        t_high + thickness_step_angstroms,
        thickness_step_angstroms,
        device=device,
        dtype=dtype,
    )

    rows: list[torch.Tensor] = []
    for t_val in test_thicknesses:
        ps_t = calculate_ctf_thickness_1d(
            return_power_spectrum=True,
            sample_thickness_angstrom=t_val,
            defocus=defocus_um,
            voltage=voltage_kev,
            spherical_aberration=spherical_aberration_mm,
            amplitude_contrast=amplitude_contrast,
            phase_shift=phase_shift_deg,
            pixel_size=pixel_spacing_angstroms,
            n_samples=image_sidelength // 2 + 1,
            oversampling_factor=3,
        )
        rows.append(ps_t)
    simulated_ps = torch.stack(rows, dim=0).to(device=device, dtype=dtype)
    fit_mask = bg_result.fit_mask.to(device)
    simulated_ps_in_fit_range = simulated_ps[:, fit_mask]

    raps = bg_result.raps_in_fit_range.to(device)
    normalised_raps = raps / (torch.linalg.norm(raps) + 1e-8)
    norms = torch.linalg.norm(simulated_ps_in_fit_range, dim=-1, keepdim=True)
    simulated_ps_normalised = simulated_ps_in_fit_range / (norms + 1e-8)
    cross_correlations = einops.einsum(
        simulated_ps_normalised, normalised_raps, "t f, f -> t"
    )

    best_idx = int(torch.argmax(cross_correlations).item())
    best_thickness = float(test_thicknesses[best_idx].item())
    best_cc = float(cross_correlations[best_idx].item())

    return Thickness1DResults(
        thickness_angstroms=best_thickness,
        cross_correlation_final=best_cc,
        frequencies_1d=fftfreq_to_spatial_frequency(
            bg_result.freqs, pixel_spacing_angstroms
        ),
        powerspectrum_1d=bg_result.rotationally_averaged_power_spectrum,
        background_model=bg_result.background_model,
        test_thicknesses=test_thicknesses,
        cross_correlations=cross_correlations,
        low_frequency_fit=1.0 / low_ang,
        high_frequency_fit=1.0 / high_ang,
    )
