"""Estimate CTF in 1D from a power spectrum."""

from collections.abc import Callable
from typing import Optional

import torch
from torch_grid_utils.fftfreq_grid import fftfreq_to_spatial_frequency

from torch_ctf_estimation.estimate_ctf_1d.estimate_ctf_1d_utils import (
    compute_final_1d_l2_cross_correlation,
    get_background_result,
    grid_search_defocus_and_envelope_1d,
    refine_defocus_and_b_factor_1d,
)
from torch_ctf_estimation.models import CTF, Defocus1DResults, LaserParams
from torch_ctf_estimation.models.results_models import _Background1DResult
from torch_ctf_estimation.utils.fitting_bounds import (
    resolve_defocus_bounds,
    resolve_phase_shift_bounds,
)


def estimate_ctf_1d(
    power_spectrum: torch.Tensor,
    image_sidelength: int,
    frequency_fit_range_angstroms: tuple[float, float],
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    pixel_spacing_angstroms: float,
    defocus_range_microns: tuple[float, float] | None = None,
    optimize_envelope: bool = True,
    b_range: tuple[float, float] = (0.0, 100.0),
    b_step: float = 1.0,
    refine_steps: int = 40,
    refine_defocus_lr: float = 0.01,
    refine_b_factor_lr: float = 1.0,
    initial_defocus: Optional[float] = None,
    background_result: _Background1DResult | None = None,
    optimize_phase_shift: bool = False,
    initial_phase_shift: float = 0.0,
    phase_shift_range: tuple[float, float] | None = None,
    phase_shift_step: float = 5.0,
    phase_shift_lr: float = 5.0,
    use_equiphase: bool = False,
    equiphase_defocus_um: float | None = None,
    equiphase_astigmatism_um: float | None = None,
    equiphase_astigmatism_angle_deg: float | None = None,
    equiphase_phase_shift_deg: float | None = None,
    laser_params: LaserParams | None = None,
    equiphase_n_theta: int = 64,
    early_stopper: Callable[[float], bool] | None = None,
) -> Defocus1DResults:
    """
    Estimate CTF in 1D from a power spectrum.

    Fits a background spline, runs a grid search over defocus (and optionally
    B-factor envelope), then refines defocus and B by gradient descent to
    maximise zero-normalised cross correlation.

    Parameters
    ----------
    power_spectrum : torch.Tensor
        (h, w) array containing 2D rfft (no fftshift applied).
    image_sidelength : int
        Sidelength of 2D images prior to rfft calculation.
    frequency_fit_range_angstroms : tuple[float, float]
        (low, high) spatial frequency cutoffs for fitting in angstroms.
    defocus_range_microns : tuple[float, float] or None
        (low, high) defoci in microns for grid search and refinement bounds.
        If None, grid search uses a wide internal range and refinement is unbounded.
    voltage_kev : float
        Acceleration voltage in keV.
    spherical_aberration_mm : float
        Spherical aberration in mm.
    amplitude_contrast : float
        Amplitude contrast fraction.
    pixel_spacing_angstroms : float
        Isotropic pixel spacing in angstroms.
    optimize_envelope : bool
        Whether to optimize the B-factor envelope.
    b_range : tuple[float, float]
        (low, high) B-factor range for envelope optimization.
    b_step : float
        Step size for envelope optimization in grid search.
    refine_steps : int
        Number of gradient descent steps for defocus (and B) refinement. Default 40.
        Set to 0 to disable refinement and use grid-search result only.
    refine_defocus_lr : float
        Learning rate for defocus in refinement. Default 0.001.
    refine_b_factor_lr : float
        Learning rate for B factor in refinement when optimize_envelope is True.
        Default 0.1.
    initial_defocus : float, optional
        If provided, skip the grid search and only run gradient-descent refinement
        from this defocus (e.g. from a 2D fit). Background fit is still performed
        unless background_result is also provided.
    background_result : _Background1DResult, optional
        If provided together with initial_defocus, skip the background fit and use
        this pre-fitted background to subtract from the rotationally averaged
        spectrum (e.g. reuse background from mean spectrum for all patches).
    optimize_phase_shift : bool
        If True, grid search and refine phase shift. Default False.
    initial_phase_shift : float
        Initial phase shift in degrees when optimize_phase_shift is True. Default 0.0.
    phase_shift_range : tuple[float, float] or None
        (low, high) phase shift bounds in degrees. If None, phase is unbounded
        during refinement (grid search uses 0–180° internally).
    phase_shift_step : float
        Phase shift grid step in degrees for grid search. Default 5.0.
    phase_shift_lr : float
        Learning rate for phase shift in refinement. Default 1.0.
    use_equiphase : bool
        If True, use equiphase shell average for 1D spectrum. Default False.
    equiphase_defocus_um : float, optional
        Mean defocus (µm) for equiphase when use_equiphase is True.
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
    early_stopper : callable or None, optional
        Stateful ``(loss) -> should_stop`` callback for Adam refinement.
        Default None (run all ``refine_steps``).

    Returns
    -------
    Defocus1DResults
        Results from 1D CTF estimation containing frequencies, power spectrum,
        background model, and CTF fitting results (using refined defocus and B).
    """
    low_ang, high_ang = frequency_fit_range_angstroms
    defocus_range_microns = resolve_defocus_bounds(defocus_range_microns)
    phase_shift_range = resolve_phase_shift_bounds(phase_shift_range)

    # -------------------------------------------------------------------------
    # Step 1: Background — use existing or fit spline to rotationally averaged spectrum
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Branch A: Refinement only — initial_defocus provided, skip grid search
    # -------------------------------------------------------------------------
    if initial_defocus is not None:
        # Refine from given defocus only (no grid search)
        refined_defocus, refined_B, refined_phase_shift = (
            refine_defocus_and_b_factor_1d(
                initial_defocus=initial_defocus,
                initial_B=None,
                raps_in_fit_range=bg_result.raps_in_fit_range,
                spatial_freqs=bg_result.spatial_freqs,
                fit_mask=bg_result.fit_mask,
                image_sidelength=image_sidelength,
                voltage_kev=voltage_kev,
                spherical_aberration_mm=spherical_aberration_mm,
                amplitude_contrast=amplitude_contrast,
                pixel_spacing_angstroms=pixel_spacing_angstroms,
                defocus_range_microns=defocus_range_microns,
                optimize_envelope=False,
                n_iterations=refine_steps,
                defocus_lr=refine_defocus_lr,
                b_factor_lr=refine_b_factor_lr,
                initial_phase_shift=initial_phase_shift
                if optimize_phase_shift
                else None,
                optimize_phase_shift=optimize_phase_shift,
                phase_shift_lr=phase_shift_lr,
                phase_shift_range=phase_shift_range,
                early_stopper=early_stopper,
            )
        )
        phase_deg = (
            float(refined_phase_shift.cpu().item())
            if refined_phase_shift is not None
            else initial_phase_shift
        )
        cc_final = compute_final_1d_l2_cross_correlation(
            bg_result.raps_in_fit_range,
            bg_result.spatial_freqs,
            bg_result.fit_mask,
            image_sidelength,
            float(refined_defocus.detach().cpu().item()),
            envelope_B=None,
            phase_shift_deg=phase_deg,
            voltage_kev=voltage_kev,
            spherical_aberration_mm=spherical_aberration_mm,
            amplitude_contrast=amplitude_contrast,
            pixel_spacing_angstroms=pixel_spacing_angstroms,
        )
        return Defocus1DResults(
            cross_correlation_final=cc_final,
            frequencies_1d=fftfreq_to_spatial_frequency(
                bg_result.freqs, pixel_spacing_angstroms
            ),
            powerspectrum_1d=bg_result.rotationally_averaged_power_spectrum,
            background_model=bg_result.background_model,
            test_defoci=None,
            cross_correlations=None,
            ctf_model=CTF(
                defocus_um=refined_defocus,
                voltage_kev=torch.as_tensor(voltage_kev, dtype=torch.float32),
                spherical_aberration_mm=torch.as_tensor(
                    spherical_aberration_mm, dtype=torch.float32
                ),
                amplitude_contrast_fraction=torch.as_tensor(
                    amplitude_contrast, dtype=torch.float32
                ),
                phase_shift_degrees=torch.as_tensor(phase_deg, dtype=torch.float32),
                envelope_B=None,
            ),
            low_frequency_fit=1 / low_ang,
            high_frequency_fit=1 / high_ang,
            envelope_B=None,
            test_B_values=None,
            cross_correlations_2d=None,
        )

    # -------------------------------------------------------------------------
    # Step 2: Grid search — defocus (and optional B, phase shift) to maximise ZNCC
    # -------------------------------------------------------------------------
    grid_result = grid_search_defocus_and_envelope_1d(
        raps_in_fit_range=bg_result.raps_in_fit_range,
        spatial_freqs=bg_result.spatial_freqs,
        fit_mask=bg_result.fit_mask,
        image_sidelength=image_sidelength,
        defocus_range_microns=defocus_range_microns,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        optimize_envelope=optimize_envelope,
        b_range=b_range,
        b_step=b_step,
        optimize_phase_shift=optimize_phase_shift,
        phase_shift_step=phase_shift_step,
        phase_shift_range=phase_shift_range,
        fixed_phase_shift_deg=initial_phase_shift if not optimize_phase_shift else 0.0,
    )

    # -------------------------------------------------------------------------
    # Step 3: Refinement — gradient descent from grid (unless refine_steps<=0)
    # -------------------------------------------------------------------------
    if refine_steps > 0:
        initial_B_float: Optional[float] = None
        if grid_result.best_B is not None:
            initial_B_float = float(grid_result.best_B.detach().cpu().item())
        initial_phase_float: Optional[float] = None
        if grid_result.best_phase_shift is not None:
            initial_phase_float = float(
                grid_result.best_phase_shift.detach().cpu().item()
            )
        refined_defocus, refined_B, refined_phase_shift = (
            refine_defocus_and_b_factor_1d(
                initial_defocus=float(grid_result.best_defocus.detach().cpu().item()),
                initial_B=initial_B_float,
                raps_in_fit_range=bg_result.raps_in_fit_range,
                spatial_freqs=bg_result.spatial_freqs,
                fit_mask=bg_result.fit_mask,
                image_sidelength=image_sidelength,
                voltage_kev=voltage_kev,
                spherical_aberration_mm=spherical_aberration_mm,
                amplitude_contrast=amplitude_contrast,
                pixel_spacing_angstroms=pixel_spacing_angstroms,
                defocus_range_microns=defocus_range_microns,
                optimize_envelope=optimize_envelope,
                n_iterations=refine_steps,
                defocus_lr=refine_defocus_lr,
                b_factor_lr=refine_b_factor_lr,
                initial_phase_shift=initial_phase_float,
                optimize_phase_shift=optimize_phase_shift,
                phase_shift_lr=phase_shift_lr,
                phase_shift_range=phase_shift_range,
                early_stopper=early_stopper,
            )
        )
    else:
        refined_defocus = grid_result.best_defocus
        refined_B = grid_result.best_B
        refined_phase_shift = grid_result.best_phase_shift

    # -------------------------------------------------------------------------
    # Step 4: Build result — phase to [0, 90], assemble Defocus1DResults
    # -------------------------------------------------------------------------
    if refined_phase_shift is None:
        phase_deg = initial_phase_shift
    elif isinstance(refined_phase_shift, torch.Tensor):
        phase_deg = float(refined_phase_shift.cpu().item())
    else:
        phase_deg = float(refined_phase_shift)
    # Fold to [0, 90]: symmetry theta <-> 180 - theta
    phase_deg = min(phase_deg, 180.0 - phase_deg)
    b_for_cc: float | None = (
        None if refined_B is None else float(refined_B.detach().cpu().item())
    )
    cc_final = compute_final_1d_l2_cross_correlation(
        bg_result.raps_in_fit_range,
        bg_result.spatial_freqs,
        bg_result.fit_mask,
        image_sidelength,
        float(refined_defocus.detach().cpu().item()),
        envelope_B=b_for_cc,
        phase_shift_deg=phase_deg,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
    )
    return Defocus1DResults(
        cross_correlation_final=cc_final,
        frequencies_1d=fftfreq_to_spatial_frequency(
            bg_result.freqs, pixel_spacing_angstroms
        ),
        powerspectrum_1d=bg_result.rotationally_averaged_power_spectrum,
        background_model=bg_result.background_model,
        test_defoci=grid_result.test_defoci,
        cross_correlations=grid_result.cross_correlations_1d,
        ctf_model=CTF(
            defocus_um=refined_defocus,
            voltage_kev=torch.as_tensor(voltage_kev, dtype=torch.float32),
            spherical_aberration_mm=torch.as_tensor(
                spherical_aberration_mm, dtype=torch.float32
            ),
            amplitude_contrast_fraction=torch.as_tensor(
                amplitude_contrast, dtype=torch.float32
            ),
            phase_shift_degrees=torch.as_tensor(phase_deg, dtype=torch.float32),
            envelope_B=None
            if refined_B is None
            else torch.as_tensor(float(refined_B.cpu().item()), dtype=torch.float32),
        ),
        low_frequency_fit=1 / low_ang,
        high_frequency_fit=1 / high_ang,
        envelope_B=None
        if refined_B is None
        else torch.as_tensor(float(refined_B.cpu().item()), dtype=torch.float32),
        test_B_values=grid_result.test_B_values,
        cross_correlations_2d=grid_result.cross_correlations_2d,
    )
