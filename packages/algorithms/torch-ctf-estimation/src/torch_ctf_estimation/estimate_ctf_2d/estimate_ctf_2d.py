"""Estimate CTF in 2D from a power spectrum."""

from collections.abc import Callable
from typing import Literal, Optional

import torch

from torch_ctf_estimation.estimate_ctf_2d.estimate_defocus_2d_grid import (
    estimate_defocus_2d_grid,
)
from torch_ctf_estimation.estimate_ctf_2d.estimate_defocus_2d_linear import (
    estimate_defocus_2d_linear,
)
from torch_ctf_estimation.models import Defocus2DResults, LaserParams


def estimate_ctf_2d(
    patch_power_spectra: torch.Tensor,
    normalised_patch_positions: torch.Tensor,
    defocus_grid_resolution: tuple[int, int, int],
    frequency_fit_range_angstroms: tuple[float, float],
    initial_defocus: float,
    pixel_spacing_angstroms: float,
    initial_astigmatism: float = 0.0,
    initial_astigmatism_angle: float = 0.0,
    optimize_astigmatism: bool = False,
    initial_envelope_B: float = 0.0,
    n_iterations: int = 100,
    defocus_lr: float = 0.01,
    astigmatism_lr: float = 0.05,
    astigmatism_angle_lr: float = 50.0,
    defocus_model: Literal["grid", "linear"] = "grid",
    initial_defocus_gradient_magnitude: float = 0.0,
    initial_defocus_gradient_angle: float = 0.0,
    defocus_gradient_magnitude_lr: float = 0.05,
    defocus_gradient_angle_lr: float = 50.0,
    fix_defocus_0: Optional[float] = None,
    debug: bool = False,
    optimize_phase_shift: bool = False,
    phase_shift_model: Literal["grid", "quadratic"] = "grid",
    phase_shift_quadratic_perpendicular_axis: bool = False,
    initial_phase_shift: float = 0.0,
    phase_shift_lr: float = 5.0,
    voltage_kev: float = 300.0,
    spherical_aberration_mm: float = 2.7,
    amplitude_contrast_fraction: float = 0.07,
    laser_params: Optional[LaserParams] = None,
    axis_mask: Optional[torch.Tensor] = None,
    defocus_bounds_microns: tuple[float, float] | None = None,
    phase_shift_bounds_degrees: tuple[float, float] | None = None,
    fixed_phase_shift_deg: float | None = None,
    early_stopper: Callable[[float], bool] | None = None,
    use_amplitude: bool = False,
) -> Defocus2DResults:
    """
    Estimate CTF in 2D from a power spectrum.

    Optimizes a 2D+t defocus model (grid or linear tilt) and optionally astigmatism
    by maximising the correlation between simulated CTF² and patch power spectra,
    looping over the time/frame dimension with gradient accumulation.

    Parameters
    ----------
    patch_power_spectra : torch.Tensor
        Patch power spectra, shape ``(t, gh, gw, ph, pw)`` (frames, patch grid, freq).
    normalised_patch_positions : torch.Tensor
        Normalised patch positions, shape ``(t, gh, gw, 3)`` in [0, 1].
    defocus_grid_resolution : tuple[int, int, int]
        Resolution ``(nt, nh, nw)``. For grid model all three are used; for linear
        only ``nt`` is used (time knots for cubic spline when t>1).
    frequency_fit_range_angstroms : tuple[float, float]
        ``(low, high)`` frequency fit range in angstroms.
    initial_defocus : float
        Initial defocus in microns.
    pixel_spacing_angstroms : float
        Isotropic pixel spacing in angstroms.
    initial_astigmatism : float, optional
        Initial astigmatism in microns. Default 0.0.
    initial_astigmatism_angle : float, optional
        Initial astigmatism angle in degrees. Default 0.0.
    optimize_astigmatism : bool, optional
        Whether to optimize astigmatism and angle. Default False.
    initial_envelope_B : float, optional
        Initial B-factor for envelope. Default 0.0.
    n_iterations : int, optional
        Number of optimizer steps. Default 100.
    defocus_lr : float, optional
        Learning rate for the defocus (grid or base defocus_0). Default 0.01.
    astigmatism_lr : float, optional
        Learning rate for the astigmatism magnitude (when ``optimize_astigmatism``).
        Default 0.05.
    astigmatism_angle_lr : float, optional
        Learning rate for the astigmatism angle parameters (when
        ``optimize_astigmatism``). Default 50.0.
    defocus_model : {"grid", "linear"}, optional
        Defocus model: "grid" (3D spline) or "linear" (tilt). Default "grid".
    initial_defocus_gradient_magnitude : float, optional
        Initial defocus gradient magnitude for linear model. Default 0.0.
    initial_defocus_gradient_angle : float, optional
        Initial defocus gradient angle in degrees for linear model. Default 0.0.
    defocus_gradient_magnitude_lr : float, optional
        Learning rate for defocus gradient magnitude (linear). Default 0.05.
    defocus_gradient_angle_lr : float, optional
        Learning rate for defocus gradient angle (linear). Default 50.0.
    fix_defocus_0 : float, optional
        If set (e.g. from 2D fit at 1x1), fix defocus_0 and only optimize
        gradient magnitude and angle in the linear model. Default None.
    debug : bool, optional
        If True, return extra fields (traces, simulated CTF², patch spectra).
        Default False.
    optimize_phase_shift : bool, optional
        Whether to estimate phase shift (0-180 deg) alongside defocus. Default False.
    phase_shift_model : {"grid", "quadratic"}, optional
        "grid" (per-patch) or "quadratic" (directional). Default "grid".
    phase_shift_quadratic_perpendicular_axis : bool, optional
        If True and ``phase_shift_model`` is "quadratic", fit g2, k2 along t
        perpendicular to s. Default False.
    initial_phase_shift : float, optional
        Initial phase shift in degrees when optimizing. Default 0.0.
    phase_shift_lr : float, optional
        Learning rate for phase shift parameters. Default 5.0.
    voltage_kev : float, optional
        Acceleration voltage in keV for CTF simulation. Default 300.0.
    spherical_aberration_mm : float, optional
        Spherical aberration in mm for CTF simulation. Default 2.7.
    amplitude_contrast_fraction : float, optional
        Amplitude contrast fraction (0-1) for CTF simulation. Default 0.07.
    laser_params : Optional[LaserParams], optional
        If set and ``model_laser`` is True, use LPP (laser phase plate) CTF
        model; otherwise use standard calculate_ctf_2d. Laser geometry is still
        used for axis masking when ``mask_laser_axis`` is enabled. Default None.
    axis_mask : Optional[torch.Tensor], optional
        2D rFFT-layout mask (shape ``(ph, pw // 2 + 1)``, values 0/1) that
        zeros strips along the laser axis.  Folded into ``bp_filter`` so both
        data and simulated spectra are masked identically.  Default None
        (no masking).
    defocus_bounds_microns : tuple[float, float] or None, optional
        (low, high) defocus bounds in microns for 2D fitting. Default None
        uses (0, 10) µm.
    phase_shift_bounds_degrees : tuple[float, float] or None, optional
        (low, high) phase shift bounds in degrees for 2D fitting. Default None
        uses (0, 180)°. Equal values fix phase at that value.
    fixed_phase_shift_deg : float or None, optional
        Known phase shift in degrees when ``optimize_phase_shift`` is False.
        Default None (phase treated as zero unless set via equal bounds).
    early_stopper : callable or None, optional
        Stateful ``(loss) -> should_stop`` callback. Default None (run all
        ``n_iterations``).

    Returns
    -------
    Defocus2DResults
        Defocus model, astigmatism, astigmatism angle, envelope B, and optional traces.
    """
    if defocus_model == "grid":
        return estimate_defocus_2d_grid(
            patch_power_spectra=patch_power_spectra,
            normalised_patch_positions=normalised_patch_positions,
            defocus_grid_resolution=defocus_grid_resolution,
            frequency_fit_range_angstroms=frequency_fit_range_angstroms,
            initial_defocus=initial_defocus,
            pixel_spacing_angstroms=pixel_spacing_angstroms,
            initial_astigmatism=initial_astigmatism,
            initial_astigmatism_angle=initial_astigmatism_angle,
            optimize_astigmatism=optimize_astigmatism,
            initial_envelope_B=initial_envelope_B,
            n_iterations=n_iterations,
            defocus_lr=defocus_lr,
            astigmatism_lr=astigmatism_lr,
            astigmatism_angle_lr=astigmatism_angle_lr,
            debug=debug,
            optimize_phase_shift=optimize_phase_shift,
            phase_shift_model=phase_shift_model,
            phase_shift_quadratic_perpendicular_axis=phase_shift_quadratic_perpendicular_axis,
            initial_phase_shift=initial_phase_shift,
            phase_shift_lr=phase_shift_lr,
            voltage_kev=voltage_kev,
            spherical_aberration_mm=spherical_aberration_mm,
            amplitude_contrast_fraction=amplitude_contrast_fraction,
            laser_params=laser_params,
            axis_mask=axis_mask,
            defocus_bounds_microns=defocus_bounds_microns,
            phase_shift_bounds_degrees=phase_shift_bounds_degrees,
            fixed_phase_shift_deg=fixed_phase_shift_deg,
            early_stopper=early_stopper,
            use_amplitude=use_amplitude,
        )
    return estimate_defocus_2d_linear(
        patch_power_spectra=patch_power_spectra,
        normalised_patch_positions=normalised_patch_positions,
        defocus_grid_resolution=defocus_grid_resolution,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        initial_defocus=initial_defocus,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        initial_astigmatism=initial_astigmatism,
        initial_astigmatism_angle=initial_astigmatism_angle,
        optimize_astigmatism=optimize_astigmatism,
        initial_envelope_B=initial_envelope_B,
        n_iterations=n_iterations,
        defocus_lr=defocus_lr,
        astigmatism_lr=astigmatism_lr,
        astigmatism_angle_lr=astigmatism_angle_lr,
        initial_defocus_gradient_magnitude=initial_defocus_gradient_magnitude,
        initial_defocus_gradient_angle=initial_defocus_gradient_angle,
        defocus_gradient_magnitude_lr=defocus_gradient_magnitude_lr,
        defocus_gradient_angle_lr=defocus_gradient_angle_lr,
        fix_defocus_0=fix_defocus_0,
        debug=debug,
        optimize_phase_shift=optimize_phase_shift,
        phase_shift_model=phase_shift_model,
        phase_shift_quadratic_perpendicular_axis=phase_shift_quadratic_perpendicular_axis,
        initial_phase_shift=initial_phase_shift,
        phase_shift_lr=phase_shift_lr,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast_fraction=amplitude_contrast_fraction,
        laser_params=laser_params,
        axis_mask=axis_mask,
        defocus_bounds_microns=defocus_bounds_microns,
        phase_shift_bounds_degrees=phase_shift_bounds_degrees,
        fixed_phase_shift_deg=fixed_phase_shift_deg,
        early_stopper=early_stopper,
        use_amplitude=use_amplitude,
    )
