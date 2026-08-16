"""Joint 1D gradient refinement of defocus and sample thickness."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
from torch_ctf.ctf_thickness import calculate_ctf_thickness_1d
from torch_grid_utils.fftfreq_grid import fftfreq_to_spatial_frequency

from torch_ctf_estimation.estimate_ctf_1d.estimate_ctf_1d_utils import (
    get_background_result,
)
from torch_ctf_estimation.metrics.fit_metrics import l2_normalized_cross_correlation
from torch_ctf_estimation.models import CTF, Defocus1DResults, LaserParams
from torch_ctf_estimation.models.results_models import (
    Thickness1DResults,
    _Background1DResult,
)
from torch_ctf_estimation.utils.fitting_bounds import resolve_defocus_bounds


def refine_defocus_and_thickness_1d(
    power_spectrum: torch.Tensor,
    image_sidelength: int,
    frequency_fit_range_angstroms: tuple[float, float],
    initial_defocus_um: float,
    initial_thickness_angstroms: float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    pixel_spacing_angstroms: float,
    phase_shift_deg: float = 0.0,
    n_iterations: int = 100,
    defocus_lr: float = 0.01,
    thickness_lr: float = 50.0,
    defocus_range_microns: tuple[float, float] | None = None,
    thickness_range_angstroms: tuple[float, float] = (300.0, 4000.0),
    background_result: Optional[_Background1DResult] = None,
    laser_params: Optional[LaserParams] = None,
    early_stopper: Callable[[float], bool] | None = None,
    use_equiphase: bool = False,
    equiphase_defocus_um: float | None = None,
    equiphase_astigmatism_um: float | None = None,
    equiphase_astigmatism_angle_deg: float | None = None,
    equiphase_phase_shift_deg: float | None = None,
    equiphase_n_theta: int = 64,
    optimize_defocus: bool = True,
) -> tuple[Defocus1DResults, Thickness1DResults]:
    """Refine 1D thickness (and optionally defocus) with the thickness CTF.

    Standalone 1D thickness remains a grid search. This is the 1D
    gradient-descent step. Set ``optimize_defocus=False`` to hold defocus
    at ``initial_defocus_um`` and update thickness only.

    Parameters
    ----------
    power_spectrum : torch.Tensor
        Mean (or single) rFFT power spectrum, shape ``(h, w_rfft)``.
    image_sidelength : int
        Real-space sidelength used for the 1D CTF.
    frequency_fit_range_angstroms : tuple[float, float]
        ``(low, high)`` fit band in Angstroms.
    initial_defocus_um : float
        Starting defocus in micrometers.
    initial_thickness_angstroms : float
        Starting thickness from the 1D grid search.
    voltage_kev, spherical_aberration_mm, amplitude_contrast :
        Microscope parameters.
    pixel_spacing_angstroms : float
        Pixel size used for the spectrum (after any rescale).
    phase_shift_deg : float, optional
        Fixed phase shift in degrees. Default 0.0.
    n_iterations : int, optional
        Adam steps. Default 100.
    defocus_lr, thickness_lr : float, optional
        Learning rates.
    defocus_range_microns : tuple[float, float] | None, optional
        Defocus clamp bounds.
    thickness_range_angstroms : tuple[float, float], optional
        Thickness clamp bounds. Default (300, 4000).
    background_result : optional
        Reuse a pre-fitted 1D background.
    laser_params : LaserParams | None, optional
        Unused for the 1D thickness CTF (kept for call-site consistency).
    early_stopper : callable or None, optional
        Stateful ``(loss) -> should_stop`` callback. Default None (run all
        ``n_iterations``).
    optimize_defocus : bool, optional
        If False, keep defocus at ``initial_defocus_um`` and refine thickness
        only. Default True.

    Returns
    -------
    result1d : Defocus1DResults
        Updated 1D defocus result.
    thickness1d : Thickness1DResults
        Updated 1D thickness result.
    """
    defocus_bounds = resolve_defocus_bounds(defocus_range_microns)
    t_lo, t_hi = thickness_range_angstroms
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
    raps = bg_result.raps_in_fit_range.detach()
    fit_mask = bg_result.fit_mask

    thickness_param = torch.nn.Parameter(
        torch.tensor(initial_thickness_angstroms, device=device, dtype=dtype)
    )
    if optimize_defocus:
        defocus_param: torch.Tensor = torch.nn.Parameter(
            torch.tensor(initial_defocus_um, device=device, dtype=dtype)
        )
        optimiser = torch.optim.Adam(
            [
                {"params": [defocus_param], "lr": defocus_lr},
                {"params": [thickness_param], "lr": thickness_lr},
            ]
        )
    else:
        defocus_param = torch.tensor(
            initial_defocus_um, device=device, dtype=dtype
        )
        optimiser = torch.optim.Adam(
            [{"params": [thickness_param], "lr": thickness_lr}]
        )

    for _ in range(n_iterations):
        optimiser.zero_grad()
        with torch.no_grad():
            if optimize_defocus:
                defocus_param.clamp_(min=defocus_bounds[0], max=defocus_bounds[1])
            thickness_param.clamp_(min=t_lo, max=t_hi)
        simulated = calculate_ctf_thickness_1d(
            return_power_spectrum=True,
            sample_thickness_angstrom=thickness_param,
            defocus=defocus_param,
            voltage=voltage_kev,
            spherical_aberration=spherical_aberration_mm,
            amplitude_contrast=amplitude_contrast,
            phase_shift=phase_shift_deg,
            pixel_size=pixel_spacing_angstroms,
            n_samples=image_sidelength // 2 + 1,
            oversampling_factor=3,
        )
        if simulated.ndim > 1:
            simulated = simulated.reshape(-1)
        fit_mask_d = fit_mask.to(simulated.device)
        raps_d = raps.to(simulated.device)
        model = simulated[fit_mask_d]
        ny = torch.linalg.norm(raps_d)
        nm = torch.linalg.norm(model)
        if ny < 1e-12 or nm < 1e-12:
            continue
        loss = -(torch.dot(raps_d, model) / (ny * nm))
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        optimiser.step()
        if early_stopper is not None and early_stopper(
            float(loss.detach().cpu().item())
        ):
            break

    with torch.no_grad():
        if optimize_defocus:
            defocus_param.clamp_(min=defocus_bounds[0], max=defocus_bounds[1])
        thickness_param.clamp_(min=t_lo, max=t_hi)

    final_defocus = float(defocus_param.detach().cpu().item())
    final_thickness = float(thickness_param.detach().cpu().item())
    with torch.no_grad():
        simulated = calculate_ctf_thickness_1d(
            return_power_spectrum=True,
            sample_thickness_angstrom=final_thickness,
            defocus=final_defocus,
            voltage=voltage_kev,
            spherical_aberration=spherical_aberration_mm,
            amplitude_contrast=amplitude_contrast,
            phase_shift=phase_shift_deg,
            pixel_size=pixel_spacing_angstroms,
            n_samples=image_sidelength // 2 + 1,
            oversampling_factor=3,
        )
        if simulated.ndim > 1:
            simulated = simulated.reshape(-1)
        fit_mask_d = fit_mask.to(simulated.device)
        raps_d = raps.to(simulated.device)
        cc_final = l2_normalized_cross_correlation(raps_d, simulated[fit_mask_d])

    freqs = fftfreq_to_spatial_frequency(bg_result.freqs, pixel_spacing_angstroms)
    result1d = Defocus1DResults(
        cross_correlation_final=cc_final,
        frequencies_1d=freqs,
        powerspectrum_1d=bg_result.rotationally_averaged_power_spectrum,
        background_model=bg_result.background_model,
        ctf_model=CTF(
            defocus_um=torch.as_tensor(final_defocus, dtype=torch.float32),
            voltage_kev=torch.as_tensor(voltage_kev, dtype=torch.float32),
            spherical_aberration_mm=torch.as_tensor(
                spherical_aberration_mm, dtype=torch.float32
            ),
            amplitude_contrast_fraction=torch.as_tensor(
                amplitude_contrast, dtype=torch.float32
            ),
            phase_shift_degrees=torch.as_tensor(phase_shift_deg, dtype=torch.float32),
            envelope_B=None,
        ),
        low_frequency_fit=1.0 / low_ang,
        high_frequency_fit=1.0 / high_ang,
    )
    thickness1d = Thickness1DResults(
        thickness_angstroms=final_thickness,
        cross_correlation_final=cc_final,
        frequencies_1d=freqs,
        powerspectrum_1d=bg_result.rotationally_averaged_power_spectrum,
        background_model=bg_result.background_model,
        low_frequency_fit=1.0 / low_ang,
        high_frequency_fit=1.0 / high_ang,
    )
    return result1d, thickness1d
