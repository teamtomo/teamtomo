"""Convenience 1D-then-2D CTF estimation from a real-space image.

This function is a default **defocus-only** composition of public primitives
for notebooks and tests. It does not write files.

Chaining choices (PICASSO is the user-facing chainer):

- Always fit 1D defocus on the mean spectrum first, then 2D (grid or linear).
- ``use_1d_defocus_for_spatial``: build the 2D field from per-patch 1D fits
  (useful for linear tilt or a spatial grid) instead of one 2D optimisation.
- ``patch_sidelength < 0``: whole-image mode (requires ``nh=nw=1``).
- ``mask_laser_axis`` + ``LaserParams``: zero FFT strips along the laser axis.
- ``linear_fix_defocus_0_from_1x1``: seed linear ``defocus_0`` from a 1x1 2D fit.
- Thickness is **not** run here. PICASSO can add: 1D thickness grid search,
  then joint defocus+thickness refine (1D or 2D).
"""

from typing import Any, Optional

import torch

from torch_ctf_estimation.estimate_ctf_1d import (
    estimate_ctf_1d,
    fit_background_spline_1d,
)
from torch_ctf_estimation.estimate_ctf_2d import (
    estimate_ctf_2d,
    estimate_defocus_2d_at_1x1,
)
from torch_ctf_estimation.estimate_ctf_2d.estimate_background_2d import (
    estimate_background_2d,
)
from torch_ctf_estimation.models import (
    CTFFittingParams,
    Defocus1DResults,
    Defocus2DResults,
    LaserParams,
    OpticalParams,
    linear_tilt_axis_and_magnitude_deg,
)
from torch_ctf_estimation.utils.defocus_field_from_1d import defocus_field_from_1d_fits
from torch_ctf_estimation.utils.fitting_bounds import (
    resolve_defocus_bounds,
    resolve_phase_shift_fitting,
)
from torch_ctf_estimation.utils.laser_axis_mask import apply_laser_axis_mask
from torch_ctf_estimation.utils.patches import (
    compute_patch_power_spectra,
    extract_ctf_patches,
    normalised_patch_positions,
)
from torch_ctf_estimation.utils.prepare_image import prepare_image_for_ctf


def _scalar(value: Any) -> float:
    """Convert a tensor or number to float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def estimate_ctf(
    image: torch.Tensor,  # (t, h, w) or (h, w)
    optical_params: OpticalParams,
    fitting_params: CTFFittingParams,
    laser_params: LaserParams | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, Defocus1DResults, Defocus2DResults]:
    """
    Estimate CTF from a 2D or 3D image (defocus-only convenience composer).

    Parameters
    ----------
    image : torch.Tensor
        (t, h, w) or (h, w) array containing 2D or 3D image data.
    optical_params : OpticalParams
        Pixel spacing, voltage, Cs, amplitude contrast, optional rescale target.
    fitting_params : CTFFittingParams
        Defocus grid resolution, frequency range, patch size, and fitting options.
    laser_params : LaserParams | None, optional
        If set, ``laser_xy_angle_deg`` and ``dual_laser`` can drive laser-axis
        masking when ``mask_laser_axis`` is enabled. Use ``model_laser=True`` to
        fit with the LPP CTF model; if None, use standard CTF with no masking.
    device : torch.device | None, optional
        Device for computation. If None, uses cuda:0 when available, else cpu.

    Returns
    -------
    mean_ps : torch.Tensor
        Mean power spectrum of the patches.
    result1d : Defocus1DResults
        Results from 1D defocus estimation.
    result2d : Defocus2DResults
        Results from 2D defocus estimation.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image, new_spacing = prepare_image_for_ctf(
        image,
        pixel_spacing_angstroms=optical_params.pixel_spacing_angstroms,
        target_pixel_spacing_angstroms=optical_params.target_pixel_spacing_angstroms,
        device=device,
    )
    t, h, w = image.shape

    patches, patch_centers, image_sidelength_for_1d, use_whole_image = (
        extract_ctf_patches(
            image,
            patch_sidelength=fitting_params.patch_sidelength,
            defocus_grid_resolution=fitting_params.defocus_grid_resolution,
        )
    )
    patch_ps, mean_ps = compute_patch_power_spectra(patches)

    axis_mask: Optional[torch.Tensor] = None
    if fitting_params.mask_laser_axis and laser_params is not None:
        patch_ps, mean_ps, axis_mask = apply_laser_axis_mask(
            patch_ps,
            mean_ps,
            laser_xy_angle_deg=laser_params.laser_xy_angle_deg,
            dual_laser=laser_params.dual_laser,
            mask_width=fitting_params.laser_axis_mask_width,
        )

    nt, nh, nw = fitting_params.defocus_grid_resolution
    use_1d_spatial = fitting_params.use_1d_defocus_for_spatial and (
        fitting_params.defocus_model == "linear" or (nh > 1 or nw > 1)
    )
    bg_mean: Optional[Any] = None
    if use_1d_spatial:
        bg_mean = fit_background_spline_1d(
            power_spectrum=mean_ps,
            image_sidelength=image_sidelength_for_1d,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            pixel_spacing_angstroms=new_spacing,
        )

    defocus_bounds = resolve_defocus_bounds(fitting_params.defocus_range_microns)
    optimize_phase_1d, phase_shift_deg, phase_bounds = resolve_phase_shift_fitting(
        optimize_phase_shift=fitting_params.optimize_phase_shift,
        phase_shift_range_degrees=fitting_params.phase_shift_range_degrees,
        initial_phase_shift=fitting_params.initial_phase_shift,
    )
    result1d = estimate_ctf_1d(
        power_spectrum=mean_ps,
        image_sidelength=image_sidelength_for_1d,
        frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
        defocus_range_microns=defocus_bounds,
        voltage_kev=optical_params.voltage_kev,
        spherical_aberration_mm=optical_params.spherical_aberration_mm,
        amplitude_contrast=optical_params.amplitude_contrast_fraction,
        pixel_spacing_angstroms=new_spacing,
        optimize_envelope=fitting_params.optimize_envelope_1d,
        b_range=fitting_params.b_range_1d,
        b_step=fitting_params.b_step_1d,
        refine_steps=fitting_params.refine_steps_1d,
        background_result=bg_mean,
        optimize_phase_shift=optimize_phase_1d,
        initial_phase_shift=phase_shift_deg,
        phase_shift_range=phase_bounds,
        early_stopper=fitting_params.build_early_stopper(),
    )

    image_shape_2d = (
        (h, w)
        if use_whole_image
        else (image_sidelength_for_1d, image_sidelength_for_1d)
    )
    if fitting_params.use_amplitude_2d:
        patch_ps = torch.sqrt(patch_ps.clamp(min=0.0))
        mean_ps_2d = torch.sqrt(mean_ps.clamp(min=0.0))
    else:
        mean_ps_2d = mean_ps
    background_2d = estimate_background_2d(
        power_spectrum=mean_ps_2d,
        image_sidelength=image_shape_2d,
    )
    patch_ps = patch_ps - background_2d
    positions = normalised_patch_positions(patch_centers, (t, h, w))

    initial_envelope_B_2d = fitting_params.initial_envelope_B
    if initial_envelope_B_2d is None and result1d.ctf_model.envelope_B is not None:
        initial_envelope_B_2d = _scalar(result1d.ctf_model.envelope_B)
    if initial_envelope_B_2d is None:
        initial_envelope_B_2d = 0.0
    initial_defocus_2d = _scalar(result1d.ctf_model.defocus_um)
    initial_phase_shift_2d = phase_shift_deg
    if optimize_phase_1d and result1d.ctf_model.phase_shift_degrees is not None:
        initial_phase_shift_2d = _scalar(result1d.ctf_model.phase_shift_degrees)

    if use_1d_spatial:
        result_1x1 = estimate_defocus_2d_at_1x1(
            patch_power_spectra=patch_ps,
            defocus_grid_resolution=fitting_params.defocus_grid_resolution,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            initial_defocus=initial_defocus_2d,
            pixel_spacing_angstroms=new_spacing,
            optimize_astigmatism=fitting_params.optimize_astigmatism,
            initial_envelope_B=initial_envelope_B_2d,
            n_iterations=fitting_params.n_iterations_2d,
            debug=fitting_params.debug,
            optimize_phase_shift=optimize_phase_1d,
            phase_shift_model=fitting_params.phase_shift_model,
            phase_shift_quadratic_perpendicular_axis=(
                fitting_params.phase_shift_quadratic_perpendicular_axis
            ),
            initial_phase_shift=initial_phase_shift_2d,
            fixed_phase_shift_deg=phase_shift_deg if not optimize_phase_1d else None,
            voltage_kev=optical_params.voltage_kev,
            spherical_aberration_mm=optical_params.spherical_aberration_mm,
            amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
            laser_params=laser_params,
            axis_mask=axis_mask,
            defocus_bounds_microns=defocus_bounds,
            phase_shift_bounds_degrees=phase_bounds,
            early_stopper=fitting_params.build_early_stopper(),
            use_amplitude=fitting_params.use_amplitude_2d,
        )
        result2d = defocus_field_from_1d_fits(
            patch_power_spectra=patch_ps,
            normalised_patch_positions=positions,
            result_1x1=result_1x1,
            defocus_model=fitting_params.defocus_model,
            defocus_grid_resolution=fitting_params.defocus_grid_resolution,
            initial_defocus=initial_defocus_2d,
            image_sidelength=image_sidelength_for_1d,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            defocus_range_microns=defocus_bounds,
            phase_shift_range_degrees=phase_bounds,
            voltage_kev=optical_params.voltage_kev,
            spherical_aberration_mm=optical_params.spherical_aberration_mm,
            amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
            pixel_spacing_angstroms=new_spacing,
            optimize_envelope_1d=fitting_params.optimize_envelope_1d,
            b_range_1d=fitting_params.b_range_1d,
            b_step_1d=fitting_params.b_step_1d,
            refine_steps_1d=fitting_params.refine_steps_1d,
            background_result=bg_mean,
            device=patch_ps.device,
            optimize_phase_shift=optimize_phase_1d,
            use_equiphase_for_1d_spatial=fitting_params.use_equiphase_for_1d_spatial,
            laser_params=laser_params,
            equiphase_n_theta=fitting_params.equiphase_n_theta,
            fixed_phase_shift_deg=phase_shift_deg if not optimize_phase_1d else None,
        )
        if result2d.defocus_model_type == "linear":
            axis_deg, tilt_deg = linear_tilt_axis_and_magnitude_deg(
                result2d, new_spacing, min(h, w)
            )
            result2d = result2d.model_copy(
                update={
                    "tilt_axis_angle_deg": axis_deg,
                    "tilt_magnitude_deg": tilt_deg,
                }
            )
        if optimize_phase_1d and result_1x1.phase_shift_degrees is not None:
            result2d = result2d.model_copy(
                update={
                    "phase_shift_degrees": result_1x1.phase_shift_degrees,
                    "phase_shift_model_type": result_1x1.phase_shift_model_type,
                    "phase_shift_model": result_1x1.phase_shift_model,
                    "phase_shift_trace": result_1x1.phase_shift_trace,
                }
            )
        elif not optimize_phase_1d:
            result2d = result2d.model_copy(
                update={"phase_shift_degrees": phase_shift_deg}
            )
        return mean_ps, result1d, result2d

    fix_defocus_0_val = None
    initial_astigmatism_2d = 0.0
    initial_astigmatism_angle_2d = 0.0
    initial_phase_shift_for_2d = initial_phase_shift_2d
    if (
        fitting_params.defocus_model == "linear"
        and fitting_params.linear_fix_defocus_0_from_1x1
    ):
        result_1x1 = estimate_defocus_2d_at_1x1(
            patch_power_spectra=patch_ps,
            defocus_grid_resolution=fitting_params.defocus_grid_resolution,
            frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
            initial_defocus=initial_defocus_2d,
            pixel_spacing_angstroms=new_spacing,
            optimize_astigmatism=fitting_params.optimize_astigmatism,
            initial_envelope_B=initial_envelope_B_2d,
            n_iterations=fitting_params.n_iterations_2d,
            debug=fitting_params.debug,
            optimize_phase_shift=optimize_phase_1d,
            phase_shift_model=fitting_params.phase_shift_model,
            phase_shift_quadratic_perpendicular_axis=(
                fitting_params.phase_shift_quadratic_perpendicular_axis
            ),
            initial_phase_shift=initial_phase_shift_2d,
            fixed_phase_shift_deg=phase_shift_deg if not optimize_phase_1d else None,
            voltage_kev=optical_params.voltage_kev,
            spherical_aberration_mm=optical_params.spherical_aberration_mm,
            amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
            laser_params=laser_params,
            axis_mask=axis_mask,
            defocus_bounds_microns=defocus_bounds,
            phase_shift_bounds_degrees=phase_bounds,
            early_stopper=fitting_params.build_early_stopper(),
            use_amplitude=fitting_params.use_amplitude_2d,
        )
        fix_defocus_0_val = float(result_1x1.defocus_model.data.mean().cpu().item())
        if result_1x1.astigmatism is not None:
            initial_astigmatism_2d = result_1x1.astigmatism
        if result_1x1.astigmatism_angle is not None:
            initial_astigmatism_angle_2d = result_1x1.astigmatism_angle
        if optimize_phase_1d and result_1x1.phase_shift_degrees is not None:
            initial_phase_shift_for_2d = result_1x1.phase_shift_degrees

    result2d = estimate_ctf_2d(
        patch_power_spectra=patch_ps,
        normalised_patch_positions=positions,
        defocus_grid_resolution=fitting_params.defocus_grid_resolution,
        frequency_fit_range_angstroms=fitting_params.frequency_fit_range_angstroms,
        initial_defocus=initial_defocus_2d,
        pixel_spacing_angstroms=new_spacing,
        debug=fitting_params.debug,
        optimize_astigmatism=fitting_params.optimize_astigmatism,
        defocus_model=fitting_params.defocus_model,
        initial_envelope_B=initial_envelope_B_2d,
        initial_astigmatism=initial_astigmatism_2d,
        initial_astigmatism_angle=initial_astigmatism_angle_2d,
        fix_defocus_0=fix_defocus_0_val,
        n_iterations=fitting_params.n_iterations_2d,
        optimize_phase_shift=optimize_phase_1d,
        phase_shift_model=fitting_params.phase_shift_model,
        phase_shift_quadratic_perpendicular_axis=(
            fitting_params.phase_shift_quadratic_perpendicular_axis
        ),
        initial_phase_shift=initial_phase_shift_for_2d,
        fixed_phase_shift_deg=phase_shift_deg if not optimize_phase_1d else None,
        voltage_kev=optical_params.voltage_kev,
        spherical_aberration_mm=optical_params.spherical_aberration_mm,
        amplitude_contrast_fraction=optical_params.amplitude_contrast_fraction,
        laser_params=laser_params,
        axis_mask=axis_mask,
        defocus_bounds_microns=defocus_bounds,
        phase_shift_bounds_degrees=phase_bounds,
        early_stopper=fitting_params.build_early_stopper(),
        use_amplitude=fitting_params.use_amplitude_2d,
    )
    if result2d.defocus_model_type == "linear":
        axis_deg, tilt_deg = linear_tilt_axis_and_magnitude_deg(
            result2d, new_spacing, min(h, w)
        )
        result2d = result2d.model_copy(
            update={
                "tilt_axis_angle_deg": axis_deg,
                "tilt_magnitude_deg": tilt_deg,
            }
        )
    if not optimize_phase_1d:
        result2d = result2d.model_copy(update={"phase_shift_degrees": phase_shift_deg})
    return mean_ps, result1d, result2d
