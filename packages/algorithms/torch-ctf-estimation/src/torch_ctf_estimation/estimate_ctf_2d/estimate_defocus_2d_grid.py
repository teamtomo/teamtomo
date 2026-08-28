"""
2D defocus estimation using a 3D spline grid over (t, x, y).

Defocus is represented as a cubic spline over the normalised (t, x, y) patch grid.
Optimisation fits the grid control points plus optional astigmatism and phase shift.
"""

import math
from collections.abc import Callable
from typing import Literal, Optional

import einops
import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d

from torch_ctf_estimation.estimate_ctf_2d.ctf_loss_2d import (
    compute_ctf2_t,
    correlation_loss_t,
    mean_pearson_r_final_2d,
)
from torch_ctf_estimation.estimate_ctf_2d.estimate_ctf_2d_utils import (
    _astig_angle_to_m90_p90,
    _check_astig_grad_and_reset,
    _clamp_defocus_grid_after_step,
    _clamp_optional_bounds,
    _get_astig_clamped,
    _reset_astigmatism,
    _shared_astigmatism_and_env,
)
from torch_ctf_estimation.estimate_ctf_2d.phase_shift_2d import (
    PhaseShiftModels,
    build_phase_shift_result,
    clamp_phase_shift_after_step,
    init_phase_shift_models,
    phase_shift_at_positions,
    phase_shift_param_groups,
    phase_shift_trace_value,
)
from torch_ctf_estimation.models import Defocus2DResults, LaserParams
from torch_ctf_estimation.utils.fitting_bounds import (
    resolve_defocus_bounds,
    resolve_phase_shift_bounds,
)


def _setup_grid_spectra_and_shape(
    patch_power_spectra: torch.Tensor,
    defocus_grid_resolution: tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[int, int], torch.device, int]:
    """
    Derive image shape and optionally collapse time when nt==1.

    Returns (patch_power_spectra, image_shape, device, T).
    """
    ph, pw_rfft = patch_power_spectra.shape[-2], patch_power_spectra.shape[-1]
    image_shape = (ph, (pw_rfft - 1) * 2)
    device = patch_power_spectra.device
    nt = defocus_grid_resolution[0]
    if nt == 1:
        patch_power_spectra = einops.reduce(
            patch_power_spectra, "t ... -> 1 ...", reduction="mean"
        )
    T = patch_power_spectra.shape[0]
    return patch_power_spectra, image_shape, device, T


def _setup_grid_defocus_and_phase(
    defocus_grid_resolution: tuple[int, int, int],
    initial_defocus: float,
    device: torch.device,
    optimize_phase_shift: bool,
    phase_shift_model: Literal["grid", "quadratic"],
    initial_phase_shift: float,
    phase_shift_quadratic_perpendicular_axis: bool = False,
) -> tuple[CubicCatmullRomGrid3d, Optional[PhaseShiftModels]]:
    """
    Create the 3D defocus spline grid and optional phase shift models.

    Returns (defocus_model_obj, phase_models).
    """
    defocus_grid_data = (
        torch.ones(size=defocus_grid_resolution, device=device) * initial_defocus
    )
    defocus_model_obj = CubicCatmullRomGrid3d.from_grid_data(defocus_grid_data).to(
        device
    )
    phase_models = init_phase_shift_models(
        optimize_phase_shift=optimize_phase_shift,
        phase_shift_model=phase_shift_model,
        initial_phase_shift=initial_phase_shift,
        grid_resolution=defocus_grid_resolution,
        device=device,
        phase_shift_quadratic_perpendicular_axis=phase_shift_quadratic_perpendicular_axis,
    )
    return defocus_model_obj, phase_models


def _build_grid_param_groups(
    defocus_model_obj: CubicCatmullRomGrid3d,
    defocus_lr: float,
    optimize_astigmatism: bool,
    astigmatism: torch.Tensor,
    angle_u: torch.Tensor,
    angle_v: torch.Tensor,
    astigmatism_lr: float,
    astigmatism_angle_lr: float,
    phase_models: Optional[PhaseShiftModels],
    phase_shift_lr: float,
) -> list[dict]:
    """Build Adam param groups: defocus grid, then astigmatism, then phase."""
    param_groups = [{"params": defocus_model_obj.parameters(), "lr": defocus_lr}]
    if optimize_astigmatism:
        param_groups.extend(
            [
                {"params": [astigmatism], "lr": astigmatism_lr},
                {"params": [angle_u, angle_v], "lr": astigmatism_angle_lr},
            ]
        )
    param_groups.extend(phase_shift_param_groups(phase_models, phase_shift_lr))
    return param_groups


def _grid_final_scalars(
    defocus_model_obj: CubicCatmullRomGrid3d,
    astigmatism: torch.Tensor,
    angle_u: torch.Tensor,
    angle_v: torch.Tensor,
    envelope_B: torch.Tensor,
) -> tuple[float, float, float, float, float, float]:
    """
    Compute final scalar outputs from fitted grid and astigmatism.

    Returns (final_astigmatism, final_astigmatism_angle, final_envelope_B,
             mean_defocus, final_defocus_u, final_defocus_v).
    """
    final_astigmatism = float(astigmatism.detach().cpu().item())
    _fn = torch.sqrt(angle_u.detach() ** 2 + angle_v.detach() ** 2 + 1e-8)
    _fa_rad = torch.atan2(angle_v.detach() / _fn, angle_u.detach() / _fn)
    _fa_deg = float((_fa_rad.cpu().item() * (180.0 / math.pi) + 180.0) % 180.0)
    final_astigmatism_angle = _astig_angle_to_m90_p90(_fa_deg)
    final_envelope_B = float(envelope_B.detach().cpu().item())
    mean_defocus = float(defocus_model_obj.data.detach().cpu().mean().item())
    final_defocus_u = mean_defocus + final_astigmatism / 2.0
    final_defocus_v = mean_defocus - final_astigmatism / 2.0
    return (
        final_astigmatism,
        final_astigmatism_angle,
        final_envelope_B,
        mean_defocus,
        final_defocus_u,
        final_defocus_v,
    )


def estimate_defocus_2d_grid(
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
    debug: bool = False,
    optimize_phase_shift: bool = False,
    phase_shift_model: Literal["grid", "quadratic"] = "grid",
    phase_shift_quadratic_perpendicular_axis: bool = False,
    initial_phase_shift: float = 0.0,
    phase_shift_lr: float = 5.0,
    voltage_kev: float = 300.0,
    spherical_aberration_mm: float = 2.7,
    amplitude_contrast_fraction: float = 0.10,
    laser_params: Optional[LaserParams] = None,
    axis_mask: Optional[torch.Tensor] = None,
    defocus_bounds_microns: tuple[float, float] | None = None,
    phase_shift_bounds_degrees: tuple[float, float] | None = None,
    fixed_phase_shift_deg: float | None = None,
    early_stopper: Callable[[float], bool] | None = None,
    use_amplitude: bool = False,
) -> Defocus2DResults:
    """
    Estimate defocus in 2D using a 3D spline grid over (t, x, y).

    See :func:`estimate_ctf_2d` for parameter descriptions.
    """
    defocus_bounds_microns = resolve_defocus_bounds(defocus_bounds_microns)
    phase_shift_bounds_degrees = resolve_phase_shift_bounds(phase_shift_bounds_degrees)
    # --- Setup: spectra, image shape, defocus grid, phase models ---
    (
        patch_power_spectra,
        image_shape,
        device,
        T,
    ) = _setup_grid_spectra_and_shape(patch_power_spectra, defocus_grid_resolution)
    defocus_model_obj, phase_models = _setup_grid_defocus_and_phase(
        defocus_grid_resolution,
        initial_defocus,
        device,
        optimize_phase_shift,
        phase_shift_model,
        initial_phase_shift,
        phase_shift_quadratic_perpendicular_axis,
    )

    # --- Bandpass, astigmatism params, envelope (shared with linear) ---
    (
        bp_filter,
        astigmatism,
        angle_u,
        angle_v,
        _angle_u_init,
        _angle_v_init,
        envelope_B,
        env_2d,
    ) = _shared_astigmatism_and_env(
        image_shape=image_shape,
        device=device,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        initial_astigmatism=initial_astigmatism,
        initial_astigmatism_angle=initial_astigmatism_angle,
        optimize_astigmatism=optimize_astigmatism,
        initial_envelope_B=initial_envelope_B,
        axis_mask=axis_mask,
    )
    patch_power_spectra = patch_power_spectra * bp_filter

    # --- Optimiser and trace lists ---
    param_groups = _build_grid_param_groups(
        defocus_model_obj,
        defocus_lr,
        optimize_astigmatism,
        astigmatism,
        angle_u,
        angle_v,
        astigmatism_lr,
        astigmatism_angle_lr,
        phase_models,
        phase_shift_lr,
    )
    optimiser = torch.optim.Adam(params=param_groups)
    defocus_models: list[torch.Tensor] = []
    astigmatism_trace: list[float] = []
    astigmatism_angle_trace: list[float] = []
    phase_shift_trace: list[float] = []
    loss_trace: list[float] = []
    simulated_ctf2s = None

    # --- Optimization loop ---
    for _ in range(n_iterations):
        # Reset astigmatism if params became NaN/Inf
        if optimize_astigmatism:
            if (
                torch.isnan(astigmatism).any()
                or torch.isnan(angle_u).any()
                or torch.isnan(angle_v).any()
                or torch.isinf(astigmatism).any()
                or torch.isinf(angle_u).any()
                or torch.isinf(angle_v).any()
            ):
                _reset_astigmatism(
                    astigmatism,
                    angle_u,
                    angle_v,
                    initial_astigmatism,
                    _angle_u_init,
                    _angle_v_init,
                )
                continue
        astig_clamped, astig_angle_clamped = _get_astig_clamped(
            astigmatism, angle_u, angle_v, optimize_astigmatism
        )

        optimiser.zero_grad()
        loss_t_list: list[torch.Tensor] = []
        for t_idx in range(T):
            patch_ps_t = patch_power_spectra[t_idx]
            positions_t = normalised_patch_positions[t_idx]
            # Defocus from grid eval at this frame's positions
            predicted_defocus_t = defocus_model_obj(positions_t)
            predicted_defocus_t = einops.rearrange(predicted_defocus_t, "... 1 -> ...")
            predicted_defocus_t = _clamp_optional_bounds(
                predicted_defocus_t, defocus_bounds_microns
            )
            phase_shift_t, u_t, v_t = phase_shift_at_positions(
                positions_t,
                phase_models,
                phase_shift_bounds_degrees,
                fixed_phase_shift_deg=fixed_phase_shift_deg,
            )
            simulated_ctf2s_t = compute_ctf2_t(
                defocus_t=predicted_defocus_t,
                phase_shift_t=phase_shift_t,
                astig_clamped=astig_clamped,
                astig_angle_clamped=astig_angle_clamped,
                image_shape=image_shape,
                pixel_spacing_angstroms=pixel_spacing_angstroms,
                voltage_kev=voltage_kev,
                spherical_aberration_mm=spherical_aberration_mm,
                amplitude_contrast_fraction=amplitude_contrast_fraction,
                env_2d=env_2d,
                bp_filter=bp_filter,
                laser_params=laser_params,
                use_amplitude=use_amplitude,
            )
            simulated_ctf2s = simulated_ctf2s_t
            if (
                torch.isnan(simulated_ctf2s_t).any()
                or torch.isinf(simulated_ctf2s_t).any()
            ):
                continue
            loss_t = correlation_loss_t(simulated_ctf2s_t, patch_ps_t, u_t, v_t)
            loss_t_list.append(loss_t)

        # Skip step if no valid loss (e.g. all frames had NaN CTF)
        if len(loss_t_list) == 0:
            if optimize_astigmatism:
                _reset_astigmatism(
                    astigmatism,
                    angle_u,
                    angle_v,
                    initial_astigmatism,
                    _angle_u_init,
                    _angle_v_init,
                )
            continue
        total_loss = sum(loss_t_list) / len(loss_t_list)
        total_loss.backward()
        mean_loss = (sum(loss_t_list) / len(loss_t_list)).detach().cpu().item()
        if math.isnan(mean_loss) or math.isinf(mean_loss):
            if optimize_astigmatism:
                _reset_astigmatism(
                    astigmatism,
                    angle_u,
                    angle_v,
                    initial_astigmatism,
                    _angle_u_init,
                    _angle_v_init,
                )
            continue
        loss_trace.append(float(mean_loss))
        if early_stopper is not None and early_stopper(loss_trace[-1]):
            break
        # Reset astigmatism if grads are bad, then skip step
        if optimize_astigmatism and _check_astig_grad_and_reset(
            astigmatism,
            angle_u,
            angle_v,
            initial_astigmatism,
            _angle_u_init,
            _angle_v_init,
        ):
            optimiser.zero_grad()
            continue
        optimiser.step()
        _clamp_defocus_grid_after_step(defocus_model_obj, defocus_bounds_microns)
        clamp_phase_shift_after_step(phase_models, phase_shift_bounds_degrees)
        # Post-step: clamp astigmatism if needed
        if optimize_astigmatism:
            with torch.no_grad():
                if (
                    torch.isnan(astigmatism).any()
                    or torch.isnan(angle_u).any()
                    or torch.isnan(angle_v).any()
                    or torch.isinf(astigmatism).any()
                    or torch.isinf(angle_u).any()
                    or torch.isinf(angle_v).any()
                ):
                    _reset_astigmatism(
                        astigmatism,
                        angle_u,
                        angle_v,
                        initial_astigmatism,
                        _angle_u_init,
                        _angle_v_init,
                    )
                else:
                    astigmatism.clamp_(min=1e-6)
        # Record traces for this iteration
        defocus_models.append(defocus_model_obj.data.detach().clone())
        if optimize_astigmatism:
            astigmatism_trace.append(float(astigmatism.detach().cpu().item()))
            _norm = torch.sqrt(angle_u.detach() ** 2 + angle_v.detach() ** 2 + 1e-8)
            _a_rad = torch.atan2(angle_v.detach() / _norm, angle_u.detach() / _norm)
            _a_deg = float((_a_rad * (180.0 / math.pi) + 180.0) % 180.0)
            astigmatism_angle_trace.append(_astig_angle_to_m90_p90(_a_deg))
        _phase_val = phase_shift_trace_value(phase_models)
        if _phase_val is not None:
            phase_shift_trace.append(_phase_val)

    # --- Build final scalars and phase result ---
    (
        final_astigmatism,
        final_astigmatism_angle,
        final_envelope_B,
        _mean_defocus,
        final_defocus_u,
        final_defocus_v,
    ) = _grid_final_scalars(
        defocus_model_obj, astigmatism, angle_u, angle_v, envelope_B
    )
    final_phase_shift_deg, final_phase_shift_model_obj = build_phase_shift_result(
        phase_models, phase_shift_model
    )
    if not optimize_phase_shift and fixed_phase_shift_deg is not None:
        final_phase_shift_deg = fixed_phase_shift_deg

    astig_clamped_final, astig_angle_clamped_final = _get_astig_clamped(
        astigmatism, angle_u, angle_v, optimize_astigmatism
    )

    def _forward_frame_grid(t_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions_t = normalised_patch_positions[t_idx]
        predicted_defocus_t = einops.rearrange(
            defocus_model_obj(positions_t), "... 1 -> ..."
        )
        phase_shift_t, _, _ = phase_shift_at_positions(
            positions_t,
            phase_models,
            phase_shift_bounds_degrees,
            fixed_phase_shift_deg=fixed_phase_shift_deg,
        )
        return predicted_defocus_t, phase_shift_t

    cc_final = mean_pearson_r_final_2d(
        patch_power_spectra,
        _forward_frame_grid,
        astig_clamped=astig_clamped_final,
        astig_angle_clamped=astig_angle_clamped_final,
        image_shape=image_shape,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast_fraction=amplitude_contrast_fraction,
        env_2d=env_2d,
        bp_filter=bp_filter,
        laser_params=laser_params,
        use_amplitude=use_amplitude,
    )

    if debug:
        return Defocus2DResults(
            cross_correlation_final=cc_final,
            defocus_model_type="grid",
            defocus_model=defocus_model_obj,
            simulated_ctf2s=simulated_ctf2s,
            patch_power_spectra=patch_power_spectra,
            model_trace=defocus_models,
            astigmatism=final_astigmatism,
            astigmatism_angle=final_astigmatism_angle,
            astigmatism_trace=astigmatism_trace if optimize_astigmatism else None,
            astigmatism_angle_trace=astigmatism_angle_trace
            if optimize_astigmatism
            else None,
            envelope_B=final_envelope_B,
            loss_trace=loss_trace,
            defocus_u=final_defocus_u,
            defocus_v=final_defocus_v,
            phase_shift_degrees=final_phase_shift_deg,
            phase_shift_model_type=(
                phase_shift_model if optimize_phase_shift else None
            ),
            phase_shift_model=(
                final_phase_shift_model_obj if optimize_phase_shift else None
            ),
            phase_shift_trace=phase_shift_trace if optimize_phase_shift else None,
        )
    return Defocus2DResults(
        cross_correlation_final=cc_final,
        defocus_model_type="grid",
        defocus_model=defocus_model_obj,
        astigmatism=final_astigmatism,
        astigmatism_angle=final_astigmatism_angle,
        envelope_B=final_envelope_B,
        defocus_u=final_defocus_u,
        defocus_v=final_defocus_v,
        phase_shift_degrees=final_phase_shift_deg,
        phase_shift_model_type=phase_shift_model if optimize_phase_shift else None,
        phase_shift_model=(
            final_phase_shift_model_obj if optimize_phase_shift else None
        ),
        phase_shift_trace=phase_shift_trace if optimize_phase_shift else None,
    )
