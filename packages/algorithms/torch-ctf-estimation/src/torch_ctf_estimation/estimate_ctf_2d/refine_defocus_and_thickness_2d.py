"""Joint 2D gradient refinement of defocus and sample thickness."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from collections.abc import Callable

import einops
import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d

from torch_ctf_estimation.estimate_ctf_2d.ctf_loss_2d import (
    compute_thickness_ctf_ps_t,
    correlation_loss_t,
)
from torch_ctf_estimation.estimate_ctf_2d.estimate_ctf_2d_utils import (
    _clamp_defocus_grid_after_step,
    _clamp_optional_bounds,
    _get_astig_clamped,
    _shared_astigmatism_and_env,
)
from torch_ctf_estimation.estimate_ctf_2d.estimate_defocus_2d_linear import (
    _linear_defocus_at_positions,
)
from torch_ctf_estimation.metrics.fit_metrics import pearson_r_flat
from torch_ctf_estimation.models import (
    Defocus2DResults,
    LaserParams,
    LinearDefocusModel,
    Thickness2DResults,
)
from torch_ctf_estimation.utils.fitting_bounds import resolve_defocus_bounds


def refine_defocus_and_thickness_2d(
    patch_power_spectra: torch.Tensor,
    normalised_patch_positions: torch.Tensor,
    result2d: Defocus2DResults,
    initial_thickness_angstroms: float,
    frequency_fit_range_angstroms: tuple[float, float],
    pixel_spacing_angstroms: float,
    voltage_kev: float = 300.0,
    spherical_aberration_mm: float = 2.7,
    amplitude_contrast_fraction: float = 0.07,
    n_iterations: int = 100,
    defocus_lr: float = 0.01,
    thickness_lr: float = 50.0,
    thickness_grid_resolution: tuple[int, int, int] | None = None,
    defocus_bounds_microns: tuple[float, float] | None = None,
    laser_params: Optional[LaserParams] = None,
    axis_mask: Optional[torch.Tensor] = None,
    early_stopper: Callable[[float], bool] | None = None,
) -> tuple[Defocus2DResults, Thickness2DResults]:
    """Jointly refine 2D defocus and thickness with the thickness-modulated CTF.

    Seeds defocus from ``result2d`` and thickness from the 1D grid search.
    A scalar thickness (1x1x1 spline) is used unless ``thickness_grid_resolution``
    is set.

    Parameters
    ----------
    patch_power_spectra : torch.Tensor
        Background-subtracted patch power spectra ``(t, gh, gw, ph, pw)``.
    normalised_patch_positions : torch.Tensor
        Patch positions in ``[0, 1]``, shape ``(t, gh, gw, 3)``.
    result2d : Defocus2DResults
        Defocus result to refine (grid or linear).
    initial_thickness_angstroms : float
        Starting thickness from the 1D grid search.
    frequency_fit_range_angstroms : tuple[float, float]
        Fit band in Angstroms.
    pixel_spacing_angstroms : float
        Pixel size after any rescale.
    n_iterations, defocus_lr, thickness_lr :
        Optimiser settings.
    thickness_grid_resolution : tuple[int, int, int] | None
        Thickness spline resolution. ``None`` uses ``(1, 1, 1)``.
    defocus_bounds_microns : tuple[float, float] | None
        Defocus clamp bounds.
    laser_params, axis_mask :
        Optional LPP / laser-axis mask.
    early_stopper : callable or None, optional
        Stateful ``(loss) -> should_stop`` callback. Default None (run all
        ``n_iterations``).

    Returns
    -------
    result2d : Defocus2DResults
        Updated 2D defocus result.
    thickness2d : Thickness2DResults
        Fitted thickness spline and mean thickness.
    """
    defocus_bounds = resolve_defocus_bounds(defocus_bounds_microns)
    if thickness_grid_resolution is None:
        thickness_grid_resolution = (1, 1, 1)

    ph, pw_rfft = patch_power_spectra.shape[-2], patch_power_spectra.shape[-1]
    image_shape = (ph, (pw_rfft - 1) * 2)
    device = patch_power_spectra.device
    T = patch_power_spectra.shape[0]
    phase_shift_deg = (
        float(result2d.phase_shift_degrees)
        if result2d.phase_shift_degrees is not None
        else 0.0
    )
    astig = float(result2d.astigmatism or 0.0)
    astig_angle = float(result2d.astigmatism_angle or 0.0)
    env_b = float(result2d.envelope_B) if result2d.envelope_B is not None else 0.0

    (
        bp_filter,
        astigmatism_t,
        angle_u,
        angle_v,
        _u0,
        _v0,
        envelope_B,
        env_2d,
    ) = _shared_astigmatism_and_env(
        image_shape=image_shape,
        device=device,
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        initial_astigmatism=astig,
        initial_astigmatism_angle=astig_angle,
        optimize_astigmatism=False,
        initial_envelope_B=env_b,
        axis_mask=axis_mask,
    )
    patch_power_spectra = patch_power_spectra * bp_filter
    astig_clamped, astig_angle_clamped = _get_astig_clamped(
        astigmatism_t, angle_u, angle_v, optimize_astigmatism=False
    )

    thickness_data = (
        torch.ones(size=thickness_grid_resolution, device=device)
        * initial_thickness_angstroms
    )
    thickness_model = CubicCatmullRomGrid3d.from_grid_data(thickness_data).to(device)

    is_linear = result2d.defocus_model_type == "linear"
    defocus_grid: CubicCatmullRomGrid3d | None = None
    defocus_0_param: torch.nn.Parameter | None = None
    grad_mag_param: torch.nn.Parameter | None = None
    angle_u_param: torch.nn.Parameter | None = None
    angle_v_param: torch.nn.Parameter | None = None

    if is_linear:
        lm = result2d.defocus_model
        if not isinstance(lm, LinearDefocusModel):
            raise TypeError("Expected LinearDefocusModel for linear joint refine")
        defocus_0_param = torch.nn.Parameter(
            torch.tensor(lm.defocus_0, device=device, dtype=torch.float32)
        )
        grad_mag_param = torch.nn.Parameter(
            torch.tensor(
                lm.defocus_gradient_magnitude, device=device, dtype=torch.float32
            )
        )
        angle_rad = lm.defocus_gradient_angle * math.pi / 180.0
        angle_u_param = torch.nn.Parameter(
            torch.tensor(math.cos(angle_rad), device=device, dtype=torch.float32)
        )
        angle_v_param = torch.nn.Parameter(
            torch.tensor(math.sin(angle_rad), device=device, dtype=torch.float32)
        )
        param_groups = [
            {"params": [defocus_0_param], "lr": defocus_lr},
            {"params": [grad_mag_param, angle_u_param, angle_v_param], "lr": defocus_lr},
            {"params": list(thickness_model.parameters()), "lr": thickness_lr},
        ]
    else:
        grid = result2d.defocus_model
        if not isinstance(grid, CubicCatmullRomGrid3d):
            raise TypeError("Expected CubicCatmullRomGrid3d for grid joint refine")
        defocus_grid = CubicCatmullRomGrid3d.from_grid_data(grid.data.detach().clone())
        defocus_grid = defocus_grid.to(device)
        param_groups = [
            {"params": list(defocus_grid.parameters()), "lr": defocus_lr},
            {"params": list(thickness_model.parameters()), "lr": thickness_lr},
        ]

    optimiser = torch.optim.Adam(param_groups)
    loss_trace: list[float] = []

    def _defocus_at(positions_t: torch.Tensor) -> torch.Tensor:
        if is_linear:
            assert defocus_0_param is not None
            assert grad_mag_param is not None
            assert angle_u_param is not None
            assert angle_v_param is not None
            return _linear_defocus_at_positions(
                positions_t,
                defocus_0_param,
                grad_mag_param,
                angle_u_param,
                angle_v_param,
            )
        assert defocus_grid is not None
        predicted = einops.rearrange(defocus_grid(positions_t), "... 1 -> ...")
        return _clamp_optional_bounds(predicted, defocus_bounds)

    for _ in range(n_iterations):
        optimiser.zero_grad()
        loss_t_list: list[torch.Tensor] = []
        for t_idx in range(T):
            positions_t = normalised_patch_positions[t_idx]
            thickness_t = einops.rearrange(thickness_model(positions_t), "... 1 -> ...")
            defocus_t = _defocus_at(positions_t)
            simulated = compute_thickness_ctf_ps_t(
                thickness_t=thickness_t,
                defocus_t=defocus_t,
                astig_clamped=astig_clamped,
                astig_angle_clamped=astig_angle_clamped,
                phase_shift_deg=phase_shift_deg,
                image_shape=image_shape,
                pixel_spacing_angstroms=pixel_spacing_angstroms,
                voltage_kev=voltage_kev,
                spherical_aberration_mm=spherical_aberration_mm,
                amplitude_contrast_fraction=amplitude_contrast_fraction,
                env_2d=env_2d,
                bp_filter=bp_filter,
                laser_params=laser_params,
            )
            if torch.isnan(simulated).any() or torch.isinf(simulated).any():
                continue
            loss_t_list.append(
                correlation_loss_t(simulated, patch_power_spectra[t_idx])
            )
        if not loss_t_list:
            continue
        total_loss = sum(loss_t_list) / len(loss_t_list)
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            continue
        total_loss.backward()
        loss_trace.append(float(total_loss.detach().cpu().item()))
        optimiser.step()
        if early_stopper is not None and early_stopper(loss_trace[-1]):
            break
        with torch.no_grad():
            thickness_model.data.clamp_(min=1.0)
        if defocus_grid is not None:
            _clamp_defocus_grid_after_step(defocus_grid, defocus_bounds)
        elif defocus_0_param is not None:
            with torch.no_grad():
                defocus_0_param.clamp_(min=defocus_bounds[0], max=defocus_bounds[1])

    mean_thickness = float(thickness_model.data.detach().cpu().mean().item())
    rs: list[float] = []
    with torch.no_grad():
        for t_idx in range(T):
            positions_t = normalised_patch_positions[t_idx]
            thickness_t = einops.rearrange(thickness_model(positions_t), "... 1 -> ...")
            defocus_t = _defocus_at(positions_t)
            simulated = compute_thickness_ctf_ps_t(
                thickness_t=thickness_t,
                defocus_t=defocus_t,
                astig_clamped=astig_clamped,
                astig_angle_clamped=astig_angle_clamped,
                phase_shift_deg=phase_shift_deg,
                image_shape=image_shape,
                pixel_spacing_angstroms=pixel_spacing_angstroms,
                voltage_kev=voltage_kev,
                spherical_aberration_mm=spherical_aberration_mm,
                amplitude_contrast_fraction=amplitude_contrast_fraction,
                env_2d=env_2d,
                bp_filter=bp_filter,
                laser_params=laser_params,
            )
            if torch.isnan(simulated).any() or torch.isinf(simulated).any():
                continue
            rs.append(
                pearson_r_flat(
                    patch_power_spectra[t_idx].reshape(-1),
                    simulated.reshape(-1),
                )
            )
    cc_final = float(sum(rs) / len(rs)) if rs else None

    if is_linear:
        assert defocus_0_param is not None
        assert grad_mag_param is not None
        assert angle_u_param is not None
        assert angle_v_param is not None
        au = float(angle_u_param.detach().cpu().item())
        av = float(angle_v_param.detach().cpu().item())
        angle_deg = (math.atan2(av, au) * 180.0 / math.pi + 180.0) % 180.0
        mean_defocus = float(defocus_0_param.detach().cpu().item())
        updated_defocus: LinearDefocusModel | CubicCatmullRomGrid3d = LinearDefocusModel(
            defocus_0=mean_defocus,
            defocus_gradient_magnitude=float(grad_mag_param.detach().cpu().item()),
            defocus_gradient_angle=angle_deg,
        )
        model_type = "linear"
    else:
        assert defocus_grid is not None
        updated_defocus = defocus_grid
        mean_defocus = float(defocus_grid.data.detach().cpu().mean().item())
        model_type = "grid"

    result2d_out = result2d.model_copy(
        update={
            "defocus_model_type": model_type,
            "defocus_model": updated_defocus,
            "defocus_u": mean_defocus + astig / 2.0,
            "defocus_v": mean_defocus - astig / 2.0,
            "cross_correlation_final": cc_final,
            "envelope_B": float(envelope_B.detach().cpu().item()),
        }
    )
    thickness2d = Thickness2DResults(
        mean_thickness=mean_thickness,
        cross_correlation_final=cc_final,
        thickness_model=thickness_model,
        envelope_B=float(envelope_B.detach().cpu().item()),
        loss_trace=loss_trace,
    )
    return result2d_out, thickness2d
