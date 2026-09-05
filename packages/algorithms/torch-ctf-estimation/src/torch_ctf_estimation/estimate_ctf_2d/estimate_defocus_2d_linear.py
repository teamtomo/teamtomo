"""
2D defocus estimation using a linear (tilt) model in (x, y).

Defocus = defocus_0 + gradient_magnitude * (projected position along tilt direction).
When nt > 1, defocus_0 / gradient / angle can be 1D splines in t; otherwise scalars.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Optional, TypeAlias

import einops
import torch

from torch_ctf_estimation.estimate_ctf_2d.ctf_loss_2d import (
    compute_ctf2_t,
    correlation_loss_t,
    mean_pearson_r_final_2d,
)
from torch_ctf_estimation.estimate_ctf_2d.estimate_ctf_2d_utils import (
    _astig_angle_to_m90_p90,
    _check_astig_grad_and_reset,
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
from torch_ctf_estimation.models import (
    Defocus2DResults,
    LaserParams,
    LinearDefocusModel,
)
from torch_ctf_estimation.utils.fitting_bounds import (
    resolve_defocus_bounds,
    resolve_phase_shift_bounds,
)

# Type for 1D spline params; use Any to avoid two TypeAlias assignments
_Spline1D: TypeAlias = Any

try:
    from torch_cubic_spline_grids import CubicCatmullRomGrid1d
except ImportError:
    CubicCatmullRomGrid1d = None


@dataclass
class _LinearDefocusParams:
    """
    Container for linear defocus parameters: either 1D splines in t or scalar params.

    Exactly one of (spline branch) or (scalar branch) is set; the other is None.
    """

    defocus_0_fixed: Optional[float]
    defocus_0_param: Optional[torch.nn.Parameter]
    grad_mag_param: Optional[torch.nn.Parameter]
    grad_angle_u: Optional[torch.nn.Parameter]
    grad_angle_v: Optional[torch.nn.Parameter]
    defocus_0_spline_1d: Optional[_Spline1D]
    grad_mag_spline_1d: Optional[_Spline1D]
    grad_angle_u_spline_1d: Optional[_Spline1D]
    grad_angle_v_spline_1d: Optional[_Spline1D]


def _setup_linear_spectra_and_shape(
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


def _setup_linear_defocus_params(
    nt: int,
    device: torch.device,
    initial_defocus: float,
    initial_defocus_gradient_angle: float,
    initial_defocus_gradient_magnitude: float,
    fix_defocus_0: Optional[float],
) -> _LinearDefocusParams:
    """
    Create linear defocus parameters.

    Either 1D splines in t (if nt>1 and available) or scalar Parameters.
    Gradient direction is stored as (angle_u, angle_v) on unit circle.
    """
    _grad_angle_rad = initial_defocus_gradient_angle * math.pi / 180.0
    _grad_angle_u_init = math.cos(_grad_angle_rad)
    _grad_angle_v_init = math.sin(_grad_angle_rad)
    init_grad_mag = (
        initial_defocus_gradient_magnitude
        if initial_defocus_gradient_magnitude != 0
        else 0.05
    )
    use_linear_splines = nt > 1 and CubicCatmullRomGrid1d is not None

    if use_linear_splines:
        if fix_defocus_0 is not None:
            defocus_0_spline_1d = CubicCatmullRomGrid1d.from_grid_data(
                torch.ones(nt, device=device) * fix_defocus_0
            ).to(device)
            for p in defocus_0_spline_1d.parameters():
                p.requires_grad = False
        else:
            defocus_0_spline_1d = CubicCatmullRomGrid1d.from_grid_data(
                torch.ones(nt, device=device) * initial_defocus
            ).to(device)
        grad_mag_spline_1d = CubicCatmullRomGrid1d.from_grid_data(
            torch.ones(nt, device=device) * init_grad_mag
        ).to(device)
        grad_angle_u_spline_1d = CubicCatmullRomGrid1d.from_grid_data(
            torch.ones(nt, device=device) * _grad_angle_u_init
        ).to(device)
        grad_angle_v_spline_1d = CubicCatmullRomGrid1d.from_grid_data(
            torch.ones(nt, device=device) * _grad_angle_v_init
        ).to(device)
        return _LinearDefocusParams(
            defocus_0_fixed=fix_defocus_0,
            defocus_0_param=None,
            grad_mag_param=None,
            grad_angle_u=None,
            grad_angle_v=None,
            defocus_0_spline_1d=defocus_0_spline_1d,
            grad_mag_spline_1d=grad_mag_spline_1d,
            grad_angle_u_spline_1d=grad_angle_u_spline_1d,
            grad_angle_v_spline_1d=grad_angle_v_spline_1d,
        )
    # Scalar branch
    defocus_0_param = None
    if fix_defocus_0 is None:
        defocus_0_param = torch.nn.Parameter(
            torch.tensor(initial_defocus, device=device, dtype=torch.float32)
        ).to(device)
    grad_mag_param = torch.nn.Parameter(
        torch.tensor(init_grad_mag, device=device, dtype=torch.float32)
    ).to(device)
    grad_angle_u = torch.nn.Parameter(
        torch.tensor(_grad_angle_u_init, device=device, dtype=torch.float32)
    ).to(device)
    grad_angle_v = torch.nn.Parameter(
        torch.tensor(_grad_angle_v_init, device=device, dtype=torch.float32)
    ).to(device)
    return _LinearDefocusParams(
        defocus_0_fixed=fix_defocus_0,
        defocus_0_param=defocus_0_param,
        grad_mag_param=grad_mag_param,
        grad_angle_u=grad_angle_u,
        grad_angle_v=grad_angle_v,
        defocus_0_spline_1d=None,
        grad_mag_spline_1d=None,
        grad_angle_u_spline_1d=None,
        grad_angle_v_spline_1d=None,
    )


def _linear_defocus_at_positions(
    positions_t: torch.Tensor,
    defocus_0_t: torch.Tensor,
    grad_mag_t: torch.Tensor,
    angle_u_t: torch.Tensor,
    angle_v_t: torch.Tensor,
) -> torch.Tensor:
    """
    Compute defocus at positions from linear tilt model.

    defocus = defocus_0 + grad_mag * (projected distance along tilt direction).
    Tilt direction from (angle_u_t, angle_v_t); positions use x_norm, y_norm in [0,1].
    """
    x_norm = positions_t[..., 1]
    y_norm = positions_t[..., 2]
    _eps = 1e-8
    _norm = torch.sqrt(angle_u_t**2 + angle_v_t**2 + _eps)
    _dir_u = angle_u_t / _norm
    _dir_v = angle_v_t / _norm
    angle_rad = torch.atan2(_dir_v, _dir_u)
    projected = (x_norm - 0.5) * torch.cos(angle_rad) + (y_norm - 0.5) * torch.sin(
        angle_rad
    )
    return defocus_0_t + grad_mag_t * projected


def _clamp_linear_defocus_params_after_step(
    params: _LinearDefocusParams,
    defocus_bounds_microns: tuple[float, float] | None,
) -> None:
    """Clamp linear defocus base parameters after an optimizer step."""
    if defocus_bounds_microns is None:
        return
    lo, hi = defocus_bounds_microns
    with torch.no_grad():
        if params.defocus_0_param is not None:
            params.defocus_0_param.clamp_(min=lo, max=hi)
        if params.defocus_0_spline_1d is not None:
            params.defocus_0_spline_1d.data.clamp_(min=lo, max=hi)


def _linear_get_defocus_components_t(
    params: _LinearDefocusParams,
    t_norm: torch.Tensor,
    x_norm: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get (defocus_0_t, grad_mag_t, angle_u_t, angle_v_t) for one frame.

    Uses spline eval or scalar expand depending on params.
    """
    if params.defocus_0_spline_1d is not None:
        assert params.grad_mag_spline_1d is not None
        assert params.grad_angle_u_spline_1d is not None
        assert params.grad_angle_v_spline_1d is not None
        defocus_0_t = params.defocus_0_spline_1d(t_norm).squeeze(-1)
        grad_mag_t = params.grad_mag_spline_1d(t_norm).squeeze(-1)
        angle_u_t = params.grad_angle_u_spline_1d(t_norm).squeeze(-1)
        angle_v_t = params.grad_angle_v_spline_1d(t_norm).squeeze(-1)
    else:
        assert params.grad_mag_param is not None
        assert params.grad_angle_u is not None
        assert params.grad_angle_v is not None
        if params.defocus_0_param is not None:
            defocus_0_t = params.defocus_0_param.expand_as(x_norm)
        else:
            assert params.defocus_0_fixed is not None
            defocus_0_t = torch.full_like(x_norm, params.defocus_0_fixed, device=device)
        grad_mag_t = params.grad_mag_param.expand_as(x_norm)
        angle_u_t = params.grad_angle_u.expand_as(x_norm)
        angle_v_t = params.grad_angle_v.expand_as(x_norm)
    return defocus_0_t, grad_mag_t, angle_u_t, angle_v_t


def _linear_trace_snapshot(
    params: _LinearDefocusParams,
    device: torch.device,
) -> torch.Tensor:
    """Current [d0, grad_mag, angle_deg, 0] for appending to defocus_models trace."""
    if params.defocus_0_spline_1d is not None:
        assert params.grad_mag_spline_1d is not None
        assert params.grad_angle_u_spline_1d is not None
        assert params.grad_angle_v_spline_1d is not None
        d0 = params.defocus_0_spline_1d.data.detach().mean().cpu().item()
        gm = params.grad_mag_spline_1d.data.detach().mean().cpu().item()
        au = params.grad_angle_u_spline_1d.data.detach().mean().cpu().item()
        av = params.grad_angle_v_spline_1d.data.detach().mean().cpu().item()
    else:
        assert params.grad_mag_param is not None
        assert params.grad_angle_u is not None
        assert params.grad_angle_v is not None
        d0 = (
            params.defocus_0_param.detach().cpu().item()
            if params.defocus_0_param is not None
            else params.defocus_0_fixed
        )
        gm = params.grad_mag_param.detach().cpu().item()
        au = params.grad_angle_u.detach().cpu().item()
        av = params.grad_angle_v.detach().cpu().item()
    _norm = (au**2 + av**2 + 1e-8) ** 0.5
    angle_deg = (math.atan2(av / _norm, au / _norm) * (180.0 / math.pi) + 180.0) % 180.0
    return torch.tensor([d0, gm, angle_deg, 0.0], device=device)


def _linear_final_scalars(
    params: _LinearDefocusParams,
    astigmatism: torch.Tensor,
) -> tuple[float, float, float, float, float, float]:
    """
    Final scalar values from fitted linear params and astigmatism.

    Returns (final_defocus_0, final_grad_mag, final_grad_angle, mean_defocus,
             final_defocus_u, final_defocus_v).
    """
    final_astigmatism = float(astigmatism.detach().cpu().item())

    if params.defocus_0_spline_1d is not None:
        assert params.grad_mag_spline_1d is not None
        assert params.grad_angle_u_spline_1d is not None
        assert params.grad_angle_v_spline_1d is not None
        _gn = torch.sqrt(
            params.grad_angle_u_spline_1d.data**2
            + params.grad_angle_v_spline_1d.data**2
            + 1e-8
        )
        _grad_angle_deg = (
            torch.atan2(
                params.grad_angle_v_spline_1d.data / _gn,
                params.grad_angle_u_spline_1d.data / _gn,
            )
            * (180.0 / math.pi)
            + 180.0
        )
        _grad_angle_deg = _grad_angle_deg % 180.0
        final_grad_angle = float(_grad_angle_deg.mean().cpu().item())
        final_defocus_0 = float(params.defocus_0_spline_1d.data.mean().cpu().item())
        final_grad_mag = float(params.grad_mag_spline_1d.data.mean().cpu().item())
        mean_defocus = float(
            params.defocus_0_spline_1d.data.detach().cpu().mean().item()
        )
    else:
        assert params.grad_mag_param is not None
        assert params.grad_angle_u is not None
        assert params.grad_angle_v is not None
        _gn = torch.sqrt(
            params.grad_angle_u.detach() ** 2 + params.grad_angle_v.detach() ** 2 + 1e-8
        )
        final_grad_angle = float(
            (
                torch.atan2(
                    params.grad_angle_v.detach() / _gn,
                    params.grad_angle_u.detach() / _gn,
                )
                .cpu()
                .item()
                * (180.0 / math.pi)
                + 180.0
            )
            % 180.0
        )
        if params.defocus_0_param is not None:
            final_defocus_0 = float(params.defocus_0_param.detach().cpu().item())
        else:
            assert params.defocus_0_fixed is not None
            final_defocus_0 = float(params.defocus_0_fixed)
        final_grad_mag = float(params.grad_mag_param.detach().cpu().item())
        if params.defocus_0_param is not None:
            mean_defocus = float(params.defocus_0_param.detach().cpu().item())
        else:
            assert params.defocus_0_fixed is not None
            mean_defocus = float(params.defocus_0_fixed)
    final_defocus_u = mean_defocus + final_astigmatism / 2.0
    final_defocus_v = mean_defocus - final_astigmatism / 2.0
    return (
        final_defocus_0,
        final_grad_mag,
        final_grad_angle,
        mean_defocus,
        final_defocus_u,
        final_defocus_v,
    )


def _build_linear_param_groups(
    params: _LinearDefocusParams,
    defocus_lr: float,
    defocus_gradient_magnitude_lr: float,
    defocus_gradient_angle_lr: float,
    optimize_astigmatism: bool,
    astigmatism: torch.Tensor,
    angle_u: torch.Tensor,
    angle_v: torch.Tensor,
    astigmatism_lr: float,
    astigmatism_angle_lr: float,
    phase_models: Optional[PhaseShiftModels],
    phase_shift_lr: float,
) -> list[dict]:
    """Build Adam param groups for linear defocus, astigmatism, and phase."""
    param_groups = []
    if params.defocus_0_param is not None:
        param_groups.extend(
            [
                {"params": [params.defocus_0_param], "lr": defocus_lr},
                {
                    "params": [params.grad_mag_param],
                    "lr": defocus_gradient_magnitude_lr,
                },
                {
                    "params": [params.grad_angle_u, params.grad_angle_v],
                    "lr": defocus_gradient_angle_lr,
                },
            ]
        )
    if (
        params.defocus_0_fixed is not None
        and params.defocus_0_param is None
        and params.grad_mag_param is not None
    ):
        param_groups.extend(
            [
                {
                    "params": [params.grad_mag_param],
                    "lr": defocus_gradient_magnitude_lr,
                },
                {
                    "params": [params.grad_angle_u, params.grad_angle_v],
                    "lr": defocus_gradient_angle_lr,
                },
            ]
        )
    if params.defocus_0_spline_1d is not None:
        assert params.grad_mag_spline_1d is not None
        assert params.grad_angle_u_spline_1d is not None
        assert params.grad_angle_v_spline_1d is not None
        param_groups.extend(
            [
                {"params": params.defocus_0_spline_1d.parameters(), "lr": defocus_lr},
                {
                    "params": params.grad_mag_spline_1d.parameters(),
                    "lr": defocus_gradient_magnitude_lr,
                },
                {
                    "params": params.grad_angle_u_spline_1d.parameters(),
                    "lr": defocus_gradient_angle_lr,
                },
                {
                    "params": params.grad_angle_v_spline_1d.parameters(),
                    "lr": defocus_gradient_angle_lr,
                },
            ]
        )
    if optimize_astigmatism:
        param_groups.extend(
            [
                {"params": [astigmatism], "lr": astigmatism_lr},
                {"params": [angle_u, angle_v], "lr": astigmatism_angle_lr},
            ]
        )
    param_groups.extend(phase_shift_param_groups(phase_models, phase_shift_lr))
    return param_groups


def _spline_data_or_none(spline: Optional[_Spline1D]) -> Optional[torch.Tensor]:
    """Return spline.data.detach().clone() or None; avoids union-attr on optional."""
    if spline is None:
        return None
    data = getattr(spline, "data", None)
    return data.detach().clone() if data is not None else None


def _build_linear_defocus_model(
    params: _LinearDefocusParams,
    final_defocus_0: float,
    final_grad_mag: float,
    final_grad_angle: float,
) -> LinearDefocusModel:
    """Build the LinearDefocusModel with optional spline data for serialisation."""
    return LinearDefocusModel(
        defocus_0=final_defocus_0,
        defocus_gradient_magnitude=final_grad_mag,
        defocus_gradient_angle=final_grad_angle,
        defocus_0_spline_data=_spline_data_or_none(params.defocus_0_spline_1d),
        gradient_magnitude_spline_data=_spline_data_or_none(params.grad_mag_spline_1d),
        angle_u_spline_data=_spline_data_or_none(params.grad_angle_u_spline_1d),
        angle_v_spline_data=_spline_data_or_none(params.grad_angle_v_spline_1d),
    )


def estimate_defocus_2d_linear(
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
    Estimate defocus in 2D using a linear (tilt) model in (x, y).

    Optional cubic spline in t when nt > 1. Only the first element of
    defocus_grid_resolution (nt) is used.
    See :func:`estimate_ctf_2d` for other parameter descriptions.
    """
    defocus_bounds_microns = resolve_defocus_bounds(defocus_bounds_microns)
    phase_shift_bounds_degrees = resolve_phase_shift_bounds(phase_shift_bounds_degrees)
    # --- Setup: spectra, shape, linear defocus params, phase models ---
    (
        patch_power_spectra,
        image_shape,
        device,
        T,
    ) = _setup_linear_spectra_and_shape(patch_power_spectra, defocus_grid_resolution)
    nt = defocus_grid_resolution[0]
    linear_params = _setup_linear_defocus_params(
        nt,
        device,
        initial_defocus,
        initial_defocus_gradient_angle,
        initial_defocus_gradient_magnitude,
        fix_defocus_0,
    )
    phase_models_linear = init_phase_shift_models(
        optimize_phase_shift=optimize_phase_shift,
        phase_shift_model=phase_shift_model,
        initial_phase_shift=initial_phase_shift,
        grid_resolution=defocus_grid_resolution,
        device=device,
        phase_shift_quadratic_perpendicular_axis=phase_shift_quadratic_perpendicular_axis,
    )

    # --- Bandpass, astigmatism, envelope (shared with grid) ---
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
    param_groups = _build_linear_param_groups(
        linear_params,
        defocus_lr,
        defocus_gradient_magnitude_lr,
        defocus_gradient_angle_lr,
        optimize_astigmatism,
        astigmatism,
        angle_u,
        angle_v,
        astigmatism_lr,
        astigmatism_angle_lr,
        phase_models_linear,
        phase_shift_lr,
    )
    optimiser = torch.optim.Adam(params=param_groups)
    defocus_models: list[torch.Tensor] = []
    astigmatism_trace: list[float] = []
    astigmatism_angle_trace: list[float] = []
    phase_shift_trace_linear: list[float] = []
    loss_trace: list[float] = []
    simulated_ctf2s = None

    # --- Optimization loop ---
    for _ in range(n_iterations):
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
            t_norm = positions_t[..., 0:1]
            x_norm = positions_t[..., 1]
            (
                defocus_0_t,
                grad_mag_t,
                angle_u_t,
                angle_v_t,
            ) = _linear_get_defocus_components_t(linear_params, t_norm, x_norm, device)
            predicted_defocus_t = _linear_defocus_at_positions(
                positions_t, defocus_0_t, grad_mag_t, angle_u_t, angle_v_t
            )
            predicted_defocus_t = _clamp_optional_bounds(
                predicted_defocus_t, defocus_bounds_microns
            )

            phase_shift_t, u_t, v_t = phase_shift_at_positions(
                positions_t,
                phase_models_linear,
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
        _clamp_linear_defocus_params_after_step(linear_params, defocus_bounds_microns)
        clamp_phase_shift_after_step(phase_models_linear, phase_shift_bounds_degrees)
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
        defocus_models.append(_linear_trace_snapshot(linear_params, device))
        if optimize_astigmatism:
            astigmatism_trace.append(float(astigmatism.detach().cpu().item()))
            _norm = torch.sqrt(angle_u.detach() ** 2 + angle_v.detach() ** 2 + 1e-8)
            _a_rad = torch.atan2(angle_v.detach() / _norm, angle_u.detach() / _norm)
            _a_deg = float((_a_rad * (180.0 / math.pi) + 180.0) % 180.0)
            astigmatism_angle_trace.append(_astig_angle_to_m90_p90(_a_deg))
        _phase_val_lin = phase_shift_trace_value(phase_models_linear)
        if _phase_val_lin is not None:
            phase_shift_trace_linear.append(_phase_val_lin)

    # --- Final scalars, linear defocus model, phase result ---
    (
        final_defocus_0,
        final_grad_mag,
        final_grad_angle,
        _mean_defocus,
        final_defocus_u,
        final_defocus_v,
    ) = _linear_final_scalars(linear_params, astigmatism)
    final_astigmatism = float(astigmatism.detach().cpu().item())
    _fn = torch.sqrt(angle_u.detach() ** 2 + angle_v.detach() ** 2 + 1e-8)
    _fa_rad = torch.atan2(angle_v.detach() / _fn, angle_u.detach() / _fn)
    _fa_deg = float((_fa_rad.cpu().item() * (180.0 / math.pi) + 180.0) % 180.0)
    final_astigmatism_angle = _astig_angle_to_m90_p90(_fa_deg)
    final_envelope_B = float(envelope_B.detach().cpu().item())

    defocus_model_obj = _build_linear_defocus_model(
        linear_params, final_defocus_0, final_grad_mag, final_grad_angle
    )
    final_phase_shift_deg_linear, final_phase_shift_model_obj_linear = (
        build_phase_shift_result(phase_models_linear, phase_shift_model)
    )
    if not optimize_phase_shift and fixed_phase_shift_deg is not None:
        final_phase_shift_deg_linear = fixed_phase_shift_deg

    astig_clamped_final, astig_angle_clamped_final = _get_astig_clamped(
        astigmatism, angle_u, angle_v, optimize_astigmatism
    )

    def _forward_frame_linear(t_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions_t = normalised_patch_positions[t_idx]
        t_norm = positions_t[..., 0:1]
        x_norm = positions_t[..., 1]
        (
            defocus_0_t,
            grad_mag_t,
            angle_u_t,
            angle_v_t,
        ) = _linear_get_defocus_components_t(linear_params, t_norm, x_norm, device)
        predicted_defocus_t = _linear_defocus_at_positions(
            positions_t, defocus_0_t, grad_mag_t, angle_u_t, angle_v_t
        )
        phase_shift_t, _, _ = phase_shift_at_positions(
            positions_t,
            phase_models_linear,
            phase_shift_bounds_degrees,
            fixed_phase_shift_deg=fixed_phase_shift_deg,
        )
        return predicted_defocus_t, phase_shift_t

    cc_final = mean_pearson_r_final_2d(
        patch_power_spectra,
        _forward_frame_linear,
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
            defocus_model_type="linear",
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
            phase_shift_degrees=final_phase_shift_deg_linear,
            phase_shift_model_type=phase_shift_model if optimize_phase_shift else None,
            phase_shift_model=final_phase_shift_model_obj_linear
            if optimize_phase_shift
            else None,
            phase_shift_trace=phase_shift_trace_linear
            if optimize_phase_shift
            else None,
        )
    return Defocus2DResults(
        cross_correlation_final=cc_final,
        defocus_model_type="linear",
        defocus_model=defocus_model_obj,
        astigmatism=final_astigmatism,
        astigmatism_angle=final_astigmatism_angle,
        envelope_B=final_envelope_B,
        defocus_u=final_defocus_u,
        defocus_v=final_defocus_v,
        phase_shift_degrees=final_phase_shift_deg_linear,
        phase_shift_model_type=phase_shift_model if optimize_phase_shift else None,
        phase_shift_model=final_phase_shift_model_obj_linear
        if optimize_phase_shift
        else None,
        phase_shift_trace=phase_shift_trace_linear if optimize_phase_shift else None,
    )
