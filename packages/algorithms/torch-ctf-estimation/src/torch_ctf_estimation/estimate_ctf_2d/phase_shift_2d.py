"""Phase shift models for 2D CTF estimation (grid u/v or quadratic in s,t)."""

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d

from torch_ctf_estimation.models import QuadraticPhaseShiftModel


@dataclass
class PhaseShiftModels:
    """
    Phase shift parameterisation: grid (u, v) or quadratic (C, g1, k1, g2, k2, alpha).

    When not optimizing phase, u_grid and v_grid are None and quad_params is None.
    """

    u_grid: Optional[CubicCatmullRomGrid3d] = None
    v_grid: Optional[CubicCatmullRomGrid3d] = None
    quad_params: Optional[dict[str, torch.nn.Parameter]] = None
    quadratic_perpendicular_axis: bool = False


def init_phase_shift_models(
    optimize_phase_shift: bool,
    phase_shift_model: Literal["grid", "quadratic"],
    initial_phase_shift: float,
    grid_resolution: tuple[int, int, int],
    device: torch.device,
    phase_shift_quadratic_perpendicular_axis: bool = False,
) -> Optional[PhaseShiftModels]:
    """
    Initialise phase shift models (grid u/v or quadratic params).

    Returns
    -------
    PhaseShiftModels or None
        When optimize_phase_shift is False, returns None.
    """
    if not optimize_phase_shift:
        return None
    if phase_shift_model == "grid":
        theta_rad = initial_phase_shift * (math.pi / 180.0)
        u_init = math.cos(2.0 * theta_rad)
        v_init = math.sin(2.0 * theta_rad)
        u_data = torch.ones(size=grid_resolution, device=device) * u_init
        v_data = torch.ones(size=grid_resolution, device=device) * v_init
        u_grid = CubicCatmullRomGrid3d.from_grid_data(u_data).to(device)
        v_grid = CubicCatmullRomGrid3d.from_grid_data(v_data).to(device)
        return PhaseShiftModels(
            u_grid=u_grid,
            v_grid=v_grid,
            quad_params=None,
            quadratic_perpendicular_axis=False,
        )
    # quadratic: f = C + g1*s + k1*s^2 + g2*t + k2*t^2
    g2 = torch.nn.Parameter(
        torch.tensor(0.0, device=device, dtype=torch.float32),
        requires_grad=phase_shift_quadratic_perpendicular_axis,
    )
    k2 = torch.nn.Parameter(
        torch.tensor(0.0, device=device, dtype=torch.float32),
        requires_grad=phase_shift_quadratic_perpendicular_axis,
    )
    quad_params = {
        "C": torch.nn.Parameter(
            torch.tensor(initial_phase_shift, device=device, dtype=torch.float32)
        ),
        "g1": torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float32)),
        "k1": torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float32)),
        "g2": g2,
        "k2": k2,
        "alpha": torch.nn.Parameter(
            torch.tensor(0.0, device=device, dtype=torch.float32)
        ),
    }
    return PhaseShiftModels(
        u_grid=None,
        v_grid=None,
        quad_params=quad_params,
        quadratic_perpendicular_axis=phase_shift_quadratic_perpendicular_axis,
    )


def phase_shift_at_positions(
    positions_t: torch.Tensor,
    phase_models: Optional[PhaseShiftModels],
    phase_shift_bounds: tuple[float, float] | None = None,
    fixed_phase_shift_deg: float | None = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Evaluate phase shift (and u, v for unit-circle penalty) at normalised positions.

    positions_t has shape (..., 3) with [t_norm, x_norm, y_norm] in [0,1].

    Returns
    -------
    phase_shift_t : torch.Tensor
        Phase shift in degrees, shape (...,). When phase_models is None, returns
        ``fixed_phase_shift_deg`` everywhere if set, otherwise zero.
    u_t, v_t : torch.Tensor or None
        For grid model, (u,v) on unit circle for penalty; otherwise None.
    """
    if phase_models is None:
        if fixed_phase_shift_deg is not None:
            phase = torch.full(
                positions_t.shape[:-1],
                fixed_phase_shift_deg,
                device=positions_t.device,
                dtype=positions_t.dtype,
            )
            return phase, None, None
        return (
            torch.zeros(positions_t.shape[:-1], device=positions_t.device),
            None,
            None,
        )
    if phase_models.u_grid is not None and phase_models.v_grid is not None:
        u_t = phase_models.u_grid(positions_t).squeeze(-1)
        v_t = phase_models.v_grid(positions_t).squeeze(-1)
        phase_shift_t = torch.remainder(
            0.5 * torch.atan2(v_t, u_t) * (180.0 / math.pi), 180.0
        )
        if phase_shift_bounds is not None:
            lo, hi = phase_shift_bounds
            phase_shift_t = torch.clamp(phase_shift_t, min=lo, max=hi)
        return phase_shift_t, u_t, v_t
    assert phase_models.quad_params is not None
    x = 2.0 * positions_t[..., 1] - 1.0
    y = 2.0 * positions_t[..., 2] - 1.0
    alpha = phase_models.quad_params["alpha"]
    s = x * torch.cos(alpha) + y * torch.sin(alpha)
    t = -x * torch.sin(alpha) + y * torch.cos(alpha)
    qp = phase_models.quad_params
    phase_shift_t = (
        qp["C"] + qp["g1"] * s + qp["k1"] * (s**2) + qp["g2"] * t + qp["k2"] * (t**2)
    )
    if phase_shift_bounds is not None:
        lo, hi = phase_shift_bounds
        phase_shift_t = torch.clamp(phase_shift_t, min=lo, max=hi)
    return phase_shift_t, None, None


def clamp_phase_shift_after_step(
    phase_models: Optional[PhaseShiftModels],
    phase_shift_bounds: tuple[float, float] | None = None,
) -> None:
    """Clamp quadratic C and grid u/v mean phase after optimizer step."""
    if phase_models is None:
        return
    if phase_models.quad_params is not None:
        if phase_shift_bounds is None:
            return
        lo, hi = phase_shift_bounds
        with torch.no_grad():
            phase_models.quad_params["C"].clamp_(min=lo, max=hi)
        return
    if (
        phase_models.u_grid is not None
        and phase_models.v_grid is not None
        and phase_shift_bounds is not None
    ):
        lo, hi = phase_shift_bounds
        with torch.no_grad():
            u_mean = phase_models.u_grid.data.mean()
            v_mean = phase_models.v_grid.data.mean()
            phase_deg = (
                0.5 * torch.atan2(v_mean, u_mean) * (180.0 / math.pi)
            ).item()
            phase_deg = max(lo, min(hi, phase_deg))
            theta_rad = phase_deg * (math.pi / 180.0)
            u_new = math.cos(2.0 * theta_rad)
            v_new = math.sin(2.0 * theta_rad)
            phase_models.u_grid.data.fill_(u_new)
            phase_models.v_grid.data.fill_(v_new)


def build_phase_shift_result(
    phase_models: Optional[PhaseShiftModels],
    _phase_shift_model: Literal["grid", "quadratic"],
) -> tuple[Optional[float], Optional[object]]:
    """
    Build (final_phase_shift_deg, final_phase_shift_model_obj) for Defocus2DResults.

    Returns
    -------
    final_phase_shift_deg : float or None
    final_phase_shift_model_obj :
        tuple of (u_grid, v_grid), QuadraticPhaseShiftModel, or None
    """
    if phase_models is None:
        return None, None
    if phase_models.u_grid is not None and phase_models.v_grid is not None:
        _mu = phase_models.u_grid.data.detach().cpu().mean().item()
        _mv = phase_models.v_grid.data.detach().cpu().mean().item()
        _p = (0.5 * math.degrees(math.atan2(_mv, _mu))) % 180.0
        final_deg = min(_p, 180.0 - _p)
        return final_deg, (phase_models.u_grid, phase_models.v_grid)
    assert phase_models.quad_params is not None
    qp = phase_models.quad_params
    _c = float(qp["C"].detach().cpu().item())
    final_deg = min(_c, 180.0 - _c)
    model_obj = QuadraticPhaseShiftModel(
        C=float(qp["C"].detach().cpu().item()),
        alpha_rad=float(qp["alpha"].detach().cpu().item()),
        g1=float(qp["g1"].detach().cpu().item()),
        k1=float(qp["k1"].detach().cpu().item()),
        g2=float(qp["g2"].detach().cpu().item()),
        k2=float(qp["k2"].detach().cpu().item()),
    )
    return final_deg, model_obj


def phase_shift_param_groups(
    phase_models: Optional[PhaseShiftModels],
    phase_shift_lr: float,
) -> list[dict]:
    """Return param groups for Adam for phase shift models."""
    if phase_models is None:
        return []
    out: list[dict] = []
    if phase_models.u_grid is not None and phase_models.v_grid is not None:
        out.append({"params": phase_models.u_grid.parameters(), "lr": phase_shift_lr})
        out.append({"params": phase_models.v_grid.parameters(), "lr": phase_shift_lr})
    if phase_models.quad_params is not None:
        trainable = [p for p in phase_models.quad_params.values() if p.requires_grad]
        if trainable:
            out.append({"params": trainable, "lr": phase_shift_lr})
    return out


def phase_shift_trace_value(
    phase_models: Optional[PhaseShiftModels],
) -> Optional[float]:
    """Single scalar for trace list (mean phase this step)."""
    if phase_models is None:
        return None
    if phase_models.u_grid is not None and phase_models.v_grid is not None:
        _mu = phase_models.u_grid.data.detach().cpu().mean().item()
        _mv = phase_models.v_grid.data.detach().cpu().mean().item()
        _p = (0.5 * math.degrees(math.atan2(_mv, _mu))) % 180.0
        return min(_p, 180.0 - _p)
    assert phase_models.quad_params is not None
    _c = float(phase_models.quad_params["C"].detach().cpu().item())
    return min(_c, 180.0 - _c)
