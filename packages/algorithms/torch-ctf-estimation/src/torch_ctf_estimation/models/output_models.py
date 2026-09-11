"""Pydantic models for CTF estimation JSON output (data_io export schema)."""

from __future__ import annotations

from typing import Any, Literal

from teamtomo_basemodel import BaseModelTeamTomo


class LinearDefocusOutput(BaseModelTeamTomo):
    """Linear defocus model output: defocus_0 + gradient."""

    defocus_0: float
    defocus_gradient_magnitude: float
    defocus_gradient_angle: float
    defocus_0_spline_data: list[float] | None = None
    gradient_magnitude_spline_data: list[float] | None = None
    angle_u_spline_data: list[float] | None = None
    angle_v_spline_data: list[float] | None = None


class GridDefocusOutput(BaseModelTeamTomo):
    """Grid defocus model output: shape and values."""

    shape: list[int]
    values: list[Any]  # nested lists from .tolist()


class DefocusResultsOutput(BaseModelTeamTomo):
    """Defocus results: scalars plus either linear or grid model."""

    defocus_u: float
    defocus_v: float
    astigmatism_angle_deg: float | None = None
    defocus_model_type: Literal["grid", "linear"]
    linear_defocus: LinearDefocusOutput | None = None
    grid_defocus: GridDefocusOutput | None = None
    tilt_axis_angle_deg: float | None = None
    tilt_magnitude_deg: float | None = None


class PhaseShiftQuadraticOutput(BaseModelTeamTomo):
    """Quadratic phase shift coefficients (s,t) with perpendicular axis."""

    C: float
    alpha_rad: float
    g1: float
    k1: float
    g2: float
    k2: float


class PhaseShiftGridOutput(BaseModelTeamTomo):
    """Phase shift grid: u and v grids (each shape + values)."""

    grid_u: dict[str, Any]  # {"shape": [...], "values": [...]}
    grid_v: dict[str, Any]


class PhaseShiftParamsOutput(BaseModelTeamTomo):
    """Phase shift params: scalar plus either quadratic or grid model."""

    phase_shift_degrees: float
    phase_shift_model_type: Literal["grid", "quadratic"]
    quadratic: PhaseShiftQuadraticOutput | None = None
    grid: PhaseShiftGridOutput | None = None


class CTFResultsOutput(BaseModelTeamTomo):
    """
    Top-level CTF estimation results for JSON export.

    cross_correlation_final, when set, is the mean Pearson r from the 2D fit
    (heuristic reliability; not comparable to 1D L2 CC).
    """

    defocus_results: DefocusResultsOutput
    phase_shift_params: PhaseShiftParamsOutput | None = None
    envelope_B: float | None = None
    cross_correlation_final: float | None = None
