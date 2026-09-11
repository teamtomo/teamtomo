"""JSON serialization of CTF estimation results using teamtomo-basemodel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from torch_cubic_spline_grids import CubicCatmullRomGrid3d

from torch_ctf_estimation.models import (
    CTFResultsOutput,
    DefocusResultsOutput,
    GridDefocusOutput,
    LinearDefocusModel,
    LinearDefocusOutput,
    PhaseShiftGridOutput,
    PhaseShiftParamsOutput,
    PhaseShiftQuadraticOutput,
    QuadraticPhaseShiftModel,
)

if TYPE_CHECKING:
    from torch_ctf_estimation.models import Defocus2DResults


def _astigmatism_angle_to_m90_p90(angle_0_180: float | None) -> float | None:
    """Map astigmatism angle from [0, 180) to [-90, 90] for output."""
    if angle_0_180 is None:
        return None

    a = angle_0_180 % 180.0
    return a if a <= 90.0 else a - 180.0


def _tensor_to_list(x: Any) -> Any:
    """Convert tensor to JSON-serializable list; pass through other types."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().tolist()
    return x


# ---------------------------------------------------------------------------
# Conversion from Defocus2DResults (output models live in models.output_models)
# ---------------------------------------------------------------------------


def _grid_to_shape_and_values(
    grid: CubicCatmullRomGrid3d,
) -> tuple[list[int], list[Any]]:
    """Extract shape and JSON-serializable values from a 3D grid (nt, nh, nw)."""
    data = grid.data
    # Squeeze leading singular dimensions so we store (nt, nh, nw) not (1, nt, nh, nw)
    while data.dim() > 3 and data.shape[0] == 1:
        data = data.squeeze(0)
    shape = list(data.shape)
    values = data.detach().cpu().tolist()
    return shape, values


def results_to_output_model(result2d: Defocus2DResults) -> CTFResultsOutput:
    """
    Build CTFResultsOutput from Defocus2DResults for JSON export.

    Maps defocus (linear or grid), phase shift (quadratic or grid), and
    envelope_B. All fields are JSON-serializable.
    """
    # Defocus
    astig_angle = _astigmatism_angle_to_m90_p90(result2d.astigmatism_angle)
    linear_defocus: LinearDefocusOutput | None = None
    grid_defocus: GridDefocusOutput | None = None

    if result2d.defocus_model_type == "linear":
        lm = result2d.defocus_model
        if not isinstance(lm, LinearDefocusModel):
            raise TypeError(
                "Expected LinearDefocusModel when defocus_model_type is linear"
            )
        linear_defocus = LinearDefocusOutput(
            defocus_0=lm.defocus_0,
            defocus_gradient_magnitude=lm.defocus_gradient_magnitude,
            defocus_gradient_angle=lm.defocus_gradient_angle,
            defocus_0_spline_data=_tensor_to_list(lm.defocus_0_spline_data)
            if lm.defocus_0_spline_data is not None
            else None,
            gradient_magnitude_spline_data=_tensor_to_list(
                lm.gradient_magnitude_spline_data
            )
            if lm.gradient_magnitude_spline_data is not None
            else None,
            angle_u_spline_data=_tensor_to_list(lm.angle_u_spline_data)
            if lm.angle_u_spline_data is not None
            else None,
            angle_v_spline_data=_tensor_to_list(lm.angle_v_spline_data)
            if lm.angle_v_spline_data is not None
            else None,
        )
    else:
        grid = result2d.defocus_model
        if not isinstance(grid, CubicCatmullRomGrid3d):
            raise TypeError(
                "Expected CubicCatmullRomGrid3d when defocus_model_type is grid"
            )
        shape, values = _grid_to_shape_and_values(grid)
        grid_defocus = GridDefocusOutput(shape=shape, values=values)

    defocus_u = result2d.defocus_u if result2d.defocus_u is not None else 0.0
    defocus_v = result2d.defocus_v if result2d.defocus_v is not None else 0.0

    defocus_results = DefocusResultsOutput(
        defocus_u=defocus_u,
        defocus_v=defocus_v,
        astigmatism_angle_deg=astig_angle,
        defocus_model_type=result2d.defocus_model_type,
        linear_defocus=linear_defocus,
        grid_defocus=grid_defocus,
        tilt_axis_angle_deg=result2d.tilt_axis_angle_deg,
        tilt_magnitude_deg=result2d.tilt_magnitude_deg,
    )

    # Phase shift
    phase_shift_params: PhaseShiftParamsOutput | None = None
    if result2d.phase_shift_degrees is not None and result2d.phase_shift_model_type:
        model_type = result2d.phase_shift_model_type
        quad: PhaseShiftQuadraticOutput | None = None
        grid_ps: PhaseShiftGridOutput | None = None
        if result2d.phase_shift_model is not None:
            if isinstance(result2d.phase_shift_model, QuadraticPhaseShiftModel):
                q = result2d.phase_shift_model
                quad = PhaseShiftQuadraticOutput(
                    C=q.C,
                    alpha_rad=q.alpha_rad,
                    g1=q.g1,
                    k1=q.k1,
                    g2=q.g2,
                    k2=q.k2,
                )
            elif (
                isinstance(result2d.phase_shift_model, tuple)
                and len(result2d.phase_shift_model) == 2
            ):
                gu, gv = result2d.phase_shift_model
                su, vu = _grid_to_shape_and_values(gu)
                sv, vv = _grid_to_shape_and_values(gv)
                grid_ps = PhaseShiftGridOutput(
                    grid_u={"shape": su, "values": vu},
                    grid_v={"shape": sv, "values": vv},
                )
        phase_shift_params = PhaseShiftParamsOutput(
            phase_shift_degrees=result2d.phase_shift_degrees,
            phase_shift_model_type=model_type,
            quadratic=quad,
            grid=grid_ps,
        )

    # Envelope B
    envelope_B: float | None = None
    if result2d.envelope_B is not None:
        envelope_B = float(result2d.envelope_B)

    cc_final = result2d.cross_correlation_final

    return CTFResultsOutput(
        defocus_results=defocus_results,
        phase_shift_params=phase_shift_params,
        envelope_B=envelope_B,
        cross_correlation_final=cc_final,
    )


def write_results_json(
    result2d: Defocus2DResults,
    path: str | Path,
    indent: int = 2,
) -> None:
    """
    Write CTF estimation results to a JSON file.

    Parameters
    ----------
    result2d : Defocus2DResults
        The 2D defocus (and optional phase shift) result from estimate_ctf.
    path : str or Path
        Output file path for the JSON file.
    indent : int, optional
        JSON indent for pretty-printing. Default 2.
    """
    path = Path(path)
    output = results_to_output_model(result2d)
    data = output.model_dump()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def read_results_json(path: str | Path) -> CTFResultsOutput:
    """
    Load CTF results from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    CTFResultsOutput
        The deserialized results model.
    """
    return cast("CTFResultsOutput", CTFResultsOutput.from_json(path))


__all__ = [
    "CTFResultsOutput",
    "DefocusResultsOutput",
    "GridDefocusOutput",
    "LinearDefocusOutput",
    "PhaseShiftGridOutput",
    "PhaseShiftParamsOutput",
    "PhaseShiftQuadraticOutput",
    "read_results_json",
    "results_to_output_model",
    "write_results_json",
]
