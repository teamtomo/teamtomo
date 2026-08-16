"""Resolve defocus and phase-shift fitting bounds from CTFFittingParams."""

from __future__ import annotations

DEFAULT_DEFOCUS_BOUNDS_MICRONS: tuple[float, float] = (0.0, 10.0)
DEFAULT_PHASE_SHIFT_BOUNDS_DEG: tuple[float, float] = (0.0, 180.0)


def resolve_defocus_bounds(
    defocus_range_microns: tuple[float, float] | None,
) -> tuple[float, float]:
    """Return user bounds or the default (0, 10) µm."""
    if defocus_range_microns is not None:
        return defocus_range_microns
    return DEFAULT_DEFOCUS_BOUNDS_MICRONS


def resolve_phase_shift_bounds(
    phase_shift_range_degrees: tuple[float, float] | None,
) -> tuple[float, float]:
    """Return user bounds or the default (0, 180) degrees."""
    if phase_shift_range_degrees is not None:
        return phase_shift_range_degrees
    return DEFAULT_PHASE_SHIFT_BOUNDS_DEG


def bounds_are_fixed(
    bounds: tuple[float, float],
    *,
    tol: float = 1e-9,
) -> bool:
    """True when lower and upper bound are equal (fixed parameter value)."""
    return abs(bounds[0] - bounds[1]) <= tol


def resolve_phase_shift_fitting(
    *,
    optimize_phase_shift: bool,
    phase_shift_range_degrees: tuple[float, float] | None,
    initial_phase_shift: float,
) -> tuple[bool, float, tuple[float, float]]:
    """
    Resolve phase-shift optimisation mode, value, and effective bounds.

    When bounds are equal (e.g. ``(45.0, 45.0)``), phase is fixed at that value
    and ``optimize_phase_shift`` is forced to False.

    Returns
    -------
    optimize_phase_shift : bool
    phase_shift_deg : float
        Phase shift used in the CTF (fixed value or initial for optimisation).
    phase_bounds : tuple[float, float]
        Effective bounds (always set; defaults to 0–180°).
    """
    phase_bounds = resolve_phase_shift_bounds(phase_shift_range_degrees)
    if bounds_are_fixed(phase_bounds):
        return False, phase_bounds[0], phase_bounds
    if optimize_phase_shift:
        return True, initial_phase_shift, phase_bounds
    return False, initial_phase_shift, phase_bounds
