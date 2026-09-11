"""Shared utilities for 2D CTF estimation (astigmatism, bandpass, 1x1 helper)."""

import math
from collections.abc import Callable
from typing import Literal, Optional

import torch
from torch_fourier_filter.bandpass import bandpass_filter
from torch_fourier_filter.envelopes import b_envelope
from torch_grid_utils.fftfreq_grid import spatial_frequency_to_fftfreq

from torch_ctf_estimation.models import Defocus2DResults, LaserParams


def _astig_angle_to_m90_p90(angle_0_180: float) -> float:
    """Map astigmatism angle from [0, 180) to [-90, 90] for output."""
    a = angle_0_180 % 180.0
    return a if a <= 90.0 else a - 180.0


def _clamp_optional_bounds(
    values: torch.Tensor,
    bounds: tuple[float, float] | None,
) -> torch.Tensor:
    """Clamp ``values`` to ``bounds`` when set; otherwise return unchanged."""
    if bounds is None:
        return values
    lo, hi = bounds
    return values.clamp(min=lo, max=hi)


def _clamp_defocus_grid_after_step(
    defocus_model_obj: torch.nn.Module,
    defocus_bounds_microns: tuple[float, float] | None,
) -> None:
    """Clamp defocus spline grid control values after an optimizer step."""
    if defocus_bounds_microns is None:
        return
    lo, hi = defocus_bounds_microns
    with torch.no_grad():
        defocus_model_obj.data.clamp_(min=lo, max=hi)


def _shared_astigmatism_and_env(
    *,
    image_shape: tuple[int, int],
    device: torch.device,
    frequency_fit_range_angstroms: tuple[float, float],
    pixel_spacing_angstroms: float,
    initial_astigmatism: float,
    initial_astigmatism_angle: float,
    optimize_astigmatism: bool,
    initial_envelope_B: float,
    axis_mask: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
    torch.Tensor,
    torch.Tensor,
]:
    """Build bandpass filter, astigmatism params, and B-factor envelope."""
    low_ang, high_ang = frequency_fit_range_angstroms
    low_fftfreq = spatial_frequency_to_fftfreq(
        1 / low_ang, spacing=pixel_spacing_angstroms
    )
    high_fftfreq = spatial_frequency_to_fftfreq(
        1 / high_ang, spacing=pixel_spacing_angstroms
    )
    bp_filter = bandpass_filter(
        low=low_fftfreq,
        high=high_fftfreq,
        falloff=0,
        image_shape=image_shape,
        rfft=True,
        fftshift=False,
        device=device,
    )
    if axis_mask is not None:
        bp_filter = bp_filter * axis_mask.to(device=device, dtype=bp_filter.dtype)
    _angle_rad = initial_astigmatism_angle * math.pi / 180.0
    _angle_u_init = math.cos(_angle_rad)
    _angle_v_init = math.sin(_angle_rad)
    if optimize_astigmatism:
        init_astig = initial_astigmatism if initial_astigmatism > 0 else 0.05
        astigmatism = torch.nn.Parameter(torch.tensor(init_astig, device=device))
        angle_u = torch.nn.Parameter(torch.tensor(_angle_u_init, device=device))
        angle_v = torch.nn.Parameter(torch.tensor(_angle_v_init, device=device))
    else:
        astigmatism = torch.tensor(initial_astigmatism, device=device)
        angle_u = torch.tensor(_angle_u_init, device=device)
        angle_v = torch.tensor(_angle_v_init, device=device)
    envelope_B = torch.tensor(initial_envelope_B, device=device)
    env_2d = b_envelope(
        B=envelope_B,
        image_shape=image_shape,
        pixel_size=pixel_spacing_angstroms,
        rfft=True,
        fftshift=False,
        device=device,
    )
    return (
        bp_filter,
        astigmatism,
        angle_u,
        angle_v,
        _angle_u_init,
        _angle_v_init,
        envelope_B,
        env_2d,
    )


def _get_astig_clamped(
    astigmatism: torch.Tensor,
    angle_u: torch.Tensor,
    angle_v: torch.Tensor,
    optimize_astigmatism: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (astig_clamped, astig_angle_clamped) in degrees [0, 180)."""
    if optimize_astigmatism:
        astig_clamped = torch.clamp(astigmatism, min=1e-6)
        _eps = 1e-8
        _norm = torch.sqrt(angle_u**2 + angle_v**2 + _eps)
        _dir_u = angle_u / _norm
        _dir_v = angle_v / _norm
        _angle_rad = torch.atan2(_dir_v, _dir_u)
        _angle_deg = _angle_rad * (180.0 / math.pi)
        astig_angle_clamped = torch.remainder(_angle_deg + 180.0, 180.0)
    else:
        astig_clamped = astigmatism
        _angle_rad = torch.atan2(angle_v, angle_u)
        _angle_deg = _angle_rad * (180.0 / math.pi)
        astig_angle_clamped = torch.remainder(_angle_deg + 180.0, 180.0)
    return astig_clamped, astig_angle_clamped


def _reset_astigmatism(
    astigmatism: torch.Tensor,
    angle_u: torch.Tensor,
    angle_v: torch.Tensor,
    initial_astigmatism: float,
    _angle_u_init: float,
    _angle_v_init: float,
) -> None:
    """Reset astigmatism params to initial values (in-place)."""
    with torch.no_grad():
        astigmatism.fill_(initial_astigmatism if initial_astigmatism > 0 else 0.05)
        angle_u.fill_(_angle_u_init)
        angle_v.fill_(_angle_v_init)


def _check_astig_grad_and_reset(
    astigmatism: torch.Tensor,
    angle_u: torch.Tensor,
    angle_v: torch.Tensor,
    initial_astigmatism: float,
    _angle_u_init: float,
    _angle_v_init: float,
) -> bool:
    """If astigmatism or angle_u/angle_v have NaN/Inf grads, reset and return True."""
    if astigmatism.grad is not None and (
        torch.isnan(astigmatism.grad).any() or torch.isinf(astigmatism.grad).any()
    ):
        _reset_astigmatism(
            astigmatism,
            angle_u,
            angle_v,
            initial_astigmatism,
            _angle_u_init,
            _angle_v_init,
        )
        return True
    if (
        angle_u.grad is not None
        and (torch.isnan(angle_u.grad).any() or torch.isinf(angle_u.grad).any())
    ) or (
        angle_v.grad is not None
        and (torch.isnan(angle_v.grad).any() or torch.isinf(angle_v.grad).any())
    ):
        _reset_astigmatism(
            astigmatism,
            angle_u,
            angle_v,
            initial_astigmatism,
            _angle_u_init,
            _angle_v_init,
        )
        return True
    return False


def estimate_defocus_2d_at_1x1(
    patch_power_spectra: torch.Tensor,
    defocus_grid_resolution: tuple[int, int, int],
    frequency_fit_range_angstroms: tuple[float, float],
    initial_defocus: float,
    pixel_spacing_angstroms: float,
    optimize_astigmatism: bool = False,
    initial_envelope_B: float = 0.0,
    n_iterations: int = 100,
    debug: bool = False,
    optimize_phase_shift: bool = False,
    phase_shift_model: Literal["grid", "quadratic"] = "grid",
    phase_shift_quadratic_perpendicular_axis: bool = False,
    initial_phase_shift: float = 0.0,
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
    Run 2D defocus estimation at 1x1 spatial resolution (center only).

    Averages patch power spectra over the spatial grid (gh, gw), builds
    single center positions per frame, and calls estimate_defocus_2d_grid with
    defocus_grid_resolution=(nt, 1, 1) to get astigmatism and center defocus.

    Parameters
    ----------
    patch_power_spectra : torch.Tensor
        Shape (t, gh, gw, ph, pw).
    defocus_grid_resolution : tuple[int, int, int]
        (nt, nh, nw); nt is used for the time dimension.
    frequency_fit_range_angstroms, initial_defocus, pixel_spacing_angstroms :
        Passed through to estimate_defocus_2d_grid.
    optimize_astigmatism : bool
        Whether to optimize astigmatism in the 2D fit.
    initial_envelope_B : float
        Initial B-factor for envelope.
    n_iterations : int, optional
        Number of optimizer steps for 2D fit. Default 100.
    debug : bool
        If True, return debug info from 2D fit.
    optimize_phase_shift : bool, optional
        Whether to estimate phase shift in the 2D fit. Default False.
    phase_shift_model : {"grid", "quadratic"}, optional
        Phase shift model passed to estimate_defocus_2d_grid. Default "grid".
    phase_shift_quadratic_perpendicular_axis : bool, optional
        If True and quadratic phase model, fit perpendicular-axis g2, k2. Default False.
    initial_phase_shift : float, optional
        Initial phase shift in degrees when optimizing. Default 0.0.
    voltage_kev : float, optional
        Acceleration voltage in keV for CTF simulation. Default 300.0.
    spherical_aberration_mm : float, optional
        Spherical aberration in mm for CTF simulation. Default 2.7.
    amplitude_contrast_fraction : float, optional
        Amplitude contrast fraction (0-1) for CTF simulation. Default 0.07.
    laser_params : Optional[LaserParams], optional
        If set and ``model_laser`` is True, use LPP CTF model for 2D fit; if None
        or ``model_laser`` is False, use standard CTF. Default None.
    early_stopper : callable or None, optional
        Stateful ``(loss) -> should_stop`` callback passed through to the 1x1
        grid fit. Default None (run all ``n_iterations``).

    Returns
    -------
    Defocus2DResults
        Result from 2D fit at 1x1 (defocus, astigmatism, envelope_B, etc.).
    """
    t, _gh, _gw, _ph, _pw = patch_power_spectra.shape
    nt = defocus_grid_resolution[0]
    device = patch_power_spectra.device
    # Mean over spatial patch grid -> (t, ph, pw)
    patch_ps_mean = patch_power_spectra.mean(dim=(1, 2))
    # (t, 1, 1, ph, pw)
    patch_ps_1x1 = patch_ps_mean.unsqueeze(1).unsqueeze(1)
    # Positions: (t, 1, 1, 3) with [t_norm, 0.5, 0.5]
    if t == 1:
        t_vals = torch.tensor([0.5], device=device, dtype=patch_power_spectra.dtype)
    else:
        t_vals = torch.linspace(0, 1, t, device=device, dtype=patch_power_spectra.dtype)
    positions_1x1 = torch.zeros(
        t, 1, 1, 3, device=device, dtype=patch_power_spectra.dtype
    )
    positions_1x1[:, 0, 0, 0] = t_vals
    positions_1x1[:, 0, 0, 1] = 0.5
    positions_1x1[:, 0, 0, 2] = 0.5
    from torch_ctf_estimation.estimate_ctf_2d.estimate_defocus_2d_grid import (
        estimate_defocus_2d_grid,
    )

    return estimate_defocus_2d_grid(
        patch_power_spectra=patch_ps_1x1,
        normalised_patch_positions=positions_1x1,
        defocus_grid_resolution=(nt, 1, 1),
        frequency_fit_range_angstroms=frequency_fit_range_angstroms,
        initial_defocus=initial_defocus,
        pixel_spacing_angstroms=pixel_spacing_angstroms,
        initial_astigmatism=0.0,
        initial_astigmatism_angle=0.0,
        optimize_astigmatism=optimize_astigmatism,
        initial_envelope_B=initial_envelope_B,
        n_iterations=n_iterations,
        debug=debug,
        optimize_phase_shift=optimize_phase_shift,
        phase_shift_model=phase_shift_model,
        phase_shift_quadratic_perpendicular_axis=phase_shift_quadratic_perpendicular_axis,
        initial_phase_shift=initial_phase_shift,
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


_estimate_defocus_2d_at_1x1 = estimate_defocus_2d_at_1x1
