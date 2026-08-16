"""
2D thickness estimation using a 3D spline grid over (t, x, y).

Thickness is represented as a cubic spline over the normalised (t, x, y) patch
grid.  Defocus is held fixed (scalar or a pre-fitted CubicCatmullRomGrid3d from a
prior 2D defocus estimation) while the Adam optimiser fits the thickness control
points to maximise the zero-normalised cross-correlation between the observed
patch power spectra and the thickness-modulated power spectrum.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

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
    _get_astig_clamped,
    _shared_astigmatism_and_env,
)
from torch_ctf_estimation.metrics.fit_metrics import pearson_r_flat
from torch_ctf_estimation.models import LaserParams, Thickness2DResults


def _eval_defocus_at_positions(
    defocus: float | CubicCatmullRomGrid3d,
    positions_t: torch.Tensor,
) -> torch.Tensor:
    """Return defocus values at ``positions_t`` for one time frame.

    Parameters
    ----------
    defocus : float | CubicCatmullRomGrid3d
        Either a scalar (uniform defocus, µm) or a pre-fitted 3D spline.
    positions_t : torch.Tensor
        Normalised patch positions, shape (gh, gw, 3).

    Returns
    -------
    torch.Tensor
        Defocus tensor, shape (gh, gw).
    """
    if isinstance(defocus, CubicCatmullRomGrid3d):
        return einops.rearrange(defocus(positions_t).detach(), "... 1 -> ...")
    device = positions_t.device
    gh, gw, _ = positions_t.shape
    return torch.full((gh, gw), float(defocus), device=device, dtype=positions_t.dtype)


def estimate_thickness_2d(
    patch_power_spectra: torch.Tensor,
    normalised_patch_positions: torch.Tensor,
    thickness_grid_resolution: tuple[int, int, int],
    frequency_fit_range_angstroms: tuple[float, float],
    initial_thickness: float,
    defocus: float | CubicCatmullRomGrid3d,
    astigmatism: float = 0.0,
    astigmatism_angle: float = 0.0,
    pixel_spacing_angstroms: float = 1.0,
    phase_shift_deg: float = 0.0,
    initial_envelope_B: float = 0.0,
    n_iterations: int = 100,
    thickness_lr: float = 50.0,
    voltage_kev: float = 300.0,
    spherical_aberration_mm: float = 2.7,
    amplitude_contrast_fraction: float = 0.10,
    laser_params: LaserParams | None = None,
    debug: bool = False,
    early_stopper: Callable[[float], bool] | None = None,
) -> Thickness2DResults:
    """
    Estimate sample thickness in 2D using a 3D spline grid over (t, x, y).

    Defocus is fixed throughout; only the thickness spline is optimised.

    Parameters
    ----------
    patch_power_spectra : torch.Tensor
        Shape ``(t, gh, gw, ph, pw_rfft)`` — per-patch rfft power spectra.
    normalised_patch_positions : torch.Tensor
        Shape ``(t, gh, gw, 3)`` — normalised (t, y, x) positions in [0, 1].
    thickness_grid_resolution : tuple[int, int, int]
        ``(nt, nh, nw)`` control-point resolution for the thickness spline.
    frequency_fit_range_angstroms : tuple[float, float]
        ``(low, high)`` spatial frequency cutoffs for fitting in Angstroms.
    initial_thickness : float
        Starting thickness in Angstroms for the spline grid.
    defocus : float | CubicCatmullRomGrid3d
        Fixed defocus in micrometers.  Pass a scalar for uniform defocus or a
        pre-fitted ``CubicCatmullRomGrid3d`` (from a prior 2D defocus fit) for
        spatially / temporally varying defocus.
    astigmatism : float
        Fixed astigmatism in micrometers. Default 0.0.
    astigmatism_angle : float
        Fixed astigmatism angle in degrees. Default 0.0.
    pixel_spacing_angstroms : float
        Pixel size in Angstroms. Default 1.0.
    phase_shift_deg : float
        Fixed phase shift in degrees. Default 0.0.
    initial_envelope_B : float
        Initial B-factor for the envelope (Å²). Default 0.0 (no envelope).
    n_iterations : int
        Number of Adam optimiser steps. Default 100.
    thickness_lr : float
        Adam learning rate for the thickness spline control points.  Units are
        Angstroms; a value of ~50 is appropriate for 100 Å-scale changes.
        Default 50.0.
    voltage_kev : float
        Acceleration voltage in keV. Default 300.0.
    spherical_aberration_mm : float
        Spherical aberration in mm. Default 2.7.
    amplitude_contrast_fraction : float
        Amplitude contrast fraction. Default 0.10.
    laser_params : LaserParams, optional
        If set, use the LPP thickness CTF model. Default None.
    debug : bool
        If True, include per-iteration thickness model traces and final
        simulated power spectrum in the result. Default False.
    early_stopper : callable or None, optional
        Stateful ``(loss) -> should_stop`` callback. Default None (run all
        ``n_iterations``).

    Returns
    -------
    Thickness2DResults
        Fitted thickness spline, mean thickness, loss trace, and (if debug)
        per-iteration model traces.
    """
    # --- Setup: collapse time dim when nt==1, derive image shape ---
    ph, pw_rfft = patch_power_spectra.shape[-2], patch_power_spectra.shape[-1]
    image_shape = (ph, (pw_rfft - 1) * 2)
    device = patch_power_spectra.device
    nt = thickness_grid_resolution[0]
    if nt == 1:
        patch_power_spectra = einops.reduce(
            patch_power_spectra, "t ... -> 1 ...", reduction="mean"
        )
    T = patch_power_spectra.shape[0]

    # --- Thickness spline grid ---
    thickness_grid_data = (
        torch.ones(size=thickness_grid_resolution, device=device) * initial_thickness
    )
    thickness_model = CubicCatmullRomGrid3d.from_grid_data(thickness_grid_data).to(
        device
    )

    # --- Bandpass filter, (fixed) astigmatism tensors, and B-factor envelope ---
    (
        bp_filter,
        astigmatism_t,
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
        initial_astigmatism=astigmatism,
        initial_astigmatism_angle=astigmatism_angle,
        optimize_astigmatism=False,
        initial_envelope_B=initial_envelope_B,
    )
    patch_power_spectra = patch_power_spectra * bp_filter

    astig_clamped, astig_angle_clamped = _get_astig_clamped(
        astigmatism_t, angle_u, angle_v, optimize_astigmatism=False
    )

    # --- Optimiser ---
    optimiser = torch.optim.Adam(params=thickness_model.parameters(), lr=thickness_lr)

    thickness_models: list[torch.Tensor] = []
    loss_trace: list[float] = []
    simulated_ps_last = None

    # --- Optimisation loop ---
    for _ in range(n_iterations):
        optimiser.zero_grad()
        loss_t_list: list[torch.Tensor] = []

        for t_idx in range(T):
            patch_ps_t = patch_power_spectra[t_idx]
            positions_t = normalised_patch_positions[t_idx]

            # Thickness from spline
            predicted_thickness_t = einops.rearrange(
                thickness_model(positions_t), "... 1 -> ..."
            )
            # Defocus fixed (detached scalar broadcast or spline eval)
            defocus_t = _eval_defocus_at_positions(defocus, positions_t)

            simulated_ps_t = compute_thickness_ctf_ps_t(
                thickness_t=predicted_thickness_t,
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
            simulated_ps_last = simulated_ps_t

            if torch.isnan(simulated_ps_t).any() or torch.isinf(simulated_ps_t).any():
                continue

            loss_t = correlation_loss_t(simulated_ps_t, patch_ps_t)
            loss_t_list.append(loss_t)

        if len(loss_t_list) == 0:
            continue

        total_loss = sum(loss_t_list) / T
        total_loss.backward()
        mean_loss = (sum(loss_t_list) / len(loss_t_list)).detach().cpu().item()
        if math.isnan(mean_loss) or math.isinf(mean_loss):
            continue

        loss_trace.append(float(mean_loss))
        optimiser.step()
        if early_stopper is not None and early_stopper(loss_trace[-1]):
            break

        # Clamp thickness to physically plausible range (> 0)
        with torch.no_grad():
            thickness_model.data.clamp_(min=1.0)

        if debug:
            thickness_models.append(thickness_model.data.detach().clone())

    # --- Final cross-correlation (mean Pearson r) ---
    mean_thickness = float(thickness_model.data.detach().cpu().mean().item())
    rs: list[float] = []
    with torch.no_grad():
        for t_idx in range(T):
            positions_t = normalised_patch_positions[t_idx]
            predicted_thickness_t = einops.rearrange(
                thickness_model(positions_t), "... 1 -> ..."
            )
            defocus_t = _eval_defocus_at_positions(defocus, positions_t)
            sim_ps_t = compute_thickness_ctf_ps_t(
                thickness_t=predicted_thickness_t,
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
            if torch.isnan(sim_ps_t).any() or torch.isinf(sim_ps_t).any():
                continue
            rs.append(
                pearson_r_flat(
                    patch_power_spectra[t_idx].reshape(-1),
                    sim_ps_t.reshape(-1),
                )
            )
    cc_final: float | None = float(sum(rs) / len(rs)) if rs else None

    if debug:
        return Thickness2DResults(
            mean_thickness=mean_thickness,
            cross_correlation_final=cc_final,
            thickness_model=thickness_model,
            patch_power_spectra=patch_power_spectra,
            model_trace=thickness_models,
            simulated_ps=simulated_ps_last,
            envelope_B=float(envelope_B.detach().cpu().item()),
            loss_trace=loss_trace,
        )
    return Thickness2DResults(
        mean_thickness=mean_thickness,
        cross_correlation_final=cc_final,
        thickness_model=thickness_model,
        envelope_B=float(envelope_B.detach().cpu().item()),
        loss_trace=loss_trace,
    )
