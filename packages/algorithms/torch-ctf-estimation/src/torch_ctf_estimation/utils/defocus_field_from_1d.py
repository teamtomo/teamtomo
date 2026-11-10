"""Build defocus field from per-patch 1D CTF fits."""

import math
import warnings
from typing import Any, Literal

import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d

from torch_ctf_estimation.estimate_ctf_1d import estimate_ctf_1d
from torch_ctf_estimation.models import (
    Defocus2DResults,
    LaserParams,
    LinearDefocusModel,
)


def defocus_field_from_1d_fits(
    patch_power_spectra: torch.Tensor,
    normalised_patch_positions: torch.Tensor,
    result_1x1: Defocus2DResults,
    defocus_model: Literal["grid", "linear"],
    defocus_grid_resolution: tuple[int, int, int],
    initial_defocus: float,
    image_sidelength: int,
    frequency_fit_range_angstroms: tuple[float, float],
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast_fraction: float,
    pixel_spacing_angstroms: float,
    optimize_envelope_1d: bool,
    b_range_1d: tuple[float, float],
    b_step_1d: float,
    refine_steps_1d: int,
    background_result: Any | None,
    device: torch.device,
    optimize_phase_shift: bool = False,
    use_equiphase_for_1d_spatial: bool = False,
    laser_params: LaserParams | None = None,
    equiphase_n_theta: int = 64,
    defocus_range_microns: tuple[float, float] | None = None,
    phase_shift_range_degrees: tuple[float, float] | None = None,
    fixed_phase_shift_deg: float | None = None,
) -> Defocus2DResults:
    """
    Build defocus field from per-patch 1D fits; fit grid or linear to those values.

    Runs `estimate_ctf_1d` on each patch, then fits either a 3D spline grid or
    a linear (defocus_0 + gradient) model to the (position, defocus_1d) data.
    Astigmatism and envelope come from result_1x1.

    Each per-patch 1D fit starts from the mean defocus of `result_1x1`, so the
    1D grid search is skipped and only gradient-descent refinement of defocus
    (and optionally phase shift) runs, without envelope refinement.

    Parameters
    ----------
    patch_power_spectra : torch.Tensor
        Per-patch power spectra, shape `(t, gh, gw, ph, pw // 2 + 1)`.
    normalised_patch_positions : torch.Tensor
        Patch positions, shape `(t, gh, gw, 3)`, as normalised `(t, y, x)` in
        [0, 1].
    result_1x1 : Defocus2DResults
        Whole-image (1x1) 2D fit. Its mean defocus seeds every patch fit; its
        astigmatism, astigmatism angle, envelope B and (unless
        `fixed_phase_shift_deg` is set) phase shift are reused.
    defocus_model : {"grid", "linear"}
        Model fitted to the per-patch defocus values.
    defocus_grid_resolution : tuple[int, int, int]
        Grid resolution `(nt, nh, nw)` for the "grid" model. Unused for "linear".
    initial_defocus : float
        Initial value (µm) of every grid node before fitting the "grid" model.
        Unused for "linear", where `defocus_0` is the mean defocus of
        `result_1x1`.
    image_sidelength : int
        Side length in pixels of each (square) patch.
    frequency_fit_range_angstroms : tuple[float, float]
        `(low, high)` fit band in Å for the per-patch 1D fits.
    voltage_kev : float
        Acceleration voltage in keV.
    spherical_aberration_mm : float
        Spherical aberration in mm.
    amplitude_contrast_fraction : float
        Fraction of amplitude contrast.
    pixel_spacing_angstroms : float
        Pixel spacing in Å.
    optimize_envelope_1d : bool
        Passed to `estimate_ctf_1d` as `optimize_envelope`. Has no effect here
        because the per-patch fits always skip the grid search.
    b_range_1d : tuple[float, float]
        Passed to `estimate_ctf_1d` as `b_range` (B-factor range in Å²). Has no
        effect here for the same reason.
    b_step_1d : float
        Passed to `estimate_ctf_1d` as `b_step`. Has no effect here for the
        same reason.
    refine_steps_1d : int
        Number of gradient-descent refinement steps per patch.
    background_result : _Background1DResult or None
        Pre-fitted 1D background (e.g. from the mean spectrum) to reuse for every
        patch. If None, a background is fitted per patch.
    device : torch.device
        Device for the per-patch defocus values and the fitted defocus model.
    optimize_phase_shift : bool, optional
        If True, refine the phase shift in each per-patch fit. Default False.
    use_equiphase_for_1d_spatial : bool, optional
        If True, the per-patch 1D spectra use equiphase averaging with the
        astigmatism of `result_1x1`. Default False.
    laser_params : LaserParams or None, optional
        Laser phase plate parameters for the equiphase phase calculation.
        Default None.
    equiphase_n_theta : int, optional
        Azimuth samples per shell for equiphase averaging. Default 64.
    defocus_range_microns : tuple[float, float] or None, optional
        Defocus bounds in µm for the per-patch fits. None uses the default
        bounds (0, 10). Default None.
    phase_shift_range_degrees : tuple[float, float] or None, optional
        Phase shift bounds in degrees for the per-patch fits. None uses the
        default bounds (0, 180). Default None.
    fixed_phase_shift_deg : float or None, optional
        If set, used as the initial (and equiphase) phase shift for every patch
        instead of the phase shift of `result_1x1`. Default None.

    Returns
    -------
    Defocus2DResults
        Result with `defocus_model_type` set to `defocus_model`, the fitted
        `CubicCatmullRomGrid3d` or `LinearDefocusModel`, `patch_power_spectra`,
        astigmatism and envelope B copied from `result_1x1`, and `defocus_u` /
        `defocus_v` from the mean defocus ± astigmatism / 2.

    Warns
    -----
    UserWarning
        For the linear model, if the fitted defocus gradient is zero or NaN.
    """
    t, gh, gw, _ph, _pw = patch_power_spectra.shape
    nt, nh, nw = defocus_grid_resolution
    defocus_2d_center = float(result_1x1.defocus_model.data.mean().cpu().item())
    if fixed_phase_shift_deg is not None:
        equiphase_phase_shift = fixed_phase_shift_deg
        patch_initial_phase = fixed_phase_shift_deg
    else:
        initial_phase_from_1x1 = 0.0
        if result_1x1.phase_shift_degrees is not None:
            initial_phase_from_1x1 = result_1x1.phase_shift_degrees
        if isinstance(initial_phase_from_1x1, torch.Tensor):
            equiphase_phase_shift = float(initial_phase_from_1x1.cpu().item())
        else:
            equiphase_phase_shift = float(initial_phase_from_1x1)
        patch_initial_phase = equiphase_phase_shift
    astig_um = float(result_1x1.astigmatism or 0.0)
    astig_angle = float(result_1x1.astigmatism_angle or 0.0)
    defocus_list = []
    for ti in range(t):
        for gi in range(gh):
            for gj in range(gw):
                ps = patch_power_spectra[ti, gi, gj]
                r1d = estimate_ctf_1d(
                    power_spectrum=ps,
                    image_sidelength=image_sidelength,
                    frequency_fit_range_angstroms=frequency_fit_range_angstroms,
                    defocus_range_microns=defocus_range_microns,
                    voltage_kev=voltage_kev,
                    spherical_aberration_mm=spherical_aberration_mm,
                    amplitude_contrast=amplitude_contrast_fraction,
                    pixel_spacing_angstroms=pixel_spacing_angstroms,
                    optimize_envelope=optimize_envelope_1d,
                    b_range=b_range_1d,
                    b_step=b_step_1d,
                    refine_steps=refine_steps_1d,
                    initial_defocus=defocus_2d_center,
                    background_result=background_result,
                    optimize_phase_shift=optimize_phase_shift,
                    initial_phase_shift=patch_initial_phase,
                    phase_shift_range=phase_shift_range_degrees,
                    use_equiphase=use_equiphase_for_1d_spatial,
                    equiphase_defocus_um=defocus_2d_center,
                    equiphase_astigmatism_um=astig_um,
                    equiphase_astigmatism_angle_deg=astig_angle,
                    equiphase_phase_shift_deg=equiphase_phase_shift,
                    laser_params=laser_params,
                    equiphase_n_theta=equiphase_n_theta,
                )
                d = r1d.ctf_model.defocus_um
                if isinstance(d, torch.Tensor):
                    d = float(d.cpu().item())
                else:
                    d = float(d)
                defocus_list.append(d)
    defocus_vals = torch.tensor(
        defocus_list, device=device, dtype=patch_power_spectra.dtype
    ).view(t, gh, gw)
    positions_flat = normalised_patch_positions.reshape(-1, 3)
    defocus_flat = defocus_vals.reshape(-1, 1)

    astig = result_1x1.astigmatism or 0.0
    env_b = result_1x1.envelope_B
    envelope_B = float(env_b) if env_b is not None else None

    if defocus_model == "linear":
        defocus_0 = float(result_1x1.defocus_model.data.mean().cpu().item())
        design = torch.stack(
            [
                positions_flat[:, 1] - 0.5,
                positions_flat[:, 2] - 0.5,
            ],
            dim=1,
        )
        target = (defocus_flat.squeeze(1) - defocus_0).to(torch.float64).unsqueeze(1)
        design = design.to(torch.float64)
        sol = (torch.linalg.pinv(design) @ target).squeeze(1)  # pylint: disable=no-member
        # Replace NaN from singular/rank-deficient design with 0 for debuggable result
        sol = torch.nan_to_num(sol, nan=0.0, posinf=0.0, neginf=0.0)
        u = float(sol[0].item())
        v = float(sol[1].item()) if sol.numel() > 1 else 0.0
        if math.isnan(u):
            u = 0.0
        if math.isnan(v):
            v = 0.0
        grad_mag = math.sqrt(u * u + v * v)
        if math.isnan(grad_mag) or grad_mag <= 0.0:
            grad_mag = 0.0
            angle_deg = 0.0
            warnings.warn(
                "Linear defocus gradient from 1D fits is zero or NaN (e.g. singular "
                "design matrix or no spatial defocus variation). Check patch positions "
                "and per-patch defocus values.",
                UserWarning,
                stacklevel=2,
            )
        else:
            angle_rad = math.atan2(v, u)
            angle_deg = (angle_rad * 180.0 / math.pi + 180.0) % 180.0
            if math.isnan(angle_deg):
                angle_deg = 0.0
                warnings.warn(
                    "Linear defocus gradient angle was NaN; set to 0.",
                    UserWarning,
                    stacklevel=2,
                )
        linear_model = LinearDefocusModel(
            defocus_0=defocus_0,
            defocus_gradient_magnitude=grad_mag,
            defocus_gradient_angle=angle_deg,
        )
        mean_defocus = defocus_0
        return Defocus2DResults(
            defocus_model_type="linear",
            defocus_model=linear_model,
            patch_power_spectra=patch_power_spectra,
            astigmatism=astig,
            astigmatism_angle=result_1x1.astigmatism_angle or 0.0,
            envelope_B=envelope_B,
            defocus_u=mean_defocus + astig / 2.0,
            defocus_v=mean_defocus - astig / 2.0,
        )
    # grid: fit 3D spline to (positions, defocus_flat)
    grid_data = (
        torch.ones((nt, nh, nw), device=device, dtype=patch_power_spectra.dtype)
        * initial_defocus
    )
    grid_model = CubicCatmullRomGrid3d.from_grid_data(grid_data).to(device)
    optimiser = torch.optim.Adam(grid_model.parameters(), lr=0.01)
    n_fit_steps = 100
    for _ in range(n_fit_steps):
        optimiser.zero_grad()
        pred = grid_model(positions_flat).squeeze(-1)
        loss = ((pred - defocus_flat.squeeze(1)) ** 2).mean()
        loss.backward()
        optimiser.step()
    mean_defocus = float(grid_model.data.mean().cpu().item())
    return Defocus2DResults(
        defocus_model_type="grid",
        defocus_model=grid_model,
        patch_power_spectra=patch_power_spectra,
        astigmatism=astig,
        astigmatism_angle=result_1x1.astigmatism_angle or 0.0,
        envelope_B=envelope_B,
        defocus_u=mean_defocus + astig / 2.0,
        defocus_v=mean_defocus - astig / 2.0,
    )


_defocus_field_from_1d_fits = defocus_field_from_1d_fits
