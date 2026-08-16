"""Tilt-corrected mean power spectrum (CTFFIND5-style, Elferich et al., eLife 2024).

Per-tile 2D rFFT power is **resampled isotropically in spatial frequency** so Thon rings
line up at a reference defocus: with ``m = sqrt(|Δf_local/Δf_average|)`` (thin-sample
phase: χ ∝ Δf q²), use ``k_in = k_out * sqrt(|Δf_average/Δf_local|)`` when reading the
patch spectrum onto the common output grid (same convention as plotting each patch 1D
profile vs ``q * m``). Bilinear interpolation uses
:func:`torch_image_interpolation.sample_image_2d`.

``effective_pixel_spacing_tilt_magnification`` gives ``Δx' = Δx / m``, i.e. the pixel
that would assign physical frequency ``m * q_bin`` to FFT bin ``j`` at spacing ``Δx``.

The 1D API averages tilt-corrected 2D spectra, then applies equiphase or rotational
averaging once on the mean 2D spectrum (defocus reference ``Δf_average``).
"""

from __future__ import annotations

import math
from typing import TypedDict

import torch
from torch_cubic_spline_grids import CubicCatmullRomGrid3d
from torch_fourier_filter.dft_utils import rotational_average_dft_2d
from torch_image_interpolation import sample_image_2d

from torch_ctf_estimation.estimate_ctf_1d.equiphase_ctf_1d import (
    equiphase_average_power_to_1d_rfft,
)
from torch_ctf_estimation.models import (
    Defocus1DResults,
    Defocus2DResults,
    LaserParams,
    LinearDefocusModel,
    OpticalParams,
)


def _tensor_to_float(x: float | torch.Tensor) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


class TiltCorrectedPsAux(TypedDict, total=False):
    """Optional diagnostics from tilt-corrected averaging."""

    defocus_local_um: torch.Tensor
    defocus_average_um: float
    m: torch.Tensor


def effective_pixel_spacing_tilt_magnification(
    base_pixel_spacing_angstroms: float,
    *,
    defocus_local_um: float,
    defocus_average_um: float,
    eps: float = 1e-12,
) -> float:
    """Effective sample spacing (Å) for CTF at reference defocus: ``Δx' = Δx / m``.

    With ``m = sqrt(|Δf_local/Δf_avg|)`` (stabilised near zero), using
    ``Δx' = Δx / m`` matches **multiply pixel size by ``1/m``**: the physical
    frequency at FFT bin ``j`` is ``fftfreq(j)/Δx' = m * fftfreq(j)/Δx``.

    Parameters
    ----------
    base_pixel_spacing_angstroms
        True spacing used to form the patch and compute ``rfft``.
    defocus_local_um, defocus_average_um
        Local and reference defocus (µm); ratio stabilised near zero.
    eps
        Floor on ``|Δf_avg|`` for division.
    """
    d_avg = defocus_average_um
    d_loc = defocus_local_um
    if abs(d_avg) < eps:
        d_avg = eps if d_avg >= 0.0 else -eps
    m = math.sqrt(abs(d_loc / d_avg) + eps)
    return base_pixel_spacing_angstroms / m


def _iy_float_from_ky_vectorized(
    ky: torch.Tensor,
    h: int,
    pixel_spacing: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map spatial frequency ky (cycles/Å) to fractional row index in rFFT layout."""
    fy = torch.fft.fftfreq(h, d=pixel_spacing, device=device, dtype=dtype)
    iy = torch.arange(h, device=device, dtype=dtype)
    order = torch.argsort(fy)
    fy_s = fy[order]
    iy_s = iy[order]
    ky_flat = ky.reshape(-1)
    idx = torch.searchsorted(fy_s, ky_flat)
    idx = idx.clamp(1, h - 1)
    f0 = fy_s[idx - 1]
    f1 = fy_s[idx]
    t = (ky_flat - f0) / (f1 - f0 + 1e-20)
    t = t.clamp(0.0, 1.0)
    iy0 = iy_s[idx - 1].to(dtype)
    iy1 = iy_s[idx].to(dtype)
    out = iy0 * (1.0 - t) + iy1 * t
    return out.reshape(ky.shape)


def _ix_float_from_kx_vectorized(
    kx: torch.Tensor,
    w: int,
    w_rfft: int,
    pixel_spacing: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map spatial frequency kx (cycles/Å) to fractional rFFT column index."""
    fx = torch.fft.rfftfreq(w, d=pixel_spacing, device=device, dtype=dtype)
    ix = torch.arange(w_rfft, device=device, dtype=dtype)
    order = torch.argsort(fx)
    fx_s = fx[order]
    ix_s = ix[order]
    kx_flat = kx.reshape(-1)
    idx = torch.searchsorted(fx_s, kx_flat)
    idx = idx.clamp(1, w_rfft - 1)
    f0 = fx_s[idx - 1]
    f1 = fx_s[idx]
    t = (kx_flat - f0) / (f1 - f0 + 1e-20)
    t = t.clamp(0.0, 1.0)
    ix0 = ix_s[idx - 1].to(dtype)
    ix1 = ix_s[idx].to(dtype)
    out = ix0 * (1.0 - t) + ix1 * t
    return out.reshape(kx.shape)


def warp_rfft_power_isotropic(
    power_rfft: torch.Tensor,
    *,
    image_sidelength: int,
    pixel_spacing_angstroms: float,
    scale_factor: float | torch.Tensor,
) -> torch.Tensor:
    """
    Resample rFFT power so output spatial frequency f reads input at f * scale_factor.

    With magnification ``m = sqrt(Δf_local/Δf_average)``, map to reference defocus using
    ``scale_factor = sqrt(|Δf_average/Δf_local|)`` (positive; defocus ratio stabilised
    in the caller when |Δf| is tiny). Frequency rescaling is ``k_in = s * k_out`` with
    scalar ``s``. Interpolation: bilinear via ``sample_image_2d`` (y, x) in pixel index
    coordinates ``[0, N-1]``.
    """
    device = power_rfft.device
    dtype = power_rfft.dtype
    h = image_sidelength
    ph, w_rfft = power_rfft.shape
    if ph != h:
        raise ValueError(f"power_rfft height {ph} != image_sidelength {h}")
    w = (w_rfft - 1) * 2
    sf = torch.as_tensor(scale_factor, device=device, dtype=dtype)
    sf0 = float(sf.reshape(-1)[0].item()) if sf.numel() > 0 else 1.0
    if abs(sf0 - 1.0) < 1e-7:
        return power_rfft.clone()
    s = float(sf.reshape(-1)[0].item())
    fy = torch.fft.fftfreq(h, d=pixel_spacing_angstroms, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(w, d=pixel_spacing_angstroms, device=device, dtype=dtype)
    ky_out = fy.view(h, 1).expand(h, w_rfft)
    kx_out = fx.view(1, w_rfft).expand(h, w_rfft)
    ky_in = ky_out * s
    kx_in = kx_out * s
    iy_f = _iy_float_from_ky_vectorized(
        ky_in, h, pixel_spacing_angstroms, device, dtype
    )
    ix_f = _ix_float_from_kx_vectorized(
        kx_in, w, w_rfft, pixel_spacing_angstroms, device, dtype
    )
    iy_f = iy_f.clamp(0.0, float(h - 1))
    ix_f = ix_f.clamp(0.0, float(w_rfft - 1))
    coordinates = torch.stack([iy_f, ix_f], dim=-1)
    out = sample_image_2d(
        power_rfft.float(),
        coordinates,
        interpolation="bilinear",
    )
    return out.to(dtype)


def _defocus_local_per_patch(
    result2d: Defocus2DResults,
    normalised_patch_positions: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Δf_local (µm) per patch, shape (t, gh, gw)."""
    pos = normalised_patch_positions.to(device=device, dtype=dtype)
    if result2d.defocus_model_type == "grid":
        model = result2d.defocus_model
        if not isinstance(model, CubicCatmullRomGrid3d):
            raise TypeError("grid defocus model must be CubicCatmullRomGrid3d")
        flat = pos.reshape(-1, 3)
        d = model(flat).squeeze(-1)
        t, gh, gw = pos.shape[:3]
        return d.reshape(t, gh, gw)
    if result2d.defocus_model_type == "linear":
        lm = result2d.defocus_model
        if not isinstance(lm, LinearDefocusModel):
            raise TypeError("linear defocus model must be LinearDefocusModel")
        angle_rad = math.radians(float(lm.defocus_gradient_angle))
        au = math.cos(angle_rad)
        av = math.sin(angle_rad)
        _eps = 1e-8
        _norm = math.sqrt(au * au + av * av + _eps)
        dir_u = au / _norm
        dir_v = av / _norm
        angle_r = math.atan2(dir_v, dir_u)
        cos_a = math.cos(angle_r)
        sin_a = math.sin(angle_r)
        x_norm = pos[..., 1]
        y_norm = pos[..., 2]
        projected = (x_norm - 0.5) * cos_a + (y_norm - 0.5) * sin_a
        d0 = float(lm.defocus_0)
        gm = float(lm.defocus_gradient_magnitude)
        return torch.as_tensor(d0, device=device, dtype=dtype) + projected * gm
    raise ValueError(f"Unknown defocus_model_type {result2d.defocus_model_type}")


def _safe_defocus_average_um(
    d_avg: float,
    *,
    eps: float = 1e-12,
) -> float:
    if abs(d_avg) < eps:
        return eps if d_avg >= 0.0 else -eps
    return d_avg


def _mean_tilt_corrected_ps_2d_rfft(
    result1d: Defocus1DResults,
    result2d: Defocus2DResults,
    *,
    normalised_patch_positions: torch.Tensor,
    optical_params: OpticalParams,
    laser_params: LaserParams | None,
    defocus_average_um: float | None,
    eps_defocus_avg: float,
) -> tuple[torch.Tensor, TiltCorrectedPsAux]:
    del laser_params  # reserved for API compatibility; warp path does not use CTF here
    patches = result2d.patch_power_spectra
    if patches is None:
        raise ValueError("result2d.patch_power_spectra is required for tilt correction")
    t, gh, gw, ph, pw_r = patches.shape
    device = patches.device
    dtype = patches.dtype
    image_sidelength = ph
    pixel = optical_params.pixel_spacing_angstroms

    d_avg = (
        defocus_average_um
        if defocus_average_um is not None
        else _tensor_to_float(result1d.ctf_model.defocus_um)
    )
    d_avg_safe = _safe_defocus_average_um(d_avg, eps=eps_defocus_avg)

    d_local = _defocus_local_per_patch(
        result2d, normalised_patch_positions, device, dtype
    )
    eps_l = torch.as_tensor(eps_defocus_avg, device=device, dtype=dtype)
    d_loc_den = torch.where(
        d_local.abs() < eps_l,
        torch.where(d_local >= 0, eps_l, -eps_l),
        d_local,
    )
    # m = sqrt(|Δf_local/Δf_avg|); warp uses k_in = k_out * sqrt(|Δf_avg/Δf_local|)
    ratio_avg_over_local = d_avg_safe / d_loc_den
    scale = torch.sqrt(torch.abs(ratio_avg_over_local) + 1e-30).clamp(max=1.0e3)
    m = torch.sqrt(torch.abs(d_loc_den / d_avg_safe) + 1e-30).clamp(max=1.0e3)

    accum = torch.zeros((ph, pw_r), device=device, dtype=dtype)
    n = 0
    for ti in range(t):
        for gi in range(gh):
            for gj in range(gw):
                ps = patches[ti, gi, gj]
                sf = float(scale[ti, gi, gj].item())
                wps = warp_rfft_power_isotropic(
                    ps,
                    image_sidelength=image_sidelength,
                    pixel_spacing_angstroms=pixel,
                    scale_factor=sf,
                )
                accum = accum + wps
                n += 1
    mean_2d = accum / max(n, 1)
    aux: TiltCorrectedPsAux = {
        "defocus_local_um": d_local,
        "defocus_average_um": d_avg,
        "m": m,
    }
    return mean_2d, aux


def tilt_corrected_mean_ps_2d(
    result1d: Defocus1DResults,
    result2d: Defocus2DResults,
    *,
    normalised_patch_positions: torch.Tensor,
    optical_params: OpticalParams,
    laser_params: LaserParams | None = None,
    defocus_average_um: float | None = None,
    eps_defocus_avg: float = 1e-12,
) -> tuple[torch.Tensor, TiltCorrectedPsAux]:
    """Isotropic frequency warp per patch, then average tilt-corrected 2D power."""
    return _mean_tilt_corrected_ps_2d_rfft(
        result1d,
        result2d,
        normalised_patch_positions=normalised_patch_positions,
        optical_params=optical_params,
        laser_params=laser_params,
        defocus_average_um=defocus_average_um,
        eps_defocus_avg=eps_defocus_avg,
    )


def tilt_corrected_mean_ps_1d(
    result1d: Defocus1DResults,
    result2d: Defocus2DResults,
    *,
    normalised_patch_positions: torch.Tensor,
    optical_params: OpticalParams,
    laser_params: LaserParams | None = None,
    use_equiphase: bool = True,
    equiphase_n_theta: int = 64,
    defocus_average_um: float | None = None,
    eps_defocus_avg: float = 1e-12,
) -> tuple[torch.Tensor, TiltCorrectedPsAux]:
    """Mean tilt-corrected 2D spectrum, then 1D equiphase or rotational average."""
    mean_2d, aux = _mean_tilt_corrected_ps_2d_rfft(
        result1d,
        result2d,
        normalised_patch_positions=normalised_patch_positions,
        optical_params=optical_params,
        laser_params=laser_params,
        defocus_average_um=defocus_average_um,
        eps_defocus_avg=eps_defocus_avg,
    )
    d_avg_ref = float(aux["defocus_average_um"])

    astig = float(result2d.astigmatism or 0.0)
    astig_ang = float(result2d.astigmatism_angle or 0.0)
    if result2d.phase_shift_degrees is not None:
        phase_deg = float(result2d.phase_shift_degrees)
    else:
        phase_deg = _tensor_to_float(result1d.ctf_model.phase_shift_degrees)

    ph, _pw_r = mean_2d.shape
    pixel = optical_params.pixel_spacing_angstroms

    if use_equiphase:
        ps_1d = equiphase_average_power_to_1d_rfft(
            mean_2d,
            ph,
            pixel,
            defocus_um=d_avg_ref,
            astigmatism_um=astig,
            astigmatism_angle_deg=astig_ang,
            phase_shift_deg=phase_deg,
            voltage_kev=optical_params.voltage_kev,
            spherical_aberration_mm=optical_params.spherical_aberration_mm,
            amplitude_contrast=optical_params.amplitude_contrast_fraction,
            laser_params=laser_params,
            n_theta=equiphase_n_theta,
        )
    else:
        averaged, _ = rotational_average_dft_2d(
            mean_2d.cpu(),
            image_shape=(ph, ph),
            rfft=True,
            fftshifted=False,
        )
        ps_1d = averaged.to(mean_2d.device)
    return ps_1d, aux
