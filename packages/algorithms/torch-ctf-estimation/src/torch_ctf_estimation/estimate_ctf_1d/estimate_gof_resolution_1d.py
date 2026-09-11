"""CTFFind-style 1D goodness-of-fit vs resolution.

CTFFind4/5 report the last spacing where a normalised 1D cross-correlation
between the experimental power-spectrum profile and the fitted CTF stays at
or above 0.5. The CC is computed in a moving window whose width is one
CTF² cycle (CTFFind4) or 1.5 cycles after thickness is fitted (CTFFind5),
so ice nodes do not kill the diagnostic.

This module is a diagnostic. It does not replace the 1D thickness grid.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch_ctf import calculate_ctf_1d
from torch_ctf.ctf_aberrations import calculate_relativistic_electron_wavelength
from torch_ctf.ctf_thickness import calculate_ctf_thickness_1d

from torch_ctf_estimation.metrics.fit_metrics import (
    l2_normalized_cross_correlation,
    pearson_r_flat,
)
from torch_ctf_estimation.models.results_models import Thickness1DResults


class GofResolution1DResult(NamedTuple):
    """Windowed 1D GoF curve and the 0.5 crossing in Angstroms."""

    fit_res_A: float
    thickness_from_node_A: float
    spacing_A: torch.Tensor
    window_pearson_r: torch.Tensor


def electron_wavelength_angstrom(voltage_kev: float) -> float:
    """Relativistic electron wavelength in Angstroms."""
    lam_m = calculate_relativistic_electron_wavelength(voltage_kev * 1000.0)
    return float(lam_m.detach().cpu().item()) * 1.0e10


def thickness_from_first_node_angstroms(d_angstroms: float, voltage_kev: float) -> float:
    """t = 1 / (lambda * g^2) = d^2 / lambda (first CTF_t node)."""
    if d_angstroms <= 0.0 or not torch.isfinite(torch.tensor(d_angstroms)):
        return float("nan")
    return (d_angstroms**2) / electron_wavelength_angstrom(voltage_kev)


def interpolate_cc_drop_angstroms(
    spacing_A: torch.Tensor,
    cc: torch.Tensor,
    threshold: float,
) -> float:
    """Last spacing (Å) where ``cc`` stays at or above ``threshold``, interpolated.

    ``spacing_A`` must be decreasing (low frequency → high frequency).
    """
    sp = spacing_A.detach().cpu().reshape(-1).float()
    y = cc.detach().cpu().reshape(-1).float()
    n = int(sp.numel())
    if n == 0:
        return float("nan")
    last_good = -1
    for i in range(n):
        v = float(y[i].item())
        if v == v and v >= threshold:
            last_good = i
        elif last_good >= 0:
            break
    if last_good < 0:
        return float(sp[0].item())
    if last_good >= n - 1:
        return float(sp[-1].item())
    y0 = float(y[last_good].item())
    y1 = float(y[last_good + 1].item())
    s0 = float(sp[last_good].item())
    s1 = float(sp[last_good + 1].item())
    if y1 >= threshold:
        return s1
    denom = y0 - y1
    if abs(denom) < 1e-12:
        return s0
    frac = min(max((y0 - threshold) / denom, 0.0), 1.0)
    return s0 + frac * (s1 - s0)


def ctf2_period_s2(defocus_um: float, voltage_kev: float) -> float:
    """Δ(s²) between consecutive CTF² maxima, ignoring Cs (Å⁻²)."""
    defocus_A = abs(defocus_um) * 1.0e4
    lam = electron_wavelength_angstrom(voltage_kev)
    return 1.0 / max(lam * defocus_A, 1e-12)


def moving_window_pearson(
    data: torch.Tensor,
    model: torch.Tensor,
    spatial_freq: torch.Tensor,
    *,
    window_s2: float,
    min_bins: int = 8,
) -> torch.Tensor:
    """Pearson r in a moving window of half-width ``window_s2 / 2`` in s²."""
    data = data.reshape(-1).float()
    model = model.reshape(-1).float()
    s2 = spatial_freq.reshape(-1).float() ** 2
    n = int(data.numel())
    half = 0.5 * window_s2
    out = torch.full((n,), float("nan"), dtype=torch.float32)
    for i in range(n):
        mask = (s2 - s2[i]).abs() <= half
        if int(mask.sum().item()) < min_bins:
            continue
        r = pearson_r_flat(data[mask], model[mask])
        if r == r:
            out[i] = r
    return out


def estimate_gof_resolution_1d(
    residual_1d: torch.Tensor,
    spatial_freq: torch.Tensor,
    model_1d: torch.Tensor,
    *,
    defocus_um: float,
    voltage_kev: float,
    window_cycles: float = 1.0,
    cc_threshold: float = 0.5,
) -> GofResolution1DResult:
    """Last spacing where windowed 1D Pearson r vs ``model_1d`` stays ≥ threshold."""
    residual_1d = residual_1d.reshape(-1)
    spatial_freq = spatial_freq.reshape(-1).to(device=residual_1d.device)
    model_1d = model_1d.reshape(-1).to(device=residual_1d.device)
    period = ctf2_period_s2(defocus_um, voltage_kev)
    r = moving_window_pearson(
        residual_1d,
        model_1d,
        spatial_freq,
        window_s2=window_cycles * period,
    )
    valid = torch.isfinite(r)
    spacing = torch.full((int(residual_1d.numel()),), float("nan"))
    spacing[valid] = 1.0 / spatial_freq[valid].detach().cpu().clamp(min=1e-12)
    fit_res = interpolate_cc_drop_angstroms(spacing[valid], r[valid], cc_threshold)
    return GofResolution1DResult(
        fit_res_A=fit_res,
        thickness_from_node_A=thickness_from_first_node_angstroms(
            fit_res, voltage_kev
        ),
        spacing_A=spacing[valid],
        window_pearson_r=r[valid],
    )


def residual_and_freqs_from_thickness1d(
    thickness: Thickness1DResults,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Background-subtracted 1D spectrum and spatial frequencies on the fit band."""
    if thickness.powerspectrum_1d is None:
        raise ValueError("Thickness1DResults.powerspectrum_1d is required")
    ps = thickness.powerspectrum_1d.reshape(-1)
    freq = thickness.frequencies_1d.reshape(-1).to(device=ps.device)
    lo = thickness.low_frequency_fit
    hi = thickness.high_frequency_fit
    if lo is None or hi is None:
        mask = torch.ones(ps.numel(), dtype=torch.bool, device=ps.device)
    else:
        mask = (freq >= lo) & (freq <= hi)
    y = ps[mask]
    f = freq[mask]
    if thickness.background_model is not None and int(y.numel()) > 0:
        x = torch.linspace(0.0, 1.0, steps=int(y.numel()), device=y.device)
        bg = torch.exp(thickness.background_model(x).squeeze())
        y = y - bg.to(device=y.device, dtype=y.dtype)
    return y.detach(), f.detach()


def simulate_thin_power_1d(
    *,
    n_samples: int,
    defocus_um: float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    pixel_spacing_angstroms: float,
    phase_shift_deg: float = 0.0,
) -> torch.Tensor:
    """Thin-sample 1D power spectrum (CTF²) on the rFFT frequency grid."""
    ctf = calculate_ctf_1d(
        defocus=defocus_um,
        voltage=voltage_kev,
        spherical_aberration=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift_deg,
        pixel_size=pixel_spacing_angstroms,
        n_samples=n_samples,
        oversampling_factor=3,
    )
    return ctf**2


def simulate_thin_abs_ctf_1d(
    *,
    n_samples: int,
    defocus_um: float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    pixel_spacing_angstroms: float,
    phase_shift_deg: float = 0.0,
) -> torch.Tensor:
    """Thin-sample |CTF| on the 1D rFFT frequency grid (CTFFind amplitude model)."""
    ctf = calculate_ctf_1d(
        defocus=defocus_um,
        voltage=voltage_kev,
        spherical_aberration=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift_deg,
        pixel_size=pixel_spacing_angstroms,
        n_samples=n_samples,
        oversampling_factor=3,
    )
    return ctf.abs()


def simulate_thickness_abs_ctf_1d(
    *,
    n_samples: int,
    thickness_angstroms: float,
    defocus_um: float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    pixel_spacing_angstroms: float,
    phase_shift_deg: float = 0.0,
) -> torch.Tensor:
    """|CTF_t| amplitude transfer (sinc × sin χ)."""
    ctf = calculate_ctf_thickness_1d(
        return_power_spectrum=False,
        sample_thickness_angstrom=thickness_angstroms,
        defocus=defocus_um,
        voltage=voltage_kev,
        spherical_aberration=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift_deg,
        pixel_size=pixel_spacing_angstroms,
        n_samples=n_samples,
        oversampling_factor=3,
    )
    return ctf.abs()


def local_maxima_1d(y: torch.Tensor) -> list[int]:
    """Indices of strict local maxima (plateau start counts)."""
    y = y.detach().cpu().reshape(-1)
    idx: list[int] = []
    n = int(y.numel())
    for i in range(1, n - 1):
        if float(y[i]) >= float(y[i - 1]) and float(y[i]) > float(y[i + 1]):
            idx.append(i)
    return idx


def _equalize_01(x: torch.Tensor) -> torch.Tensor:
    lo = x.min()
    hi = x.max()
    return (x - lo) / (hi - lo + 1e-8)


def estimate_gof_by_cycles(
    data_1d: torch.Tensor,
    spatial_freq: torch.Tensor,
    model_1d: torch.Tensor,
    *,
    voltage_kev: float,
    equalize: bool = True,
    cycles_per_window: float = 1.0,
    cc_threshold: float = 0.5,
    min_bins: int = 4,
) -> GofResolution1DResult:
    """NCC in intervals between consecutive maxima of ``model_1d`` (CTFFind4).

    ``cycles_per_window`` > 1 extends each interval (CTFFind5 uses 1.5 with CTF_t).
    If ``equalize`` is True, each window is rescaled to [0, 1] before NCC so the
    score follows ring phase rather than envelope.
    """
    data_1d = data_1d.reshape(-1).float()
    model_1d = model_1d.reshape(-1).float().to(device=data_1d.device)
    spatial_freq = spatial_freq.reshape(-1).float().to(device=data_1d.device)
    peaks = local_maxima_1d(model_1d)
    n = int(data_1d.numel())
    spacing_vals: list[float] = []
    cc_vals: list[float] = []
    for i in range(len(peaks) - 1):
        i0 = peaks[i]
        span = peaks[i + 1] - peaks[i]
        if cycles_per_window <= 1.0 + 1e-6:
            i1 = peaks[i + 1]
        else:
            i1 = int(round(i0 + cycles_per_window * span))
            i1 = min(max(i1, i0 + min_bins), n - 1)
        if i1 - i0 < min_bins:
            continue
        d = data_1d[i0 : i1 + 1]
        m = model_1d[i0 : i1 + 1]
        if equalize:
            d = _equalize_01(d)
            m = _equalize_01(m)
        r = l2_normalized_cross_correlation(d, m)
        if r != r:
            continue
        freq_hi = float(spatial_freq[i1].item())
        if freq_hi <= 0.0:
            continue
        spacing_vals.append(1.0 / freq_hi)
        cc_vals.append(r)
    if not spacing_vals:
        empty = torch.empty(0, dtype=torch.float32)
        return GofResolution1DResult(
            fit_res_A=float("nan"),
            thickness_from_node_A=float("nan"),
            spacing_A=empty,
            window_pearson_r=empty,
        )
    spacing = torch.tensor(spacing_vals, dtype=torch.float32)
    cc = torch.tensor(cc_vals, dtype=torch.float32)
    fit_res = interpolate_cc_drop_angstroms(spacing, cc, cc_threshold)
    return GofResolution1DResult(
        fit_res_A=fit_res,
        thickness_from_node_A=thickness_from_first_node_angstroms(
            fit_res, voltage_kev
        ),
        spacing_A=spacing,
        window_pearson_r=cc,
    )


def simulate_thickness_power_1d(
    *,
    n_samples: int,
    thickness_angstroms: float,
    defocus_um: float,
    voltage_kev: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    pixel_spacing_angstroms: float,
    phase_shift_deg: float = 0.0,
) -> torch.Tensor:
    """CTF_t 1D power spectrum (McMullan / CTFFind5 form)."""
    return calculate_ctf_thickness_1d(
        return_power_spectrum=True,
        sample_thickness_angstrom=thickness_angstroms,
        defocus=defocus_um,
        voltage=voltage_kev,
        spherical_aberration=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift_deg,
        pixel_size=pixel_spacing_angstroms,
        n_samples=n_samples,
        oversampling_factor=3,
    )
