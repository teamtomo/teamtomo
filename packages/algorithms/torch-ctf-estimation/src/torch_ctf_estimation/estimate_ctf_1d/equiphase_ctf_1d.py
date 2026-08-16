"""Equiphase averaging of 2D rFFT power to 1D, using torch_ctf symmetric phase chi."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch_ctf._ctf_core import _build_freq_grid, _phase_symmetric, _prepare_inputs
from torch_ctf.ctf_aberrations import apply_astigmatism_to_defocus
from torch_ctf.ctf_lpp import _make_lpp_phase_shift_provider
from torch_ctf.ctf_utils import calculate_total_phase_shift
from torch_fourier_filter.dft_utils import (
    _find_shell_indices_2d,
    _frequency_bin_split_values,
)
from torch_grid_utils.fftfreq_grid import fftfreq_grid

if TYPE_CHECKING:
    from torch_ctf_estimation.models import LaserParams


def _shell_index_lists_for_rfft(
    *,
    image_shape: tuple[int, int],
    n_bins: int,
    device: torch.device,
) -> list[torch.Tensor]:
    # torch_fourier_filter shells index a CPU coordinate grid; build on CPU then move.
    cpu = torch.device("cpu")
    grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=True,
        fftshift=False,
        norm=True,
        device=cpu,
    )
    split_values = _frequency_bin_split_values(n_bins, device=cpu)
    shells: list[torch.Tensor] = _find_shell_indices_2d(
        values=grid, split_values=split_values
    )[:-1]
    return [s.to(device) for s in shells]


def _chi_symmetric_grid(
    *,
    defocus_um: float | torch.Tensor,
    astigmatism_um: float | torch.Tensor,
    astigmatism_angle_deg: float | torch.Tensor,
    voltage_kev: float | torch.Tensor,
    spherical_aberration_mm: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift_deg: float | torch.Tensor,
    pixel_size_angstrom: float | torch.Tensor,
    image_shape: tuple[int, int],
    laser_params: LaserParams | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-grid chi (radians) and fft_freq_grid in cycles/Angstrom."""
    (
        defocus,
        astigmatism,
        astigmatism_angle,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        pixel_size,
        _device,
    ) = _prepare_inputs(
        defocus=defocus_um,
        astigmatism=astigmatism_um,
        astigmatism_angle=astigmatism_angle_deg,
        voltage=voltage_kev,
        spherical_aberration=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift_deg,
        pixel_size=pixel_size_angstrom,
    )
    defocus = defocus.to(device)
    astigmatism = astigmatism.to(device)
    astigmatism_angle = astigmatism_angle.to(device)
    voltage = voltage.to(device)
    spherical_aberration = spherical_aberration.to(device)
    amplitude_contrast = amplitude_contrast.to(device)
    phase_shift = phase_shift.to(device)
    pixel_size = pixel_size.to(device)
    fft_freq_grid, fft_freq_grid_squared, rho, theta = _build_freq_grid(
        image_shape=image_shape,
        pixel_size=pixel_size,
        rfft=True,
        fftshift=False,
        device=device,
        transform_matrix=None,
    )
    defocus_eff = apply_astigmatism_to_defocus(
        defocus=defocus,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        fft_freq_grid=fft_freq_grid,
        fft_freq_grid_squared=fft_freq_grid_squared,
    )
    if laser_params is None or not laser_params.model_laser:
        chi = _phase_symmetric(
            defocus=defocus_eff,
            voltage=voltage,
            spherical_aberration=spherical_aberration,
            amplitude_contrast=amplitude_contrast,
            phase_shift=phase_shift,
            fft_freq_grid_squared=fft_freq_grid_squared,
            rho=rho,
            theta=theta,
            even_zernike_coeffs=None,
            phase_shift_provider=None,
            fft_freq_grid=None,
        )
    else:
        lp = laser_params
        provider = _make_lpp_phase_shift_provider(
            NA=lp.NA,
            laser_wavelength_angstrom=lp.laser_wavelength_angstrom,
            focal_length_angstrom=lp.focal_length_angstrom,
            laser_xy_angle_deg=lp.laser_xy_angle_deg,
            laser_xz_angle_deg=lp.laser_xz_angle_deg,
            laser_long_offset_angstrom=lp.laser_long_offset_angstrom,
            laser_trans_offset_angstrom=lp.laser_trans_offset_angstrom,
            laser_polarization_angle_deg=lp.laser_polarization_angle_deg,
            peak_phase_deg=lp.peak_phase_deg,
            dual_laser=lp.dual_laser,
        )
        zero_phase = torch.zeros_like(phase_shift)
        chi = _phase_symmetric(
            defocus=defocus_eff,
            voltage=voltage,
            spherical_aberration=spherical_aberration,
            amplitude_contrast=amplitude_contrast,
            phase_shift=zero_phase,
            fft_freq_grid_squared=fft_freq_grid_squared,
            rho=rho,
            theta=theta,
            even_zernike_coeffs=None,
            phase_shift_provider=provider,
            fft_freq_grid=fft_freq_grid,
        )
    return chi.squeeze(), fft_freq_grid.squeeze()


def _iy_float_from_ky(
    ky: torch.Tensor,
    h: int,
    pixel_spacing: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    fy = torch.fft.fftfreq(h, d=pixel_spacing, device=device, dtype=dtype)
    iy = torch.arange(h, device=device, dtype=dtype)
    order = torch.argsort(fy)
    fy_s = fy[order]
    iy_s = iy[order]
    ky_flat = ky.reshape(-1)
    idx = torch.searchsorted(fy_s, ky_flat).clamp(1, h - 1)
    f0 = fy_s[idx - 1]
    f1 = fy_s[idx]
    t = ((ky_flat - f0) / (f1 - f0 + 1e-20)).clamp(0.0, 1.0)
    out = iy_s[idx - 1].to(dtype) * (1.0 - t) + iy_s[idx].to(dtype) * t
    return out.reshape(ky.shape)


def _ix_float_from_kx(
    kx: torch.Tensor,
    image_sidelength: int,
    w_rfft: int,
    pixel_spacing: float,
) -> torch.Tensor:
    """Map kx (cycles/Å) to rFFT column index: ``ix = kx * N * Δx``."""
    return (kx * image_sidelength * pixel_spacing).clamp(0.0, float(w_rfft - 1))


def _fold_hermitian_rfft_kxy(
    kx: torch.Tensor, ky: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map (kx, ky) onto the rFFT half-plane using even power P(-k)=P(k)."""
    neg_x = kx < 0
    return torch.where(neg_x, -kx, kx), torch.where(neg_x, -ky, ky)


def _equiphase_s_scale_window(
    defocus_um: float | torch.Tensor,
    astigmatism_um: float | torch.Tensor,
) -> tuple[float, float]:
    """Scale window around shell |q| for the local χ root (not the Cs-turnover alias)."""
    if isinstance(defocus_um, torch.Tensor):
        df = abs(float(defocus_um.detach().cpu().reshape(-1)[0].item()))
    else:
        df = abs(float(defocus_um))
    if isinstance(astigmatism_um, torch.Tensor):
        ast = abs(float(astigmatism_um.detach().cpu().reshape(-1)[0].item()))
    else:
        ast = abs(float(astigmatism_um))
    frac = ast / max(df, 1e-6)
    pad = 0.15 + 0.5 * frac
    return max(0.4, 1.0 - 2.0 * pad), min(2.5, 1.0 + 2.0 * pad)


def _chi_at_kxy_batch(
    kx: torch.Tensor,
    ky: torch.Tensor,
    *,
    defocus_um: float | torch.Tensor,
    astigmatism_um: float | torch.Tensor,
    astigmatism_angle_deg: float | torch.Tensor,
    voltage_kev: float | torch.Tensor,
    spherical_aberration_mm: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift_deg: float | torch.Tensor,
    pixel_size_angstrom: float | torch.Tensor,
    laser_params: LaserParams | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Chi at query points; kx, ky in cycles/Angstrom, same shapes."""
    n = kx.numel()
    freq_y = ky.reshape(1, n, 1)
    freq_x = kx.reshape(1, n, 1)
    fft_freq_query = torch.stack([freq_y, freq_x], dim=-1)
    fft_freq_sq = freq_y**2 + freq_x**2

    (
        defocus,
        astigmatism,
        astigmatism_angle,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        _pixel_size,
        _,
    ) = _prepare_inputs(
        defocus=defocus_um,
        astigmatism=astigmatism_um,
        astigmatism_angle=astigmatism_angle_deg,
        voltage=voltage_kev,
        spherical_aberration=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift_deg,
        pixel_size=pixel_size_angstrom,
    )
    defocus = defocus.to(device=device, dtype=dtype)
    astigmatism = astigmatism.to(device=device, dtype=dtype)
    astigmatism_angle = astigmatism_angle.to(device=device, dtype=dtype)
    voltage = voltage.to(device=device, dtype=dtype)
    spherical_aberration = spherical_aberration.to(device=device, dtype=dtype)
    amplitude_contrast = amplitude_contrast.to(device=device, dtype=dtype)
    phase_shift = phase_shift.to(device=device, dtype=dtype)
    defocus_eff = apply_astigmatism_to_defocus(
        defocus=defocus,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        fft_freq_grid=fft_freq_query,
        fft_freq_grid_squared=fft_freq_sq,
    )
    vk = voltage.reshape(1, 1, 1).expand_as(defocus_eff).reshape(1, n)
    cshape = spherical_aberration.reshape(1, 1, 1).expand_as(defocus_eff).reshape(1, n)
    ashape = amplitude_contrast.reshape(1, 1, 1).expand_as(defocus_eff).reshape(1, n)

    if laser_params is None or not laser_params.model_laser:
        phase_deg_query = phase_shift.expand_as(defocus_eff).reshape(1, n)
        chi_q = calculate_total_phase_shift(
            defocus_um=defocus_eff.reshape(1, n),
            voltage_kv=vk,
            spherical_aberration_mm=cshape,
            phase_shift_degrees=phase_deg_query,
            amplitude_contrast_fraction=ashape,
            fftfreq_grid_angstrom_squared=fft_freq_sq.reshape(1, n),
        )
    else:
        lp = laser_params
        provider = _make_lpp_phase_shift_provider(
            NA=lp.NA,
            laser_wavelength_angstrom=lp.laser_wavelength_angstrom,
            focal_length_angstrom=lp.focal_length_angstrom,
            laser_xy_angle_deg=lp.laser_xy_angle_deg,
            laser_xz_angle_deg=lp.laser_xz_angle_deg,
            laser_long_offset_angstrom=lp.laser_long_offset_angstrom,
            laser_trans_offset_angstrom=lp.laser_trans_offset_angstrom,
            laser_polarization_angle_deg=lp.laser_polarization_angle_deg,
            peak_phase_deg=lp.peak_phase_deg,
            dual_laser=lp.dual_laser,
        )
        phase_deg_q = provider(fft_freq_query, voltage)
        chi_q = calculate_total_phase_shift(
            defocus_um=defocus_eff.reshape(1, n),
            voltage_kv=vk,
            spherical_aberration_mm=cshape,
            phase_shift_degrees=phase_deg_q.reshape(1, n),
            amplitude_contrast_fraction=ashape,
            fftfreq_grid_angstrom_squared=fft_freq_sq.reshape(1, n),
        )
    return chi_q.reshape(kx.shape)


def _solve_s_equiphase(
    chi_ref: float,
    phi: float,
    *,
    s_max: float,
    defocus_um: float | torch.Tensor,
    astigmatism_um: float | torch.Tensor,
    astigmatism_angle_deg: float | torch.Tensor,
    voltage_kev: float | torch.Tensor,
    spherical_aberration_mm: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift_deg: float | torch.Tensor,
    pixel_size_angstrom: float,
    laser_params: LaserParams | None,
    device: torch.device,
    qdtype: torch.dtype,
    n_iter: int = 40,
) -> float:
    cphi = torch.tensor(phi, device=device, dtype=qdtype).cos()
    sphi = torch.tensor(phi, device=device, dtype=qdtype).sin()

    def chi_vec(s_vals: torch.Tensor) -> torch.Tensor:
        kx = s_vals * cphi
        ky = s_vals * sphi
        return _chi_at_kxy_batch(
            kx,
            ky,
            defocus_um=defocus_um,
            astigmatism_um=astigmatism_um,
            astigmatism_angle_deg=astigmatism_angle_deg,
            voltage_kev=voltage_kev,
            spherical_aberration_mm=spherical_aberration_mm,
            amplitude_contrast=amplitude_contrast,
            phase_shift_deg=phase_shift_deg,
            pixel_size_angstrom=torch.as_tensor(
                pixel_size_angstrom, device=device, dtype=qdtype
            ),
            laser_params=laser_params,
            device=device,
            dtype=qdtype,
        )

    lo = torch.tensor(1e-7, device=device, dtype=qdtype)
    hi = torch.tensor(s_max, device=device, dtype=qdtype)
    f_lo = float(chi_vec(lo.unsqueeze(0))[0].item())
    f_hi = float(chi_vec(hi.unsqueeze(0))[0].item())
    target = chi_ref
    if (f_lo - target) * (f_hi - target) > 0:
        grid_s = torch.linspace(
            float(lo), float(hi), steps=33, device=device, dtype=qdtype
        )
        ch_g = chi_vec(grid_s)
        err = (ch_g - target).abs()
        ib = int(err.argmin().item())
        return max(float(grid_s[ib].item()), 1e-7)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f_mid = float(chi_vec(mid.unsqueeze(0))[0].item())
        f_lo = float(chi_vec(lo.unsqueeze(0))[0].item())
        if (f_lo - target) * (f_mid - target) <= 0:
            hi = mid
        else:
            lo = mid
    return float((0.5 * (lo + hi)).item())


def equiphase_average_power_to_1d_rfft(
    power_spectrum_rfft: torch.Tensor,
    image_sidelength: int,
    pixel_spacing_angstroms: float,
    *,
    defocus_um: float | torch.Tensor,
    astigmatism_um: float | torch.Tensor,
    astigmatism_angle_deg: float | torch.Tensor,
    phase_shift_deg: float | torch.Tensor,
    voltage_kev: float | torch.Tensor,
    spherical_aberration_mm: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    laser_params: LaserParams | None = None,
    n_theta: int = 64,
) -> torch.Tensor:
    """
    Equiphase-averaged 1D power profile (same bin count as rotational_average_dft_2d).

    Parameters
    ----------
    power_spectrum_rfft : torch.Tensor
        Real power |F|^2 on rfft grid, shape (H, W//2+1).
    image_sidelength : int
        H (=W) of the original real-space patch.
    pixel_spacing_angstroms : float
        Pixel size in Angstroms.
    defocus_um, astigmatism_um, astigmatism_angle_deg :
        torch_ctf 2D conventions (mean defocus, half-difference astigmatism).
    phase_shift_deg : float | torch.Tensor
        Uniform phase plate shift; ignored for LPP when ``model_laser`` is True.
    voltage_kev, spherical_aberration_mm, amplitude_contrast :
        Optics (same as estimate_ctf_1d).
    laser_params : LaserParams | None
        If set and ``model_laser`` is True, chi uses LPP phase provider like
        calc_LPP_ctf_2D.
    n_theta : int
        Number of azimuthal samples per shell.

    Returns
    -------
    torch.Tensor
        1D tensor of length n_bins (min shell count), same ordering as rotational
        average.
    """
    device = power_spectrum_rfft.device
    dtype = power_spectrum_rfft.dtype
    h = image_sidelength
    w_rfft = h // 2 + 1
    n_bins = min((h // 2) + 1, (h // 2) + 1)

    qdtype = torch.float32
    chi_grid, fft_freq_grid = _chi_symmetric_grid(
        defocus_um=defocus_um,
        astigmatism_um=astigmatism_um,
        astigmatism_angle_deg=astigmatism_angle_deg,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        phase_shift_deg=phase_shift_deg,
        pixel_size_angstrom=torch.as_tensor(
            pixel_spacing_angstroms, device=device, dtype=qdtype
        ),
        image_shape=(h, h),
        laser_params=laser_params,
        device=device,
    )

    ps = power_spectrum_rfft.unsqueeze(0).unsqueeze(0)
    s_max = 0.5 / pixel_spacing_angstroms + 1e-6

    shells = _shell_index_lists_for_rfft(
        image_shape=(h, h), n_bins=n_bins, device=device
    )
    two_pi = torch.tensor(2.0 * 3.141592653589793, device=device, dtype=qdtype)
    thetas = torch.arange(n_theta, device=device, dtype=qdtype) * (two_pi / n_theta)

    chi_refs = torch.zeros(n_bins, device=device, dtype=qdtype)
    q_refs = torch.zeros(n_bins, device=device, dtype=qdtype)
    valid = torch.zeros(n_bins, device=device, dtype=torch.bool)
    for shell_i, shell in enumerate(shells):
        if shell.numel() == 0:
            continue
        iy = shell[:, 0].long()
        ix = shell[:, 1].long()
        fx = fft_freq_grid[iy, ix, 1]
        fy = fft_freq_grid[iy, ix, 0]
        pick = int((fx.abs() / (fy.abs() + 1e-9)).argmax().item())
        chi_refs[shell_i] = chi_grid[iy[pick], ix[pick]].to(qdtype)
        q_refs[shell_i] = torch.sqrt(fx[pick] ** 2 + fy[pick] ** 2).to(qdtype)
        valid[shell_i] = True

    scale_lo, scale_hi = _equiphase_s_scale_window(defocus_um, astigmatism_um)
    n_grid = 65
    scales = torch.linspace(scale_lo, scale_hi, n_grid, device=device, dtype=qdtype)
    s_cand = (q_refs[:, None] * scales[None, :]).clamp(min=1e-7, max=float(s_max))
    cphi = thetas.cos()
    sphi = thetas.sin()
    kx_g = s_cand[:, :, None] * cphi[None, None, :]
    ky_g = s_cand[:, :, None] * sphi[None, None, :]
    chi_bgt = _chi_at_kxy_batch(
        kx_g.reshape(-1),
        ky_g.reshape(-1),
        defocus_um=defocus_um,
        astigmatism_um=astigmatism_um,
        astigmatism_angle_deg=astigmatism_angle_deg,
        voltage_kev=voltage_kev,
        spherical_aberration_mm=spherical_aberration_mm,
        amplitude_contrast=amplitude_contrast,
        phase_shift_deg=phase_shift_deg,
        pixel_size_angstrom=torch.as_tensor(
            pixel_spacing_angstroms, device=device, dtype=qdtype
        ),
        laser_params=laser_params,
        device=device,
        dtype=qdtype,
    ).reshape(n_bins, n_grid, n_theta)
    err = (chi_bgt - chi_refs.view(n_bins, 1, 1)).abs()
    s_sol = torch.gather(s_cand, 1, err.argmin(dim=1))
    kx = s_sol * cphi
    ky = s_sol * sphi
    kx, ky = _fold_hermitian_rfft_kxy(kx, ky)
    ix_f = _ix_float_from_kx(kx, h, w_rfft, pixel_spacing_angstroms)
    iy_f = _iy_float_from_ky(ky, h, pixel_spacing_angstroms, device, qdtype).clamp(
        0.0, float(h - 1)
    )
    x_norm = 2.0 * ix_f / max(w_rfft - 1, 1) - 1.0
    y_norm = 2.0 * iy_f / max(h - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).reshape(1, 1, n_bins * n_theta, 2)
    sampled = F.grid_sample(
        ps.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(n_bins, n_theta)
    out_bins = sampled.mean(dim=1).to(dtype)
    return torch.where(valid, out_bins, torch.zeros_like(out_bins))
