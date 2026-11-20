"""Equiphase averaging of 2D rFFT power to 1D, using torch_ctf symmetric phase chi."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch_ctf._ctf_core import _prepare_inputs
from torch_ctf.ctf_aberrations import apply_astigmatism_to_defocus
from torch_ctf.ctf_lpp import _make_lpp_phase_shift_provider
from torch_ctf.ctf_utils import calculate_total_phase_shift
from torch_grid_utils import fftfreq_to_dft_coordinates

from torch_ctf_estimation.utils.rfft_shape import as_image_shape

if TYPE_CHECKING:
    from torch_ctf_estimation.models import LaserParams


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
    """Scale window around shell |q| for the local chi root (not Cs-turnover alias)."""
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
    image_sidelength: int | tuple[int, int],
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

    With ``n = min(H, W)``, sample ``i`` is referenced to the point
    ``(kx, ky) = (i / n, 0)`` cycles/pixel, so the profile sits at
    ``rfftfreq(n)``. Each sample averages the power over ``n_theta`` azimuths at
    the radius where chi equals chi at that reference point.

    Parameters
    ----------
    power_spectrum_rfft : torch.Tensor
        Real power |F|^2 on rfft grid, shape (H, W//2+1).
    image_sidelength : int or tuple[int, int]
        Real-space ``(H, W)`` of the original image, or ``N`` for a square one.
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
        1D tensor of length ``min(H, W) // 2 + 1``, at ``rfftfreq(min(H, W))``.
    """
    device = power_spectrum_rfft.device
    dtype = power_spectrum_rfft.dtype
    h, w = as_image_shape(image_sidelength)
    w_rfft = w // 2 + 1
    if tuple(power_spectrum_rfft.shape[-2:]) != (h, w_rfft):
        raise ValueError(
            f"power_spectrum_rfft shape {tuple(power_spectrum_rfft.shape[-2:])} "
            f"does not match image shape {(h, w)} (expected {(h, w_rfft)})."
        )
    n = min(h, w)
    n_bins = n // 2 + 1

    qdtype = torch.float32
    ps = power_spectrum_rfft.unsqueeze(0).unsqueeze(0).float()
    # Append row 0 below row h - 1 so rows in (h - 1, h) interpolate across the
    # wrap from the most negative ky to ky = 0.
    ps = torch.cat([ps, ps[..., :1, :]], dim=-2)
    s_max = 0.5 / pixel_spacing_angstroms + 1e-6

    two_pi = torch.tensor(2.0 * 3.141592653589793, device=device, dtype=qdtype)
    thetas = torch.arange(n_theta, device=device, dtype=qdtype) * (two_pi / n_theta)

    # Reference radius of each shell (cycles/Å), on the +kx axis.
    q_refs = torch.arange(n_bins, device=device, dtype=qdtype) / (
        n * pixel_spacing_angstroms
    )

    scale_lo, scale_hi = _equiphase_s_scale_window(defocus_um, astigmatism_um)
    n_grid = 65
    scales = torch.linspace(scale_lo, scale_hi, n_grid, device=device, dtype=qdtype)
    s_cand = (q_refs[:, None] * scales[None, :]).clamp(min=1e-7, max=float(s_max))
    cphi = thetas.cos()
    sphi = thetas.sin()
    kx_g = s_cand[:, :, None] * cphi[None, None, :]
    ky_g = s_cand[:, :, None] * sphi[None, None, :]
    # Evaluate the candidates and the reference points in one batch, so the LPP
    # peak-phase normalisation is shared between them.
    chi_all = _chi_at_kxy_batch(
        torch.cat([kx_g.reshape(-1), q_refs]),
        torch.cat([ky_g.reshape(-1), torch.zeros_like(q_refs)]),
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
    )
    chi_bgt = chi_all[: n_bins * n_grid * n_theta].reshape(n_bins, n_grid, n_theta)
    chi_refs = chi_all[n_bins * n_grid * n_theta :]
    err = (chi_bgt - chi_refs.view(n_bins, 1, 1)).abs()
    s_sol = torch.gather(s_cand, 1, err.argmin(dim=1))
    kx = s_sol * cphi
    ky = s_sol * sphi
    kx, ky = _fold_hermitian_rfft_kxy(kx, ky)
    # fractional (row, column) in the unshifted rfft: negative ky wrap to rows in
    # (h - 1, h), between row h - 1 and row 0; |ky| is capped at Nyquist
    fftfreqs = torch.stack(
        [(ky * pixel_spacing_angstroms).clamp(-0.5, 0.5), kx * pixel_spacing_angstroms],
        dim=-1,
    ).to(qdtype)
    coordinates = fftfreq_to_dft_coordinates(fftfreqs, (h, w), rfft=True)
    iy_f = coordinates[..., 0]
    ix_f = coordinates[..., 1].clamp(0.0, float(w_rfft - 1))
    x_norm = 2.0 * ix_f / max(w_rfft - 1, 1) - 1.0
    y_norm = 2.0 * iy_f / h - 1.0  # padded height h + 1, align_corners
    grid = torch.stack([x_norm, y_norm], dim=-1).reshape(1, 1, n_bins * n_theta, 2)
    sampled = F.grid_sample(
        ps,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(n_bins, n_theta)
    return sampled.mean(dim=1).to(dtype)
