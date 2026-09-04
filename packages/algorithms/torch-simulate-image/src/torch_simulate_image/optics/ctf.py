"""CTF application in the exit-wave domain."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_ctf import calc_LPP_ctf_2D, calculate_ctf_2d

from torch_simulate_image._validate import image_shape_from_tensor, validate_exit_wave

if TYPE_CHECKING:
    from torch_simulate_image.config import CtfConfig


def _optional_beam_tilt(
    config: CtfConfig, *, device: torch.device
) -> torch.Tensor | None:
    if config.beam_tilt_mrad is None:
        return None
    return torch.tensor(config.beam_tilt_mrad, dtype=torch.float32, device=device)


def _optional_transform_matrix(
    config: CtfConfig, *, device: torch.device
) -> torch.Tensor | None:
    if config.transform_matrix is None:
        return None
    return config.transform_matrix.to(device=device, dtype=torch.float32)


def _calculate_complex_ctf(
    config: CtfConfig,
    *,
    pixel_size: float,
    image_shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """Build a complex CTF via standard or LPP ``torch-ctf`` APIs."""
    beam_tilt_mrad = _optional_beam_tilt(config, device=device)
    transform_matrix = _optional_transform_matrix(config, device=device)

    if config.lpp.apply:
        lpp = config.lpp
        ctf = calc_LPP_ctf_2D(
            defocus=config.defocus_um,
            astigmatism=config.astigmatism_um,
            astigmatism_angle=config.astigmatism_angle_deg,
            voltage=config.voltage_kv,
            spherical_aberration=config.spherical_aberration_mm,
            amplitude_contrast=config.amplitude_contrast,
            pixel_size=pixel_size,
            image_shape=image_shape,
            rfft=False,
            fftshift=False,
            NA=lpp.NA,
            laser_wavelength_angstrom=lpp.laser_wavelength_angstrom,
            focal_length_angstrom=lpp.focal_length_angstrom,
            laser_xy_angle_deg=lpp.laser_xy_angle_deg,
            laser_xz_angle_deg=lpp.laser_xz_angle_deg,
            laser_long_offset_angstrom=lpp.laser_long_offset_angstrom,
            laser_trans_offset_angstrom=lpp.laser_trans_offset_angstrom,
            laser_polarization_angle_deg=lpp.laser_polarization_angle_deg,
            peak_phase_deg=lpp.peak_phase_deg,
            dual_laser=lpp.dual_laser,
            beam_tilt_mrad=beam_tilt_mrad,
            even_zernike_coeffs=config.even_zernike_coeffs,
            odd_zernike_coeffs=config.odd_zernike_coeffs,
            transform_matrix=transform_matrix,
            return_complex_ctf=True,
        )
        return ctf.to(device=device)

    ctf = calculate_ctf_2d(
        defocus=config.defocus_um,
        astigmatism=config.astigmatism_um,
        astigmatism_angle=config.astigmatism_angle_deg,
        voltage=config.voltage_kv,
        spherical_aberration=config.spherical_aberration_mm,
        amplitude_contrast=config.amplitude_contrast,
        phase_shift=config.phase_shift_deg,
        pixel_size=pixel_size,
        image_shape=image_shape,
        rfft=False,
        fftshift=False,
        beam_tilt_mrad=beam_tilt_mrad,
        even_zernike_coeffs=config.even_zernike_coeffs,
        odd_zernike_coeffs=config.odd_zernike_coeffs,
        transform_matrix=transform_matrix,
        return_complex_ctf=True,
    )
    return ctf.to(device=device)


def apply_ctf_to_exit_wave(
    exit_wave: torch.Tensor,
    config: CtfConfig,
    *,
    pixel_size: float,
) -> torch.Tensor:
    """Multiply an exit wave by the objective CTF in Fourier space.

    Uses :func:`torch_ctf.calculate_ctf_2d` by default, or
    :func:`torch_ctf.calc_LPP_ctf_2D` when ``config.lpp.apply`` is ``True``.
    The complex CTF is applied to the full FFT of ``exit_wave``.

    Parameters
    ----------
    exit_wave : torch.Tensor
        Complex tensor with shape ``(..., H, W)``.
    config : CtfConfig
        CTF parameters (voltage, defocus, aberrations, optional Zernikes /
        beam tilt / anisotropic magnification / LPP). When ``config.apply``
        is ``False``, ``exit_wave`` is returned unchanged.
    pixel_size : float
        Pixel size in Angstroms.

    Returns
    -------
    torch.Tensor
        CTF-filtered complex exit wave.
    """
    validate_exit_wave(exit_wave)
    if not config.apply:
        return exit_wave

    image_shape = image_shape_from_tensor(exit_wave)
    device = exit_wave.device
    ctf = _calculate_complex_ctf(
        config,
        pixel_size=pixel_size,
        image_shape=image_shape,
        device=device,
    )
    ctf = ctf.to(device=device, dtype=exit_wave.dtype)
    exit_dft = torch.fft.fft2(exit_wave, dim=(-2, -1))
    filtered_dft = exit_dft * ctf
    filtered: torch.Tensor = torch.fft.ifft2(
        filtered_dft, dim=(-2, -1), norm="backward"
    )
    return filtered.to(dtype=exit_wave.dtype)
