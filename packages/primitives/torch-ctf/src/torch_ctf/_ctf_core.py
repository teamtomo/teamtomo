"""Internal core helpers for 2D CTF calculation pipelines."""

from collections.abc import Callable

import einops
import torch
from torch_grid_utils.polar_grid import fftfreq_grid_polar

from torch_ctf._ctf_preparation import (
    as_float_tensor,
    infer_device,
    prepare_frequency_grid_2d,
)
from torch_ctf.ctf_aberrations import (
    apply_astigmatism_to_defocus,
    apply_even_zernikes,
    apply_odd_zernikes,
)
from torch_ctf.ctf_utils import calculate_total_phase_shift

PhaseShiftProvider = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _prepare_inputs(
    defocus: float | torch.Tensor,
    astigmatism: float | torch.Tensor,
    astigmatism_angle: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.device,
]:
    """Prepare CTF scalar-like inputs as broadcast-ready tensors.

    Parameters
    ----------
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
        `(defocus_u + defocus_v) / 2`
    astigmatism : float | torch.Tensor
        Amount of astigmatism in micrometers.
        `(defocus_u - defocus_v) / 2`
    astigmatism_angle : float | torch.Tensor
        Angle of astigmatism in degrees. 0 places `defocus_u` along the y-axis.
    voltage : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration : float | torch.Tensor
        Spherical aberration in millimeters (mm).
    amplitude_contrast : float | torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    phase_shift : float | torch.Tensor
        Angle of phase shift applied to CTF in degrees.
    pixel_size : float | torch.Tensor
        Pixel size in Angstroms per pixel (A / px).

    Returns
    -------
    defocus : torch.Tensor
        Defocus in micrometers, converted to a tensor and reshaped for broadcasting.
    astigmatism : torch.Tensor
        Amount of astigmatism in micrometers, converted and reshaped for broadcasting.
    astigmatism_angle : torch.Tensor
        Angle of astigmatism in degrees, converted and reshaped for broadcasting.
    voltage : torch.Tensor
        Acceleration voltage in kilovolts (kV), converted and reshaped for broadcasting.
    spherical_aberration : torch.Tensor
        Spherical aberration in millimeters (mm), converted and reshaped for
        broadcasting.
    amplitude_contrast : torch.Tensor
        Fraction of amplitude contrast, converted and reshaped for broadcasting.
    phase_shift : torch.Tensor
        Angle of phase shift applied to CTF in degrees, converted and reshaped for
        broadcasting.
    device : torch.device
        Inferred device for tensor creation based on the input parameters.
    """
    device = infer_device(defocus)

    defocus = as_float_tensor(defocus, device=device)
    astigmatism = as_float_tensor(astigmatism, device=device)
    astigmatism_angle = as_float_tensor(astigmatism_angle, device=device)
    voltage = as_float_tensor(voltage, device=device)
    spherical_aberration = as_float_tensor(spherical_aberration, device=device)
    amplitude_contrast = as_float_tensor(amplitude_contrast, device=device)
    phase_shift = as_float_tensor(phase_shift, device=device)
    pixel_size = as_float_tensor(pixel_size, device=device)

    defocus = einops.rearrange(defocus, "... -> ... 1 1")
    voltage = einops.rearrange(voltage, "... -> ... 1 1")
    spherical_aberration = einops.rearrange(spherical_aberration, "... -> ... 1 1")
    amplitude_contrast = einops.rearrange(amplitude_contrast, "... -> ... 1 1")
    phase_shift = einops.rearrange(phase_shift, "... -> ... 1 1")

    return (
        defocus,
        astigmatism,
        astigmatism_angle,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        pixel_size,
        device,
    )


def _build_freq_grid(
    image_shape: tuple[int, int],
    pixel_size: float | torch.Tensor,
    rfft: bool,
    fftshift: bool,
    device: torch.device,
    transform_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct 2D frequency grids and corresponding polar coordinates.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Shape of the image, in real-space, for which to build the frequency grid.
    pixel_size : float | torch.Tensor
        Pixel size(s) in Angstroms.
    rfft : bool
        Whether the frequency grid is for an rfft (half) or fft (full) transform.
    fftshift : bool
        Whether to apply an FFT shift to the frequency grid.
    device : torch.device
        Device to use for tensor creation.
    transform_matrix : torch.Tensor | None, optional
        Optional 2x2 transformation matrix to apply to the frequency grid.

    Returns
    -------
    fft_freq_grid : torch.Tensor
        The 2D frequency grid in cycles/Angstrom, with shape (..., Hf, Wf, 2) where Hf
        and Wf are the frequency dimensions in Fourier space and 2 corresponds to the
        (fx, fy) frequency components. The leading dimension can be batched if there
        are multiple pixel sizes.
    fft_freq_grid_squared : torch.Tensor
        The squared norm of the frequency grid, with shape (..., Hf, Wf), where Hf and
        Wf are the frequency dimensions in Fourier space.
    rho : torch.Tensor
        The radial frequency grid in cycles/Angstrom, with shape (..., Hf, Wf).
    theta : torch.Tensor
        The angular frequency grid in radians, with shape (..., Hf, Wf).
    """
    fft_freq_grid, fft_freq_grid_squared = prepare_frequency_grid_2d(
        image_shape=image_shape,
        pixel_size=pixel_size,
        rfft=rfft,
        fftshift=fftshift,
        device=device,
        transform_matrix=transform_matrix,
    )
    rho, theta = fftfreq_grid_polar(fft_freq_grid)
    return fft_freq_grid, fft_freq_grid_squared, rho, theta


def _setup_ctf_context_2d(
    defocus: float | torch.Tensor,
    astigmatism: float | torch.Tensor,
    astigmatism_angle: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    image_shape: tuple[int, int],
    rfft: bool,
    fftshift: bool,
    transform_matrix: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare full shared 2D CTF context (inputs, grids, and polar coords)."""
    (
        defocus,
        astigmatism,
        astigmatism_angle,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        pixel_size,
        device,
    ) = _prepare_inputs(
        defocus=defocus,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift,
        pixel_size=pixel_size,
    )

    fft_freq_grid, fft_freq_grid_squared, rho, theta = _build_freq_grid(
        image_shape=image_shape,
        pixel_size=pixel_size,
        rfft=rfft,
        fftshift=fftshift,
        device=device,
        transform_matrix=transform_matrix,
    )
    defocus = apply_astigmatism_to_defocus(
        defocus=defocus,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        fft_freq_grid=fft_freq_grid,
        fft_freq_grid_squared=fft_freq_grid_squared,
    )

    return (
        defocus,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        fft_freq_grid,
        fft_freq_grid_squared,
        rho,
        theta,
    )


def _phase_symmetric(
    defocus: torch.Tensor,
    voltage: torch.Tensor,
    spherical_aberration: torch.Tensor,
    amplitude_contrast: torch.Tensor,
    phase_shift: torch.Tensor,
    fft_freq_grid_squared: torch.Tensor,
    rho: torch.Tensor,
    theta: torch.Tensor,
    even_zernike_coeffs: dict | None,
    phase_shift_provider: PhaseShiftProvider | None = None,
    fft_freq_grid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the symmetric phase component of the CTF.

    Parameters
    ----------
    defocus : torch.Tensor
        Defocus in micrometers, with shape (..., 1, 1) for broadcasting.
    voltage : torch.Tensor
        Acceleration voltage in kilovolts (kV), with shape (..., 1, 1) for broadcasting.
    spherical_aberration : torch.Tensor
        Spherical aberration in millimeters (mm), with shape (..., 1, 1) for
        broadcasting.
    amplitude_contrast : torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]), with shape (..., 1, 1)
        for broadcasting.
    phase_shift : torch.Tensor
        Angle of phase shift applied to CTF in degrees, with shape (..., 1, 1) for
        broadcasting.
    fft_freq_grid_squared : torch.Tensor
        Squared frequency grid in Angstroms^-2, with shape (..., Hf, Wf).
    rho : torch.Tensor
        The radial frequency grid in cycles/Angstrom, with shape (..., Hf, Wf).
    theta : torch.Tensor
        The angular frequency grid in radians, with shape (..., Hf, Wf).
    even_zernike_coeffs : dict | None
        Optional dictionary of even Zernike coefficients to apply as additional
        symmetric aberrations. The keys should be the Zernike mode indices
        (n, m) and the values should be the corresponding coefficients in radians.
    phase_shift_provider : PhaseShiftProvider | None
        Optional callable that takes the frequency grid and voltage as input and returns
        a custom phase shift in degrees to use instead of the `phase_shift` input.
    fft_freq_grid : torch.Tensor | None
        The 2D frequency grid in cycles/Angstrom, with shape (..., Hf, Wf, 2).

    Returns
    -------
    total_phase_shift : torch.Tensor
        The total symmetric phase shift of the CTF in radians, with shape (..., Hf, Wf).
    """
    phase_shift_degrees = phase_shift
    if phase_shift_provider is not None:
        if fft_freq_grid is None:
            raise ValueError(
                "fft_freq_grid must be provided when using phase_shift_provider"
            )
        phase_shift_degrees = phase_shift_provider(fft_freq_grid, voltage)

    total_phase_shift = calculate_total_phase_shift(
        defocus_um=defocus,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        phase_shift_degrees=phase_shift_degrees,
        amplitude_contrast_fraction=amplitude_contrast,
        fftfreq_grid_angstrom_squared=fft_freq_grid_squared,
    )
    if even_zernike_coeffs is not None:
        total_phase_shift = apply_even_zernikes(
            even_zernike_coeffs,
            total_phase_shift,
            rho,
            theta,
        )
    return total_phase_shift


def _phase_antisymmetric(
    reference: torch.Tensor,
    voltage: torch.Tensor,
    spherical_aberration: torch.Tensor,
    rho: torch.Tensor,
    theta: torch.Tensor,
    beam_tilt_mrad: torch.Tensor | None,
    odd_zernike_coeffs: dict | None,
) -> torch.Tensor:
    """Compute antisymmetric phase component or zeros when not requested.

    Parameters
    ----------
    reference : torch.Tensor
        Reference tensor to infer the output shape and device for the antisymmetric
        phase shift. The actual values of this tensor are not used in the calculation.
    voltage : torch.Tensor
        Acceleration voltage in kilovolts (kV), with shape (..., 1, 1) for broadcasting.
    spherical_aberration : torch.Tensor
        Spherical aberration in millimeters (mm), with shape (..., 1, 1) for
        broadcasting.
    rho : torch.Tensor
        The radial frequency grid in cycles/Angstrom, with shape (..., Hf, Wf).
    theta : torch.Tensor
        The angular frequency grid in radians, with shape (..., Hf, Wf).
    beam_tilt_mrad : torch.Tensor | None
        Optional beam tilt in milliradians (mrad) to include as an antisymmetric
        aberration, with shape (..., 1, 1) for broadcasting.
    odd_zernike_coeffs : dict | None
        Optional dictionary of odd Zernike coefficients to apply as additional
        antisymmetric aberrations. The keys should be the Zernike mode indices
        (n, m) and the values should be the corresponding coefficients in radians.

    Returns
    -------
    antisymmetric_phase_shift : torch.Tensor
        The antisymmetric phase shift of the CTF in radians, with shape (..., Hf, Wf).
        This will be zero if both `beam_tilt_mrad` and `odd_zernike_coeffs` are None.
    """
    if odd_zernike_coeffs is None and beam_tilt_mrad is None:
        return torch.zeros_like(reference)

    return apply_odd_zernikes(
        odd_zernikes=odd_zernike_coeffs,
        rho=rho,
        theta=theta,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        beam_tilt_mrad=beam_tilt_mrad,
    )


def _render_ctf(
    symmetric_phase_shift: torch.Tensor,
    antisymmetric_phase_shift: torch.Tensor,
    return_complex_ctf: bool,
    include_antisymmetric_phase: bool,
) -> torch.Tensor:
    """Render output CTF tensor for selected output mode.

    Parameters
    ----------
    symmetric_phase_shift : torch.Tensor
        The symmetric phase shift of the CTF in radians, with shape (..., Hf, Wf).
    antisymmetric_phase_shift : torch.Tensor
        The antisymmetric phase shift of the CTF in radians, with shape (..., Hf, Wf).
    return_complex_ctf : bool
        Whether to return the complex-valued CTF (True) or just the real-valued
        sine component (False).
    include_antisymmetric_phase : bool
        Whether the antisymmetric phase shift should be included in the output CTF. If
        False, the antisymmetric phase shift will be ignored and the output CTF will
        be purely real-valued.
    """
    if return_complex_ctf:
        total_phase = symmetric_phase_shift + antisymmetric_phase_shift
        return torch.exp(-1j * total_phase)

    ctf = -torch.sin(symmetric_phase_shift)
    if not include_antisymmetric_phase:
        return ctf
    return ctf * torch.exp(1j * antisymmetric_phase_shift)


def _render_modulated_transfer(
    modulated_transfer: torch.Tensor,
    antisymmetric_phase_shift: torch.Tensor,
    include_antisymmetric_phase: bool,
) -> torch.Tensor:
    """Render pre-modulated transfer output with optional antisymmetric phase."""
    if not include_antisymmetric_phase:
        return modulated_transfer
    return modulated_transfer * torch.exp(1j * antisymmetric_phase_shift)
