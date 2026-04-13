"""2D CTF calculation functions."""

import torch

from torch_ctf._ctf_core import (
    _phase_antisymmetric,
    _phase_symmetric,
    _render_ctf,
    _setup_ctf_context_2d,
)


def calculate_ctf_2d(
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
    beam_tilt_mrad: torch.Tensor | None = None,
    even_zernike_coeffs: dict | None = None,
    odd_zernike_coeffs: dict | None = None,
    transform_matrix: torch.Tensor | None = None,
    return_complex_ctf: bool = False,
) -> torch.Tensor:
    """Calculate the Contrast Transfer Function (CTF) for a 2D image.

    NOTE: The device of the input tensors is inferred from the `defocus` tensor.

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
        Pixel size in Angströms per pixel (Å px⁻¹).
    image_shape : tuple[int, int]
        Shape of 2D images onto which CTF will be applied.
    rfft : bool
        Generate the CTF containing only the non-redundant half transform from a rfft.
    fftshift : bool
        Whether to apply fftshift on the resulting CTF images.
    beam_tilt_mrad : torch.Tensor | None
        Beam tilt in milliradians. [bx, by] in mrad
    even_zernike_coeffs : dict | None
        Even Zernike coefficients.
        Example: {"Z44c": 0.1, "Z44s": 0.2, "Z60": 0.3}
    odd_zernike_coeffs : dict | None
        Odd Zernike coefficients.
        Example: {"Z31c": 0.1, "Z31s": 0.2, "Z33c": 0.3, "Z33s": 0.4}
    transform_matrix : torch.Tensor | None
        Optional 2x2 transformation matrix for anisotropic magnification.
        This should be the real-space transformation matrix A. The frequency-space
        transformation (A^-1)^T is automatically computed and applied.
    return_complex_ctf : bool
        Whether to return the complex CTF e^(-ichi)

    Returns
    -------
    ctf : torch.Tensor
        The Contrast Transfer Function for the given parameters.
    """
    (
        defocus,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        _,  # fft_freq_grid not used here
        fft_freq_grid_squared,
        rho,
        theta,
    ) = _setup_ctf_context_2d(
        defocus=defocus,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift,
        pixel_size=pixel_size,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        transform_matrix=transform_matrix,
    )

    total_phase_shift = _phase_symmetric(
        defocus=defocus,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift,
        fft_freq_grid_squared=fft_freq_grid_squared,
        rho=rho,
        theta=theta,
        even_zernike_coeffs=even_zernike_coeffs,
    )
    include_antisymmetric_phase = (
        odd_zernike_coeffs is not None or beam_tilt_mrad is not None
    )
    antisymmetric_phase_shift = _phase_antisymmetric(
        odd_zernike_coeffs=odd_zernike_coeffs,
        beam_tilt_mrad=beam_tilt_mrad,
        rho=rho,
        theta=theta,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        reference=total_phase_shift,
    )

    return _render_ctf(
        symmetric_phase_shift=total_phase_shift,
        antisymmetric_phase_shift=antisymmetric_phase_shift,
        return_complex_ctf=return_complex_ctf,
        include_antisymmetric_phase=include_antisymmetric_phase,
    )
