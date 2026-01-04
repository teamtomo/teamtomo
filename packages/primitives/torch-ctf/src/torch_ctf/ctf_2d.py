"""2D CTF calculation functions."""

import einops
import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid, transform_fftfreq_grid
from torch_grid_utils.polar_grid import fftfreq_grid_polar

from torch_ctf.ctf_aberrations import apply_even_zernikes, apply_odd_zernikes
from torch_ctf.ctf_utils import calculate_total_phase_shift


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
        fft_freq_grid_squared,
        rho,
        theta,
    ) = _setup_ctf_2d(
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

    total_phase_shift = calculate_total_phase_shift(
        defocus_um=defocus,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        phase_shift_degrees=phase_shift,
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

    # calculate ctf
    ctf = -torch.sin(total_phase_shift)

    if odd_zernike_coeffs is None and beam_tilt_mrad is None:
        return ctf

    antisymmetric_phase_shift = apply_odd_zernikes(
        odd_zernikes=odd_zernike_coeffs,
        rho=rho,
        theta=theta,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        beam_tilt_mrad=beam_tilt_mrad,
    )
    return ctf * torch.exp(1j * antisymmetric_phase_shift)


def _setup_ctf_2d(
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
]:
    """Setup parameters for 2D CTF calculation.

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
    transform_matrix : torch.Tensor | None
        Optional 2x2 transformation matrix for anisotropic magnification.
        This should be the real-space transformation matrix A. The frequency-space
        transformation (A^-1)^T is automatically computed and applied.

    Returns
    -------
    defocus : torch.Tensor
        Defocus with astigmatism adjustments applied.
    voltage : torch.Tensor
        Acceleration voltage tensor.
    spherical_aberration : torch.Tensor
        Spherical aberration tensor.
    amplitude_contrast : torch.Tensor
        Amplitude contrast tensor.
    phase_shift : torch.Tensor
        Phase shift tensor.
    fft_freq_grid_squared : torch.Tensor
        Squared frequency grid in Angstroms^-2.
    """
    if isinstance(defocus, torch.Tensor):
        device = defocus.device
    else:
        device = torch.device("cpu")

    # to torch.Tensor
    defocus = torch.as_tensor(defocus, dtype=torch.float, device=device)
    astigmatism = torch.as_tensor(astigmatism, dtype=torch.float, device=device)
    astigmatism_angle = torch.as_tensor(
        astigmatism_angle, dtype=torch.float, device=device
    )
    pixel_size = torch.as_tensor(pixel_size, dtype=torch.float, device=device)
    voltage = torch.as_tensor(voltage, dtype=torch.float, device=device)
    spherical_aberration = torch.as_tensor(
        spherical_aberration, dtype=torch.float, device=device
    )
    amplitude_contrast = torch.as_tensor(
        amplitude_contrast, dtype=torch.float, device=device
    )
    phase_shift = torch.as_tensor(phase_shift, dtype=torch.float, device=device)
    image_shape = torch.as_tensor(image_shape, dtype=torch.int, device=device)

    defocus = einops.rearrange(defocus, "... -> ... 1 1")
    voltage = einops.rearrange(voltage, "... -> ... 1 1")
    spherical_aberration = einops.rearrange(spherical_aberration, "... -> ... 1 1")
    amplitude_contrast = einops.rearrange(amplitude_contrast, "... -> ... 1 1")
    phase_shift = einops.rearrange(phase_shift, "... -> ... 1 1")

    # construct 2D frequency grids and rescale cycles / px -> cycles / Å
    fft_freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=False,
        device=device,
    )
    if transform_matrix is not None:
        fft_freq_grid = transform_fftfreq_grid(
            frequency_grid=fft_freq_grid,
            real_space_matrix=transform_matrix,
            device=device,
        )
    fft_freq_grid = fft_freq_grid / einops.rearrange(pixel_size, "... -> ... 1 1 1")
    fft_freq_grid_squared = einops.reduce(
        fft_freq_grid**2, "... f->...", reduction="sum"
    )

    # Calculate the astigmatism vector
    sin_theta = torch.sin(torch.deg2rad(astigmatism_angle))
    cos_theta = torch.cos(torch.deg2rad(astigmatism_angle))
    unit_astigmatism_vector_yx = einops.rearrange(
        [sin_theta, cos_theta], "yx ... -> ... yx"
    )
    astigmatism = einops.rearrange(astigmatism, "... -> ... 1")
    # Multiply with the square root of astigmatism
    # to get the right amplitude after squaring later
    astigmatism_vector = torch.sqrt(astigmatism) * unit_astigmatism_vector_yx
    # Calculate unitvectors from the frequency grids
    # Reuse already computed fft_freq_grid_squared to avoid redundant pow operations
    fft_freq_grid_norm = torch.sqrt(
        einops.rearrange(fft_freq_grid_squared, "... -> ... 1")
        + torch.finfo(torch.float32).eps
    )
    direction_unitvector = fft_freq_grid / fft_freq_grid_norm
    # Subtract the astigmatism from the defocus
    defocus -= einops.rearrange(astigmatism, "... -> ... 1")
    # Add the squared dotproduct between the direction unitvector
    # and the astigmatism vector
    defocus = (
        defocus
        + einops.einsum(
            direction_unitvector, astigmatism_vector, "... h w f, ... f -> ... h w"
        )
        ** 2
        * 2
    )

    # get polar coordinates
    rho, theta = fftfreq_grid_polar(fft_freq_grid)

    return (
        defocus,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        fft_freq_grid_squared,
        rho,
        theta,
    )
