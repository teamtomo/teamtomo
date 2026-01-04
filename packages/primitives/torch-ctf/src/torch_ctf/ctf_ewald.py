"""CTF calculation with Ewald sphere correction."""

import einops
import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from torch_ctf.ctf_2d import _setup_ctf_2d
from torch_ctf.ctf_aberrations import (
    apply_even_zernikes,
    apply_odd_zernikes,
    calculate_defocus_phase_aberration,
    calculate_relativistic_electron_wavelength,
)
from torch_ctf.ctf_utils import calculate_total_phase_shift


def calculate_ctfp_and_ctfq_2d(
    defocus: float | torch.Tensor,
    astigmatism: float | torch.Tensor,
    astigmatism_angle: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    image_shape: tuple[int, int],
    rfft: bool = True,
    fftshift: bool = False,
    discontinuity_angle: float = 0.0,
    blur_at_discontinuity: bool = False,
    blur_distance_degrees: float = 5.0,
    beam_tilt_mrad: torch.Tensor | None = None,
    even_zernike_coeffs: dict | None = None,
    odd_zernike_coeffs: dict | None = None,
    transform_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate CTFP and CTFQ for a 2D image.

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
        Default is True (required for Ewald sphere correction).
    fftshift : bool
        Whether to apply fftshift on the resulting CTF images.
    discontinuity_angle : float
        Discontinuity angle in degrees (default: 0.0).
        If > 0, mixes ctfp and ctfq based on angle threshold:
        - ctfp: uses ctfp at angles >= discontinuity_angle,
          ctfq at angles < discontinuity_angle
        - ctfq: uses ctfq at angles >= discontinuity_angle,
          ctfp at angles < discontinuity_angle
        If 0, behavior is unchanged (all ctfp and all ctfq).
        Note: Since ctfp and ctfq are complex conjugates, this mixing
        is equivalent to rotating the coordinate system.
    blur_at_discontinuity : bool
        Whether to apply cosine-weighted blurring at the discontinuity
        (default: False). If True, uses smooth transition instead of sharp.
    blur_distance_degrees : float
        Distance in degrees over which to blur the transition at the
        discontinuity angle (default: 5.0). Uses cosine falloff weighting.
        Only used if blur_at_discontinuity=True.
        Must be <= discontinuity_angle and <= (180 - discontinuity_angle).
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
    ctfp : torch.Tensor
        The P component of the CTF for the given parameters (complex tensor).
    ctfq : torch.Tensor
        The Q component of the CTF for the given parameters (complex tensor).
    """
    # Enforce rfft=True for Ewald sphere correction
    if not rfft:
        raise ValueError("rfft must be True for Ewald sphere correction")

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
    # ctfp: real = -sin, imag = +cos
    ctfp = torch.complex(
        -torch.sin(total_phase_shift),
        torch.cos(total_phase_shift),
    )

    # ctfq: real = -sin, imag = -cos
    ctfq = torch.complex(
        -torch.sin(total_phase_shift),
        -torch.cos(total_phase_shift),
    )

    # Apply odd zernike coefficients if provided
    if odd_zernike_coeffs is not None or beam_tilt_mrad is not None:
        antisymmetric_phase_shift = apply_odd_zernikes(
            odd_zernikes=odd_zernike_coeffs,
            rho=rho,
            theta=theta,
            voltage_kv=voltage,
            spherical_aberration_mm=spherical_aberration,
            beam_tilt_mrad=beam_tilt_mrad,
        )
        ctfp = ctfp * torch.exp(1j * antisymmetric_phase_shift)
        ctfq = ctfq * torch.exp(1j * antisymmetric_phase_shift)

    # Apply discontinuity angle mixing if specified
    if discontinuity_angle > 0.0:
        # Get device from ctfp
        device = ctfp.device

        # Get angles for each pixel
        angles = _get_fourier_angle(
            image_shape=image_shape,
            rfft=rfft,
            fftshift=fftshift,
            device=device,
        )

        # Use blurring if enabled
        if blur_at_discontinuity and blur_distance_degrees > 0.0:
            ctfp_mixed, ctfq_mixed = _mix_at_discontinuity(
                ctfp=ctfp,
                ctfq=ctfq,
                angles=angles,
                discontinuity_angle=discontinuity_angle,
                blur_distance_degrees=blur_distance_degrees,
            )
            return ctfp_mixed, ctfq_mixed

        # Sharp transition (no blurring)
        # Create mask for angles >= discontinuity_angle
        angle_mask = angles >= discontinuity_angle

        # Mix ctfp and ctfq based on angle threshold
        # ctfp: angles >= discontinuity_angle -> ctfp,
        #       angles < discontinuity_angle -> ctfq
        # ctfq: angles >= discontinuity_angle -> ctfq,
        #       angles < discontinuity_angle -> ctfp
        # Note: Since ctfp and ctfq are complex conjugates,
        # this mixing is equivalent to rotating the coordinate system.
        ctfp_mixed = torch.where(angle_mask, ctfp, ctfq)
        ctfq_mixed = torch.where(angle_mask, ctfq, ctfp)

        return ctfp_mixed, ctfq_mixed

    # Enforce Friedel symmetry when discontinuity_angle=0
    if discontinuity_angle == 0.0:
        ctfp = _enforce_symmetry(ctfp, image_shape=image_shape, fftshift=fftshift)
        ctfq = _enforce_symmetry(ctfq, image_shape=image_shape, fftshift=fftshift)

    return ctfp, ctfq


def _enforce_symmetry(
    rfft_tensor: torch.Tensor,
    image_shape: tuple[int, int],
    fftshift: bool = False,
) -> torch.Tensor:
    """Enforce Friedel symmetry along the y-axis at x=0 in rfft.

    For pixels at x=0, +y must equal the complex conjugate of -y.
    This ensures F(0, y) = F*(0, -y) for all y.

    Parameters
    ----------
    rfft_tensor : torch.Tensor
        A complex tensor in rfft format (shape: (..., h, w//2+1)).
    image_shape : tuple[int, int]
        Shape of 2D images (height, width).
    fftshift : bool
        Whether fftshift is applied to the frequency grid.

    Returns
    -------
    rfft_tensor : torch.Tensor
        The tensor with enforced symmetry.
    """
    device = rfft_tensor.device
    eps = 1e-10

    # Get frequency grid
    fft_freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=True,
        fftshift=fftshift,
        norm=False,
        device=device,
    )

    # Get x and y coordinates
    x_coords = fft_freq_grid[..., 0]  # (h, w//2+1) - x frequencies
    y_coords = fft_freq_grid[..., 1]  # (h, w//2+1) - y frequencies

    # Find points at x=0
    x_is_zero = torch.abs(x_coords) < eps

    # Find Friedel redundant points: x=0 and y<0
    y_negative = y_coords < -eps
    friedel_redundant = x_is_zero & y_negative

    rfft_tensor = rfft_tensor.clone()

    # For each point at x=0 with y<0, set it to conjugate of corresponding y>0 point
    # Get indices of redundant points
    redundant_indices = torch.where(friedel_redundant)
    h_neg = redundant_indices[0]
    w_neg = redundant_indices[1]

    # Only process if there are redundant points
    if len(h_neg) > 0:
        # For each negative y point, find its positive y counterpart
        y_neg_vals = y_coords[h_neg, w_neg]
        y_pos_vals = -y_neg_vals

        # Find matching positive y points at x=0
        # Use broadcasting to find all matches
        y_diff = torch.abs(y_coords[None, ...] - y_pos_vals[:, None, None])
        # Only consider points at x=0
        y_diff = torch.where(x_is_zero[None, ...], y_diff, torch.inf)
        # Find minimum difference for each negative y
        min_diff, min_indices_flat = torch.min(y_diff.view(len(y_neg_vals), -1), dim=1)
        # Convert flat index back to (h, w) indices
        _, w_shape = y_coords.shape
        h_pos = min_indices_flat // w_shape
        w_pos = min_indices_flat % w_shape

        # Only process where we found a valid match
        valid_match = min_diff < eps
        if valid_match.any():
            h_neg_valid = h_neg[valid_match]
            w_neg_valid = w_neg[valid_match]
            h_pos_valid = h_pos[valid_match]
            w_pos_valid = w_pos[valid_match]

            # Set negative y values to conjugate of positive y values
            # F(0, y_neg) = F*(0, y_pos)
            rfft_tensor[..., h_neg_valid, w_neg_valid] = torch.conj(
                rfft_tensor[..., h_pos_valid, w_pos_valid]
            )

    return rfft_tensor


def _get_fourier_angle(
    image_shape: tuple[int, int],
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Get the Fourier angle for each pixel in the rfft (0-180 degrees).

    Parameters
    ----------
    image_shape : tuple[int, int]
        Shape of 2D images (height, width).
    rfft : bool
        Whether to use rfft format (default: True).
    fftshift : bool
        Whether fftshift is applied (default: False).
    device : torch.device | None
        Device to create tensors on. If None, uses CPU.

    Returns
    -------
    angle_degrees : torch.Tensor
        Angle in degrees (0-180) for each pixel in the rfft.
        Shape matches the rfft output: (h, w//2+1) if rfft=True.
    """
    if device is None:
        device = torch.device("cpu")

    # Get frequency grid
    fft_freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=False,
        device=device,
    )

    # Calculate angle using atan2 (returns -π to π)
    kx = fft_freq_grid[..., 0]
    ky = fft_freq_grid[..., 1]
    theta = torch.atan2(ky, kx)

    # Convert to 0-180 degrees range
    # For rfft, we want angles from 0 to π (0-180 degrees)
    # Take absolute value to get 0 to π, then convert to degrees
    angle_radians = torch.abs(theta)
    angle_degrees = torch.rad2deg(angle_radians)

    return angle_degrees


def _mix_at_discontinuity(
    ctfp: torch.Tensor,
    ctfq: torch.Tensor,
    angles: torch.Tensor,
    discontinuity_angle: float,
    blur_distance_degrees: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix ctfp and ctfq with cosine falloff at discontinuity angle.

    Parameters
    ----------
    ctfp : torch.Tensor
        The P component of the CTF (complex tensor).
    ctfq : torch.Tensor
        The Q component of the CTF (complex tensor).
    angles : torch.Tensor
        Angle in degrees (0-180) for each pixel in the rfft.
    discontinuity_angle : float
        Discontinuity angle in degrees.
    blur_distance_degrees : float
        Distance in degrees over which to blur the transition.

    Returns
    -------
    ctfp_mixed : torch.Tensor
        The mixed P component.
    ctfq_mixed : torch.Tensor
        The mixed Q component.
    """
    # Validate blur distance
    if blur_distance_degrees > discontinuity_angle:
        raise ValueError(
            f"blur_distance_degrees ({blur_distance_degrees}) must be <= "
            f"discontinuity_angle ({discontinuity_angle})"
        )
    if blur_distance_degrees > (180.0 - discontinuity_angle):
        raise ValueError(
            f"blur_distance_degrees ({blur_distance_degrees}) must be <= "
            f"(180 - discontinuity_angle) = {180.0 - discontinuity_angle}"
        )

    # Calculate transition region boundaries
    angle_low = discontinuity_angle - blur_distance_degrees
    angle_high = discontinuity_angle + blur_distance_degrees

    # Create masks for different regions
    # Below transition: use ctfq for ctfp, ctfp for ctfq
    below_mask = angles < angle_low

    # In transition region: cosine-weighted mixing
    transition_mask = (angles >= angle_low) & (angles <= angle_high)
    # Above transition: use ctfp for ctfp, ctfq for ctfq (no change needed)

    # Calculate cosine weight for transition region
    # Normalize angle to [0, 1] within transition region
    # angle_low -> 0, angle_high -> 1
    angle_normalized = torch.zeros_like(angles)
    angle_normalized[transition_mask] = (angles[transition_mask] - angle_low) / (
        angle_high - angle_low
    )
    # Cosine falloff: 0 at angle_low (full ctfq), 1 at angle_high (full ctfp)
    # Use (1 - cos(π * x)) / 2 for smooth transition from 0 to 1
    cosine_weight = (1.0 - torch.cos(torch.pi * angle_normalized)) / 2.0

    # Initialize mixed tensors
    ctfp_mixed = ctfp.clone()
    ctfq_mixed = ctfq.clone()

    # Apply mixing
    # Below transition: ctfp gets ctfq, ctfq gets ctfp
    ctfp_mixed[below_mask] = ctfq[below_mask]
    ctfq_mixed[below_mask] = ctfp[below_mask]

    # Above transition: ctfp gets ctfp, ctfq gets ctfq (no change)
    # No action needed as they're already correct

    # In transition: weighted mixing
    # ctfp: cosine_weight * ctfp + (1 - cosine_weight) * ctfq
    # ctfq: cosine_weight * ctfq + (1 - cosine_weight) * ctfp
    ctfp_mixed[transition_mask] = (
        cosine_weight[transition_mask] * ctfp[transition_mask]
        + (1.0 - cosine_weight[transition_mask]) * ctfq[transition_mask]
    )
    ctfq_mixed[transition_mask] = (
        cosine_weight[transition_mask] * ctfq[transition_mask]
        + (1.0 - cosine_weight[transition_mask]) * ctfp[transition_mask]
    )

    return ctfp_mixed, ctfq_mixed


def get_ctf_weighting(
    defocus_um: float | torch.Tensor,
    voltage_kv: float | torch.Tensor,
    spherical_aberration_mm: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    particle_diameter_angstrom: float | torch.Tensor,
    image_shape: tuple[int, int],
    rfft: bool = True,
    fftshift: bool = False,
) -> torch.Tensor:
    """Calculate CTF weighting factor W for Ewald sphere correction.

    Implements equation (6) and (7) from the reference:
    W = 1 + A(2|sin(χ)| - 1)

    where A is the degree of overlap between pseudo-Friedel-related sidebands.

    Parameters
    ----------
    defocus_um : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
    voltage_kv : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration_mm : float | torch.Tensor
        Spherical aberration in millimeters (mm).
    pixel_size : float | torch.Tensor
        Pixel size in Angströms per pixel (Å px⁻¹).
    particle_diameter_angstrom : float | torch.Tensor
        Particle diameter in Angstroms.
    image_shape : tuple[int, int]
        Shape of 2D images (height, width).
    rfft : bool
        Whether to use rfft format (default: True).
    fftshift : bool
        Whether fftshift is applied (default: False).

    Returns
    -------
    W : torch.Tensor
        CTF weighting factor with same shape as rfft output: (h, w//2+1) if rfft=True.
    """
    # Determine device
    if isinstance(defocus_um, torch.Tensor):
        device = defocus_um.device
    else:
        device = torch.device("cpu")

    # Convert to tensors
    defocus_um = torch.as_tensor(defocus_um, dtype=torch.float, device=device)
    voltage_kv = torch.as_tensor(voltage_kv, dtype=torch.float, device=device)
    spherical_aberration_mm = torch.as_tensor(
        spherical_aberration_mm, dtype=torch.float, device=device
    )
    pixel_size = torch.as_tensor(pixel_size, dtype=torch.float, device=device)
    particle_diameter_angstrom = torch.as_tensor(
        particle_diameter_angstrom, dtype=torch.float, device=device
    )

    # Get frequency grid in Angstroms^-1
    fft_freq_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=False,
        device=device,
    )
    # Convert from cycles/pixel to cycles/Angstrom
    fft_freq_grid = fft_freq_grid / einops.rearrange(pixel_size, "... -> ... 1 1 1")

    # Calculate frequency magnitude and squared for phase aberration
    fft_freq_grid_squared = einops.reduce(
        fft_freq_grid**2, "... f->...", reduction="sum"
    )

    # Calculate resolution d = 1/|k| in Angstroms
    # Add small epsilon to avoid division by zero
    eps = 1e-12
    freq_magnitude = torch.sqrt(fft_freq_grid_squared + eps)
    resolution_d = 1.0 / freq_magnitude  # Angstroms

    # Calculate phase aberration chi
    chi = calculate_defocus_phase_aberration(
        defocus_um=defocus_um,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
        fftfreq_grid_angstrom_squared=fft_freq_grid_squared,
    )

    # Calculate wavelength lambda in Angstroms
    voltage = voltage_kv * 1e3  # kV -> V
    _lambda = (
        calculate_relativistic_electron_wavelength(voltage) * 1e10
    )  # meters -> Angstroms

    # Convert defocus from micrometers to Angstroms
    deltaF = defocus_um * 1e4  # micrometers -> Angstroms

    # Calculate A based on equation (7)
    # A = (2/π) [arccos(2ΔFλ / (dD)) - (2ΔFλ / (dD)) sin(arccos(2ΔFλ / (dD)))]
    # for 0 < ΔF < Dd/(2λ)
    # A = 0 for ΔF > Dd/(2λ)

    # Calculate threshold: Dd/(2λ)
    threshold = (particle_diameter_angstrom * resolution_d) / (2 * _lambda)

    # Calculate argument: 2ΔFλ / (dD)
    # Reshape deltaF to broadcast with resolution_d
    deltaF = einops.rearrange(deltaF, "... -> ... 1 1")
    _lambda = einops.rearrange(_lambda, "... -> ... 1 1")
    particle_diameter_angstrom = einops.rearrange(
        particle_diameter_angstrom, "... -> ... 1 1"
    )

    arg = (2 * deltaF * _lambda) / (resolution_d * particle_diameter_angstrom)

    # Calculate A piecewise
    # For 0 < ΔF < Dd/(2λ): use formula
    # For ΔF > Dd/(2λ): A = 0
    overlap_condition = (deltaF > 0) & (deltaF < threshold)

    # Calculate A for overlapping case
    arccos_arg = torch.clamp(arg, -1.0, 1.0)  # Ensure valid range for arccos
    arccos_val = torch.arccos(arccos_arg)
    A_overlap = (2 / torch.pi) * (arccos_val - arg * torch.sin(arccos_val))

    # Set A = 0 for no overlap case
    A = torch.where(overlap_condition, A_overlap, torch.tensor(0.0, device=device))

    # Calculate W = 1 + A(2|sin(χ)| - 1)
    sin_chi_abs = torch.abs(torch.sin(chi))
    W = 1.0 + A * (2 * sin_chi_abs - 1.0)

    return W
