"""1D CTF calculation functions."""

import einops
import torch

from torch_ctf.ctf_utils import calculate_total_phase_shift


def calculate_ctf_1d(
    defocus: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    n_samples: int,
    oversampling_factor: int,
) -> torch.Tensor:
    """Calculate the Contrast Transfer Function (CTF) for a 1D signal.

    Parameters
    ----------
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
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
    n_samples : int
        Number of samples in CTF.
    oversampling_factor : int
        Factor by which to oversample the CTF.

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
        fftfreq_grid_squared,
        oversampling_factor,
    ) = _setup_ctf_1d(
        defocus=defocus,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift,
        pixel_size=pixel_size,
        n_samples=n_samples,
        oversampling_factor=oversampling_factor,
    )

    # calculate ctf
    ctf = -torch.sin(
        calculate_total_phase_shift(
            defocus_um=defocus,
            voltage_kv=voltage,
            spherical_aberration_mm=spherical_aberration,
            phase_shift_degrees=phase_shift,
            amplitude_contrast_fraction=amplitude_contrast,
            fftfreq_grid_angstrom_squared=fftfreq_grid_squared,
        )
    )

    if oversampling_factor > 1:
        # reduce oversampling
        ctf = einops.reduce(
            ctf, "... os k -> ... k", reduction="mean"
        )  # oversampling reduction
    return ctf


def _setup_ctf_1d(
    defocus: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    n_samples: int,
    oversampling_factor: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Setup parameters for 1D CTF calculation.

    Parameters
    ----------
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
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
    n_samples : int
        Number of samples in CTF.
    oversampling_factor : int
        Factor by which to oversample the CTF.

    Returns
    -------
    defocus : torch.Tensor
        Defocus tensor.
    voltage : torch.Tensor
        Acceleration voltage tensor.
    spherical_aberration : torch.Tensor
        Spherical aberration tensor.
    amplitude_contrast : torch.Tensor
        Amplitude contrast tensor.
    phase_shift : torch.Tensor
        Phase shift tensor.
    fftfreq_grid_squared : torch.Tensor
        Squared frequency grid in Angstroms^-2.
    oversampling_factor : int
        Oversampling factor for post-processing.
    """
    if isinstance(defocus, torch.Tensor):
        device = defocus.device
    else:
        device = torch.device("cpu")

    # to torch.Tensor
    defocus = torch.as_tensor(defocus, dtype=torch.float, device=device)
    voltage = torch.as_tensor(voltage, dtype=torch.float, device=device)
    spherical_aberration = torch.as_tensor(
        spherical_aberration, dtype=torch.float, device=device
    )
    amplitude_contrast = torch.as_tensor(
        amplitude_contrast, dtype=torch.float, device=device
    )
    phase_shift = torch.as_tensor(phase_shift, dtype=torch.float, device=device)
    pixel_size = torch.as_tensor(pixel_size, dtype=torch.float, device=device)

    # construct frequency vector and rescale cycles / px -> cycles / Å
    fftfreq_grid = torch.linspace(0, 0.5, steps=n_samples)  # (n_samples,
    # oversampling...
    if oversampling_factor > 1:
        frequency_delta = 0.5 / (n_samples - 1)
        oversampled_frequency_delta = frequency_delta / oversampling_factor
        oversampled_interval_length = oversampled_frequency_delta * (
            oversampling_factor - 1
        )
        per_frequency_deltas = torch.linspace(
            0, oversampled_interval_length, steps=oversampling_factor
        )
        per_frequency_deltas -= oversampled_interval_length / 2
        per_frequency_deltas = einops.rearrange(per_frequency_deltas, "os -> os 1")
        fftfreq_grid = fftfreq_grid + per_frequency_deltas

    # Add singletary frequencies according to the dimensions of fftfreq_grid
    expansion_string = "... -> ... " + " ".join(["1"] * fftfreq_grid.ndim)

    pixel_size = einops.rearrange(pixel_size, expansion_string)
    defocus = einops.rearrange(defocus, expansion_string)
    voltage = einops.rearrange(voltage, expansion_string)
    spherical_aberration = einops.rearrange(spherical_aberration, expansion_string)
    phase_shift = einops.rearrange(phase_shift, expansion_string)
    amplitude_contrast = einops.rearrange(amplitude_contrast, expansion_string)

    fftfreq_grid = fftfreq_grid / pixel_size
    fftfreq_grid_squared = fftfreq_grid**2

    return (
        defocus,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        fftfreq_grid_squared,
        oversampling_factor,
    )
