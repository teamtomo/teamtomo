"""Wiener filters for 2D images and 3D volumes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_ctf import calculate_total_phase_shift
from torch_grid_utils.fftfreq_grid import fftfreq_grid


def wiener_filter(
    transfer_function: torch.Tensor,
    snr: float | torch.Tensor,
) -> torch.Tensor:
    """Wiener filter for a transfer function.

    Computes ``conj(H) / (|H|^2 + 1 / SNR)``, written as
    ``conj(H) * SNR / (|H|^2 * SNR + 1)`` so that an SNR of 0 gives 0 rather
    than NaN. The filter is on the same grid as ``H``; no frequency grid is
    built here, so any shape and DFT layout (rfft, fftshift) works.

    Parameters
    ----------
    transfer_function: torch.Tensor
        Transfer function ``H``, real or complex, e.g. a CTF from
        ``torch_ctf.calculate_ctf_2d``.
    snr: float | torch.Tensor
        Signal-to-noise ratio, broadcastable against ``transfer_function``:
        a number, one value per batch item (shape ``(b, 1, 1)``), or a grid.
        Must be non-negative.

    Returns
    -------
    torch.Tensor
        The Wiener filter, real if ``transfer_function`` is real.
    """
    h = transfer_function
    snr = torch.as_tensor(snr, dtype=h.real.dtype, device=h.device)
    return h.conj() * snr / (h.abs() ** 2 * snr + 1)


def _deconvolution_snr(
    fraction_of_nyquist: torch.Tensor,
    pixel_size: float,
    strength: torch.Tensor,
    snr_falloff: torch.Tensor,
    highpass_nyquist: torch.Tensor,
) -> torch.Tensor:
    """Model SNR for CTF deconvolution on a pre-computed frequency grid.

    Parameters
    ----------
    fraction_of_nyquist: torch.Tensor
        Frequency grid as a fraction of Nyquist, in ``[0, 1]``.
    pixel_size: float
        Pixel size in Å.
    strength: torch.Tensor
        Log-scale SNR scale; the SNR scale is ``10 ** (3 * strength)``.
    snr_falloff: torch.Tensor
        Rate at which the SNR decays with frequency.
    highpass_nyquist: torch.Tensor
        High-pass cutoff as a fraction of Nyquist.

    Returns
    -------
    torch.Tensor
        The model SNR, broadcast over the grid and parameter shapes.
    """
    ramp = torch.clamp(fraction_of_nyquist / highpass_nyquist, max=1.0)
    highpass = 1 - torch.cos(ramp * torch.pi)
    decay = torch.exp(-fraction_of_nyquist * snr_falloff * 100 / pixel_size)
    return 10 ** (3 * strength) * decay * highpass


def _infer_device(*values: float | torch.Tensor) -> torch.device:
    """Device of the first tensor in ``values``, or the CPU."""
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _as_float_tensor(value: float | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Tensor on ``device``, floating point, keeping autograd history."""
    tensor = torch.as_tensor(value, device=device)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    return tensor


def ctf_deconvolution_filter(
    image_shape: tuple[int, int] | tuple[int, int, int],
    pixel_size: float,
    defocus: float | torch.Tensor,
    voltage: float | torch.Tensor = 300.0,
    spherical_aberration: float | torch.Tensor = 2.7,
    amplitude_contrast: float | torch.Tensor = 0.07,
    phase_shift: float | torch.Tensor = 0.0,
    strength: float | torch.Tensor = 1.0,
    snr_falloff: float | torch.Tensor = 1.0,
    highpass_nyquist: float | torch.Tensor = 0.02,
    phase_flipped: bool = False,
    snr: float | torch.Tensor | None = None,
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """CTF deconvolution filter for 2D images or 3D volumes.

    A Wiener filter with an isotropic CTF and a model SNR::

        filter = -CTF * SNR / (CTF**2 * SNR + 1)
        SNR(r) = 10 ** (3 * strength)
                 * exp(-100 * snr_falloff * r / pixel_size)
                 * (1 - cos(pi * min(1, r / highpass_nyquist)))

    where ``r`` is frequency as a fraction of Nyquist, clamped to 1 in the
    corners of the grid. The high-pass term rises from 0 at zero frequency to 2
    at ``highpass_nyquist`` and above, so the zero frequency is removed and the
    lowest frequencies, where the CTF is weakest, are not over-boosted.

    The CTF here is the negative of ``torch_ctf.calculate_ctf_2d`` (which is
    ``sin(chi)``, dark at underfocus), so the filter keeps the input's
    contrast: features that are dark in the input stay dark. Negate the filter
    to recover the projected potential instead.

    Every CTF and SNR parameter can be a tensor of shape ``(b,)`` (or any
    batch shape; they are broadcast together), giving a filter of shape
    ``(b, *dft_shape)``, so each image or volume in a batch gets its own
    values. The filter is differentiable with respect to all of them, so they
    can be tuned by gradient descent as well as by hand.

    Parameters
    ----------
    image_shape: tuple[int, int] | tuple[int, int, int]
        Real-space shape of the 2D images or 3D volumes.
    pixel_size: float
        Pixel size in Å.
    defocus: float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
    voltage: float | torch.Tensor
        Acceleration voltage in kV.
    spherical_aberration: float | torch.Tensor
        Spherical aberration in mm.
    amplitude_contrast: float | torch.Tensor
        Fraction of amplitude contrast, in ``[0, 1]``.
    phase_shift: float | torch.Tensor
        Additional phase shift in degrees, e.g. from a phase plate.
    strength: float | torch.Tensor
        Overall deconvolution strength on a log scale: the SNR scale is
        ``10 ** (3 * strength)``, so 1.0 gives 1000 and 0.67 gives 100.
        Higher values boost low frequencies more.
    snr_falloff: float | torch.Tensor
        How fast the SNR decays with frequency; higher values attenuate high
        frequencies more. Values around 1.0 to 1.2 are typical.
    highpass_nyquist: float | torch.Tensor
        Cutoff of the high-pass term as a fraction of Nyquist. Must be
        positive.
    phase_flipped: bool
        Whether the data are already phase flipped; if so ``|CTF|`` is used.
    snr: float | torch.Tensor | None
        Measured SNR to use instead of the model (``strength``,
        ``snr_falloff`` and ``highpass_nyquist`` are then ignored), e.g. from
        half-set Fourier shell correlation as ``2 * FSC / (1 - FSC)``.
        Must broadcast against ``(*batch, *dft_shape)``.
    rfft: bool
        Whether the filter is for an rfft (True) or a full fft (False).
    fftshift: bool
        Whether the filter is fftshifted.
    device: torch.device | None
        Device for the filter. Defaults to the device of the first tensor
        parameter, or the CPU.

    Returns
    -------
    torch.Tensor
        Real filter of shape ``(*batch, *dft_shape)``, to multiply with the
        DFT of the images or volumes.
    """
    ndim = len(image_shape)
    if ndim not in (2, 3):
        raise ValueError(f"image_shape must be 2D or 3D, got {image_shape}")
    if device is None:
        device = _infer_device(
            defocus,
            voltage,
            spherical_aberration,
            amplitude_contrast,
            phase_shift,
            strength,
            snr_falloff,
            highpass_nyquist,
            *([snr] if snr is not None else []),
        )

    frequency = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=device,
    )
    fraction_of_nyquist = torch.clamp(2 * frequency, max=1.0)

    # every parameter -> (*batch, 1, ..., 1) so it broadcasts against the grid
    params = torch.broadcast_tensors(
        *(
            _as_float_tensor(p, device)
            for p in (
                defocus,
                voltage,
                spherical_aberration,
                amplitude_contrast,
                phase_shift,
                strength,
                snr_falloff,
                highpass_nyquist,
            )
        )
    )
    (
        defocus,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        strength,
        snr_falloff,
        highpass_nyquist,
    ) = (p.reshape(*p.shape, *([1] * ndim)) for p in params)

    chi = calculate_total_phase_shift(
        defocus_um=defocus,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        phase_shift_degrees=phase_shift,
        amplitude_contrast_fraction=amplitude_contrast,
        fftfreq_grid_angstrom_squared=(fraction_of_nyquist / (2 * pixel_size)) ** 2,
    )
    ctf = -torch.sin(chi)
    if phase_flipped:
        ctf = ctf.abs()

    if snr is None:
        if bool((highpass_nyquist <= 0).any()):
            raise ValueError("highpass_nyquist must be positive")
        snr = _deconvolution_snr(
            fraction_of_nyquist=fraction_of_nyquist,
            pixel_size=pixel_size,
            strength=strength,
            snr_falloff=snr_falloff,
            highpass_nyquist=highpass_nyquist,
        )
    return wiener_filter(ctf, snr)


def _mirror_pad(images: torch.Tensor, pads: list[int]) -> torch.Tensor:
    """Reflect-pad the last ``len(pads)`` dimensions by ``pads`` on each side."""
    if not any(pads):
        return images
    ndim = len(pads)
    batch, spatial = images.shape[:-ndim], images.shape[-ndim:]
    # reflect mode needs a single leading (channel) dimension
    widths = [width for pad in reversed(pads) for width in (pad, pad)]
    padded = F.pad(images.reshape(-1, *spatial), widths, mode="reflect")
    return padded.reshape(*batch, *padded.shape[-ndim:])


def ctf_deconvolve(
    images: torch.Tensor,
    pixel_size: float,
    defocus: float | torch.Tensor,
    *,
    ndim: int,
    voltage: float | torch.Tensor = 300.0,
    spherical_aberration: float | torch.Tensor = 2.7,
    amplitude_contrast: float | torch.Tensor = 0.07,
    phase_shift: float | torch.Tensor = 0.0,
    strength: float | torch.Tensor = 1.0,
    snr_falloff: float | torch.Tensor = 1.0,
    highpass_nyquist: float | torch.Tensor = 0.02,
    phase_flipped: bool = False,
    snr: float | torch.Tensor | None = None,
    padding: float = 0.25,
    preserve_mean_std: bool = True,
) -> torch.Tensor:
    """CTF-deconvolve a batch of 2D images or 3D volumes.

    Applies :func:`ctf_deconvolution_filter` in Fourier space. Each image or
    volume is first mirror-padded on every side and cropped back afterwards.
    Without padding, the FFT treats the image as periodic, and the filter's
    low-frequency boost turns the jump between opposite edges into bright
    bands along them. The filter removes the zero frequency, so by default
    each output is then rescaled to its input mean and standard deviation.

    Parameters
    ----------
    images: torch.Tensor
        Real images ``(..., h, w)`` with ``ndim=2``, or volumes
        ``(..., d, h, w)`` with ``ndim=3``. The leading dimensions must
        broadcast against the batch shape of the parameters, e.g. ``(b, h, w)``
        images with ``(b,)`` defocus values.
    pixel_size: float
        Pixel size in Å.
    defocus: float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
    ndim: int
        2 for images, 3 for volumes. Required, because a stack of images and a
        single volume can have the same shape.
    voltage: float | torch.Tensor
        Acceleration voltage in kV.
    spherical_aberration: float | torch.Tensor
        Spherical aberration in mm.
    amplitude_contrast: float | torch.Tensor
        Fraction of amplitude contrast, in ``[0, 1]``.
    phase_shift: float | torch.Tensor
        Additional phase shift in degrees.
    strength: float | torch.Tensor
        Overall deconvolution strength on a log scale (SNR scale
        ``10 ** (3 * strength)``).
    snr_falloff: float | torch.Tensor
        How fast the SNR decays with frequency.
    highpass_nyquist: float | torch.Tensor
        Cutoff of the high-pass term as a fraction of Nyquist.
    phase_flipped: bool
        Whether the data are already phase flipped.
    snr: float | torch.Tensor | None
        Measured SNR to use instead of the model.
    padding: float
        Mirror padding added to each side, as a fraction of each spatial
        dimension, in ``[0, 1)``. The default of 0.25 makes each dimension 1.5
        times longer. Smaller (e.g. binned) images need relatively more
        padding, since the edge artefacts reach a roughly fixed number of
        pixels into the image. 0 disables padding.
    preserve_mean_std: bool
        Rescale each output to its input's mean and standard deviation.

    Returns
    -------
    torch.Tensor
        Deconvolved images or volumes, same shape as ``images``.
    """
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")
    if not 0 <= padding < 1:
        raise ValueError(f"padding must be in [0, 1), got {padding}")
    dims = tuple(range(-ndim, 0))
    shape = tuple(images.shape[-ndim:])
    pads = [min(round(padding * n), n - 1) for n in shape]
    padded = _mirror_pad(images, pads)
    padded_shape = tuple(padded.shape[-ndim:])
    deconvolution_filter = ctf_deconvolution_filter(
        image_shape=padded_shape,
        pixel_size=pixel_size,
        defocus=defocus,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift,
        strength=strength,
        snr_falloff=snr_falloff,
        highpass_nyquist=highpass_nyquist,
        phase_flipped=phase_flipped,
        snr=snr,
        rfft=True,
        fftshift=False,
        device=images.device,
    )
    dft = torch.fft.rfftn(padded, dim=dims)
    output = torch.fft.irfftn(dft * deconvolution_filter, s=padded_shape, dim=dims)
    crop = tuple(slice(pad, pad + n) for pad, n in zip(pads, shape, strict=True))
    output = output[(..., *crop)]
    if not preserve_mean_std:
        return output

    output_std = output.std(dim=dims, keepdim=True)
    output_std = output_std.clamp_min(torch.finfo(output.dtype).tiny)
    output = (output - output.mean(dim=dims, keepdim=True)) / output_std
    return output * images.std(dim=dims, keepdim=True) + images.mean(
        dim=dims, keepdim=True
    )
