from __future__ import annotations
import numbers
from math import ceil, floor

import numpy as np
import torch
import torch.nn.functional as F

from .utils import get_target_fftfreq


def fourier_rescale_2d(
    image: torch.Tensor,
    source_spacing: float | tuple[float, float],
    target_spacing: float | tuple[float, float],
    preserve_mean: bool = True,
) -> tuple[torch.Tensor, tuple[float, float]]:
    """Rescale 2D image(s) from `source_spacing` to `target_spacing`.

    Rescaling is performed in Fourier space by either cropping or padding the
    discrete Fourier transform (DFT).

    Parameters
    ----------
    image: torch.Tensor
        `(..., h, w)` array of image data
    source_spacing: float | tuple[float, float]
        Pixel spacing in the input image.
    target_spacing: float | tuple[float, float]
        Pixel spacing in the output image.
    preserve_mean: bool = True
        Ensure that the mean (DC component) of the array is preserved after rescaling.

    Returns
    -------
    rescaled_image, (new_spacing_h, new_spacing_w)
    """
    if isinstance(source_spacing, int | float | numbers.Real):
        source_spacing = (source_spacing, source_spacing)
    if isinstance(target_spacing, int | float | numbers.Real):
        target_spacing = (target_spacing, target_spacing)
    if np.allclose(source_spacing, target_spacing):
        return image, source_spacing

    # place image center at array indices [0, 0] and compute centered rfft2
    image = torch.fft.fftshift(image, dim=(-2, -1))
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    dft = torch.fft.fftshift(dft, dim=(-2,))

    # Fourier pad/crop
    dft, new_nyquist, new_shape = fourier_rescale_rfft_2d(
        dft=dft,
        image_shape=image.shape[-2:],
        source_spacing=source_spacing,
        target_spacing=target_spacing,
    )

    # transform back to real space and recenter
    dft = torch.fft.ifftshift(dft, dim=(-2,))
    if preserve_mean:
        # we changed the number of elements in the FT so set norm='forward' to deactivate
        # default fft normalization by 1/n and normalise by the correct factor
        rescaled_image = torch.fft.irfftn(dft, dim=(-2, -1), s=new_shape, norm="forward")
        rescaled_image = rescaled_image * (1 / np.prod(image.shape[-2:]))
    else:
        rescaled_image = torch.fft.irfftn(dft, dim=(-2, -1), s=new_shape)
    rescaled_image = torch.fft.ifftshift(rescaled_image, dim=(-2, -1))

    # Calculate new spacing after rescaling
    source_spacing = np.array(source_spacing, dtype=np.float32)
    new_nyquist = np.array(new_nyquist, dtype=np.float32)
    new_spacing = 1 / (2 * new_nyquist * (1 / np.array(source_spacing)))

    return rescaled_image, tuple(new_spacing)


def fourier_rescale_rfft_2d(
    dft: torch.Tensor,
    image_shape: tuple[int, int],
    source_spacing: tuple[float, float],
    target_spacing: tuple[float, float],
) -> tuple[torch.Tensor, tuple[float, float], tuple[int, int]]:
    h, w = image_shape
    freq_h, freq_w = get_target_fftfreq(source_spacing, target_spacing)
    if freq_h > 0.5:
        dft, nyquist_h, scaled_h = _fourier_pad_h(
            dft, image_height=h, target_fftfreq=freq_h
        )
    else:
        dft, nyquist_h, scaled_h = _fourier_crop_h(
            dft, image_height=h, target_fftfreq=freq_h
        )
    if freq_w > 0.5:
        dft, nyquist_w, scaled_w = _fourier_pad_w(
            dft, image_width=w, target_fftfreq=freq_w
        )
    else:
        dft, nyquist_w, scaled_w = _fourier_crop_w(
            dft, image_width=w, target_fftfreq=freq_w
        )
    return dft, (nyquist_h, nyquist_w), (scaled_h, scaled_w)


def _fourier_crop_h(dft: torch.Tensor, image_height: int, target_fftfreq: float):
    frequencies = torch.fft.fftfreq(image_height)
    frequencies = torch.fft.fftshift(frequencies)
    idx_nyquist = torch.argmin(torch.abs(frequencies - target_fftfreq))
    new_nyquist = frequencies[idx_nyquist]
    idx_h = (frequencies >= -new_nyquist) & (frequencies < new_nyquist)
    new_h = torch.count_nonzero(idx_h)
    return dft[..., idx_h, :], new_nyquist, new_h


def _fourier_crop_w(dft: torch.Tensor, image_width: int, target_fftfreq: float):
    frequencies = torch.fft.rfftfreq(image_width)
    idx_nyquist = torch.argmin(torch.abs(frequencies - target_fftfreq))
    new_nyquist = frequencies[idx_nyquist]
    idx_w = frequencies <= new_nyquist
    new_w = (
        torch.count_nonzero(idx_w) + torch.count_nonzero(frequencies < new_nyquist) - 1
    )
    return dft[..., :, idx_w], new_nyquist, new_w


def _fourier_pad_h(dft: torch.Tensor, image_height: int, target_fftfreq: float):
    delta_fftfreq = 1 / image_height
    idx_nyquist = target_fftfreq / delta_fftfreq
    idx_nyquist = (
        ceil(idx_nyquist) if ceil(idx_nyquist) % 2 == 0 else floor(idx_nyquist)
    )
    new_nyquist = idx_nyquist * delta_fftfreq
    n_frequencies = (dft.shape[-2] // 2) + 1
    pad_h = idx_nyquist - (n_frequencies - 1)
    dft = F.pad(
        dft,
        pad=(0, 0, pad_h, pad_h - (1 if image_height % 2 == 1 else 0)),
        mode="constant",
        value=0,
    )
    new_h = dft.shape[-2]
    return dft, new_nyquist, new_h


def _fourier_pad_w(dft: torch.Tensor, image_width: int, target_fftfreq: float):
    delta_fftfreq = 1 / image_width
    idx_nyquist = target_fftfreq / delta_fftfreq
    idx_nyquist = (
        ceil(idx_nyquist) if ceil(idx_nyquist) % 2 == 0 else floor(idx_nyquist)
    )
    new_nyquist = idx_nyquist * delta_fftfreq
    n_frequencies = dft.shape[-1]
    pad_w = idx_nyquist - (n_frequencies - 1)
    dft = F.pad(dft, pad=(0, pad_w), mode="constant", value=0)
    new_w = image_width + 2 * pad_w - (1 if image_width % 2 == 1 else 0)
    return dft, new_nyquist, new_w
