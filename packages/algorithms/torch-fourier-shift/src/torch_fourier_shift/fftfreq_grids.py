from __future__ import annotations

from typing import Sequence

import einops
import torch

from torch_fourier_shift.dft_utils import rfft_shape

def fftfreq_grid_1d(
    image_shape: tuple[int],
    rfft: bool,
    spacing: float = 1,
    device: torch.device = None
) -> torch.Tensor:
    """Construct a grid of DFT sample freqs for a 1D image.

    Parameters
    ----------
    image_shape: int
        A 1D shape `(w, )` of the input image for which a grid of DFT sample freqs
        should be calculated.
    rfft: bool
        Whether the frequency grid is for a real fft (rfft).
    spacing: float
        Sample spacing in `w` dimension of the grid.
    device: torch.device
        Torch device for the resulting grid.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(w, )` array of DFT sample frequencies.
    """
    dw = spacing
    frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    (w, ) = image_shape
    freq_x = frequency_func(w, d=dw, device=device)
    return freq_x

def fftfreq_grid_2d(
    image_shape: tuple[int, int],
    rfft: bool,
    spacing: float | tuple[float, float] = 1,
    device: torch.device = None
) -> torch.Tensor:
    """Construct a grid of DFT sample freqs for a 2D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 2D shape `(h, w)` of the input image for which a grid of DFT sample freqs
        should be calculated.
    rfft: bool
        Whether the frequency grid is for a real fft (rfft).
    spacing: float | Tuple[float, float]
        Sample spacing in `h` and `w` dimensions of the grid.
    device: torch.device
        Torch device for the resulting grid.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 2)` array of DFT sample freqs.
        Order of freqs in the last dimension corresponds to the order of
        the two dimensions of the grid.
    """
    dh, dw = spacing if isinstance(spacing, Sequence) else [spacing] * 2
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    h, w = image_shape
    freq_y = torch.fft.fftfreq(h, d=dh, device=device)
    freq_x = last_axis_frequency_func(w, d=dw, device=device)
    h, w = rfft_shape(image_shape) if rfft is True else image_shape
    freq_yy = einops.repeat(freq_y, 'h -> h w', w=w)
    freq_xx = einops.repeat(freq_x, 'w -> h w', h=h)
    return einops.rearrange([freq_yy, freq_xx], 'freq h w -> h w freq')


def fftfreq_grid_3d(
    image_shape: Sequence[int],
    rfft: bool,
    spacing: float | tuple[float, float, float] = 1,
    device: torch.device = None
) -> torch.Tensor:
    """Construct a grid of DFT sample freqs for a 3D image.

    Parameters
    ----------
    image_shape: Sequence[int]
        A 3D shape `(d, h, w)` of the input image for which a grid of DFT sample freqs
        should be calculated.
    rfft: bool
        Controls Whether the frequency grid is for a real fft (rfft).
    spacing: float | Tuple[float, float, float]
        Sample spacing in `d`, `h` and `w` dimensions of the grid.
    device: torch.device
        Torch device for the resulting grid.

    Returns
    -------
    frequency_grid: torch.Tensor
        `(h, w, 3)` array of DFT sample freqs.
        Order of freqs in the last dimension corresponds to the order of dimensions
        of the grid.
    """
    dd, dh, dw = spacing if isinstance(spacing, Sequence) else [spacing] * 3
    last_axis_frequency_func = torch.fft.rfftfreq if rfft is True else torch.fft.fftfreq
    d, h, w = image_shape
    freq_z = torch.fft.fftfreq(d, d=dd, device=device)
    freq_y = torch.fft.fftfreq(h, d=dh, device=device)
    freq_x = last_axis_frequency_func(w, d=dw, device=device)
    d, h, w = rfft_shape(image_shape) if rfft is True else image_shape
    freq_zz = einops.repeat(freq_z, 'd -> d h w', h=h, w=w)
    freq_yy = einops.repeat(freq_y, 'h -> d h w', d=d, w=w)
    freq_xx = einops.repeat(freq_x, 'w -> d h w', d=d, h=h)
    return einops.rearrange([freq_zz, freq_yy, freq_xx], 'freq ... -> ... freq')
