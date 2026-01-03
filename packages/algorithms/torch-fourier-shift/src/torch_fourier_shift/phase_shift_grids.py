from functools import lru_cache

import einops
import torch

from .fftfreq_grids import fftfreq_grid_2d, fftfreq_grid_3d, fftfreq_grid_1d
from .dft_utils import fftshift_1d, fftshift_2d, fftshift_3d


def phase_shift_grid_1d(
    shifts: torch.Tensor,
    image_shape: tuple[int],
    rfft: bool = False,
    fftshift: bool = False,
):
    """Generate arrays of values for phase shifting 1D DFTs by multiplication.

    Parameters
    ----------
    shifts: torch.Tensor
        `(..., )` array of shifts in the last image dimension.
    image_shape: tuple[int]
        Shape `(w, )` of 1D signal(s) on which phase shifts will be applied.
    rfft: bool
        If `True` the phase shifts generated will be compatible with
        the non-redundant half DFT outputs of the FFT for real inputs from `rfft`.
    fftshift: bool
        If `True`, fftshift the output.
    Returns
    -------
    phase_shifts: torch.Tensor
        `(..., w)` complex valued array of phase shifts for the fft or rfft
        of signals with `image_shape`.
    """
    fftfreq_grid = fftfreq_grid_1d(
        image_shape=image_shape, rfft=rfft, device=shifts.device
    )  # (w, )
    shifts = einops.rearrange(shifts, '... -> ... 1')

    # radians/cycle * cycles/sample * samples = radians
    angles = -2 * torch.pi * fftfreq_grid * shifts  # (..., w)
    phase_shifts = torch.complex(real=torch.cos(angles), imag=torch.sin(angles))

    if fftshift is True:
        phase_shifts = fftshift_1d(phase_shifts, rfft=rfft)
    return phase_shifts


def phase_shift_grid_2d(
    shifts: torch.Tensor,
    image_shape: tuple[int, int],
    rfft: bool = False,
    fftshift: bool = False,
):
    """Generate arrays of values for phase shifting 2D DFTs by multiplication.

    Parameters
    ----------
    shifts: torch.Tensor
        `(..., 2)` array of shifts in the last two image dimensions `(h, w)`.
    image_shape: tuple[int, int]
        `(h, w)` of 2D image(s) on which phase shifts will be applied.
    rfft: bool
        If `True` the phase shifts generated will be compatible with
        the non-redundant half DFT outputs of the FFT for real inputs from `rfft`.
    fftshift: bool
        If `True`, fftshift the output.
    Returns
    -------
    phase_shifts: torch.Tensor
        `(..., h, w)` complex valued array of phase shifts for the fft or rfft
        of images with `image_shape`.
    """
    fftfreq_grid = fftfreq_grid_2d(
        image_shape=image_shape, rfft=rfft, device=shifts.device
    )  # (h, w, 2)
    shifts = einops.rearrange(shifts, '... shift -> ... 1 1 shift')

    # radians/cycle * cycles/sample * samples = radians
    angles = einops.reduce(
        -2 * torch.pi * fftfreq_grid * shifts, '... h w 2 -> ... h w', reduction='sum'
    )
    phase_shifts = torch.complex(real=torch.cos(angles), imag=torch.sin(angles))
    if fftshift is True:
        phase_shifts = fftshift_2d(phase_shifts, rfft=rfft)
    return phase_shifts


def phase_shift_grid_3d(
    shifts: torch.Tensor,
    image_shape: tuple[int, int, int],
    rfft: bool = False,
    fftshift: bool = False
):
    """Generate arrays of values for phase shifting 3D DFTs by multiplication.

    Parameters
    ----------
    shifts: torch.Tensor
        `(..., 3)` array of shifts in the last three image dimensions `(d, h, w)`.
    image_shape: tuple[int, int, int]
        `(d, h, w)` of 3D image(s) onto which phase shifts will be applied.
    rfft: bool
        If `True` the phase shifts generated will be compatible with
        the non-redundant half DFT outputs of the FFT for real inputs from `rfft`.
    fftshift: bool
        If `True`, fftshift the output.

    Returns
    -------
    phase_shifts: torch.Tensor
        `(..., d, h, w)` complex valued array of phase shifts for the fft or rfft
        of images with `image_shape`.
    """
    fftfreq_grid = fftfreq_grid_3d(
        image_shape=image_shape, rfft=rfft, device=shifts.device
    )  # (d, h, w, 3)
    shifts = einops.rearrange(shifts, '... shift -> ... 1 1 1 shift')

    # radians/cycle * cycles/sample * samples = radians
    angles = einops.reduce(
        -2 * torch.pi * fftfreq_grid * shifts, '... d h w 3 -> ... d h w', reduction='sum'
    )
    phase_shifts = torch.complex(real=torch.cos(angles), imag=torch.sin(angles))
    if fftshift is True:
        phase_shifts = fftshift_3d(phase_shifts, rfft=rfft)
    return phase_shifts
