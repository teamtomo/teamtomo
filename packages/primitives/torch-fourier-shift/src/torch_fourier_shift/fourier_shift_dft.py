"""Functions for applying Fourier-domain shifts to DFT arrays."""

import torch

from .phase_shift_grids import (
    phase_shift_grid_1d,
    phase_shift_grid_2d,
    phase_shift_grid_3d,
)


def fourier_shift_dft_1d(
    dft: torch.Tensor,
    image_shape: tuple[int],
    shifts: torch.Tensor,
    rfft: bool,
    fftshifted: bool,
    cache_intermediates: bool = False,
):
    """Apply phase shifts to 2D discrete Fourier transforms.

    Parameters
    ----------
    dft: torch.Tensor
        `(..., w)` array containing DFTs.
    image_shape: tuple[int]
        `(w, )` of images prior to DFT computation.
    shifts: torch.Tensor
        `(...)` array of 2D shifts in `w`.
    rfft: bool
        Whether the input was computed using `rfft`.
    fftshifted: bool
        Whether the DFTs have been fftshifted to center the DC component.
    cache_intermediates: bool
        If `True`, the fftfreq_grid is cached. By default, we use an LRU
        cache with size=3. You can change the size of the cache by setting
        the environmental variable TORCH_FOURIER_SHIFT_CACHE_SIZE before
        importing the module.

    Returns
    -------
    shifted_dfts: torch.Tensor
        `(..., w)` array of DFTs with phase shifts applied.
    """
    phase_shifts = phase_shift_grid_1d(
        shifts=shifts,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshifted,
        cache_intermediates=cache_intermediates,
    )
    return dft * phase_shifts


def fourier_shift_dft_2d(
    dft: torch.Tensor,
    image_shape: tuple[int, int],
    shifts: torch.Tensor,
    rfft: bool,
    fftshifted: bool,
    cache_intermediates: bool = False,
):
    """Apply phase shifts to 2D discrete Fourier transforms.

    Parameters
    ----------
    dft: torch.Tensor
        `(..., h, w)` array containing DFTs.
    image_shape: tuple[int, int]
        `(h, w)` of images prior to DFT computation.
    shifts: torch.Tensor
        `(..., 2)` array of 2D shifts in `h` and `w`.
    rfft: bool
        Whether the input was computed using `rfft`.
    fftshifted: bool
        Whether the DFTs have been fftshifted to center the DC component.
    cache_intermediates: bool
        If `True`, the fftfreq_grid is cached. By default, we use an LRU
        cache with size=3. You can change the size of the cache by setting
        the environmental variable TORCH_FOURIER_SHIFT_CACHE_SIZE before
        importing the module.

    Returns
    -------
    shifted_dfts: torch.Tensor
        `(..., h, w)` array of DFTs with phase shifts applied.
    """
    phase_shifts = phase_shift_grid_2d(
        shifts=shifts,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshifted,
        cache_intermediates=cache_intermediates,
    )
    return dft * phase_shifts


def fourier_shift_dft_3d(
    dft: torch.Tensor,
    image_shape: tuple[int, int, int],
    shifts: torch.Tensor,
    rfft: bool = False,
    fftshifted: bool = False,
    cache_intermediates: bool = False,
):
    """Apply phase shifts to 3D discrete Fourier transforms.

    Parameters
    ----------
    dft: torch.Tensor
        `(..., h, w)` array containing DFTs.
    image_shape: tuple[int, int, int]
        `(h, w)` of images prior to DFT computation.
    shifts: torch.Tensor
        `(..., 3)` array of 3D shifts in `d`, `h` and `w`.
    rfft: bool
        Whether the input was computed using `rfft`.
    fftshifted: bool
        Whether the DFTs have been fftshifted to center the DC component.
    cache_intermediates: bool
        If `True`, the fftfreq_grid is cached. By default, we use an LRU
        cache with size=3. You can change the size of the cache by setting
        the environmental variable TORCH_FOURIER_SHIFT_CACHE_SIZE before
        importing the module.

    Returns
    -------
    shifted_dfts: torch.Tensor
        `(..., h, w)` array of DFTs with phase shifts applied.
    """
    phase_shifts = phase_shift_grid_3d(
        shifts=shifts,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshifted,
        cache_intermediates=cache_intermediates,
    )
    return dft * phase_shifts
