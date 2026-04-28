"""Utils for local resolution estimation.

Batched half-maps use layout ``(B, Z, Y, X)``. This module handles FFT padding,
rFFT preparation, resolution shells, and bandpass multiply + inverse FFT.
"""

from typing import cast

import numpy as np
import torch
from torch_grid_utils import next_fft_size


def estimate_random_maps(
    reference_dist_size: int,
    max_window: float,
    map_size: tuple[int, ...],
) -> int:
    """Estimate the number of permuted maps required for sampling.

    This function estimates how many maps are needed for a target reference
    distribution size, based on the size of the map and the maximum
    sampling window.

    Parameters
    ----------
    reference_dist_size
        Target number of reference distance samples to generate.

    max_window
        Maximum sampling window radius.

    map_size
        Spatial size of the map as a tuple (e.g., ``(Y, X)`` for 2D or
        ``(Z, Y, X)`` for 3D).

    Returns
    -------
    int
        Number of random maps required for the desired number of permutation maps.
    """
    dimension = len(map_size)
    max_window_test = int(np.ceil(max_window)) * 2 + 1
    max_window_entries = np.prod([max_window_test for _ in range(dimension)])
    map_volume = np.prod(np.array(map_size) + max_window_test)
    possible_tests_non_overlapping = int(map_volume / max_window_entries)
    possible_tests_non_overlapping = possible_tests_non_overlapping**2
    return int(np.ceil(reference_dist_size / possible_tests_non_overlapping))


def spatial_shape_from_bzyx(maps_bzyx: torch.Tensor) -> tuple[int, ...]:
    """Spatial dimensions of batched maps in ``(B, Z, Y, X)`` layout.

    If ``Z == 1``, the data are treated as 2D in the plane and the returned shape
    is ``(Y, X)``. Otherwise the full volume shape ``(Z, Y, X)`` is returned.

    Parameters
    ----------
    maps_bzyx
        Tensor with shape ``(B, Z, Y, X)`` (batch, depth, height, width).

    Returns
    -------
    tuple[int, ...]
        Either ``(Y, X)`` when ``Z == 1``, or ``(Z, Y, X)`` when ``Z > 1``.
    """
    _b, z, y, x = maps_bzyx.shape
    if z == 1:
        return (y, x)
    return (z, y, x)


def spatial_map_bzyx(maps_bzyx: torch.Tensor, batch_index: int) -> torch.Tensor:
    """Extract one map from a batch, dropping a singleton depth dimension when 2D.

    Parameters
    ----------
    maps_bzyx
        Tensor with shape ``(B, Z, Y, X)``.
    batch_index
        Which batch element to take, ``0 <= batch_index < B``.

    Returns
    -------
    torch.Tensor
        If ``Z == 1``, shape ``(Y, X)``. If ``Z > 1``, shape ``(Z, Y, X)``.
    """
    slab = maps_bzyx[batch_index]
    if slab.shape[0] == 1:
        return slab[0]
    return slab


def pad_tensor(t: torch.Tensor, target_shape: tuple, mirror_fill: bool) -> torch.Tensor:
    """Pad a tensor to a target shape, centered, with optional mirror fill.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor to pad.
    target_shape : tuple
        Desired output shape.
    mirror_fill : bool
        If True, fill padding with mirrored edge values.
        If False, fill with zeros.

    Returns
    -------
    result : torch.Tensor
        Padded tensor of shape target_shape.
    """
    if mirror_fill:
        # mirror fill breaks spurious structure in edge padding around maps.

        # Add leading dims: F.pad reflect needs: 4D for 2D pad, 5D for 3D pad
        x = t
        ndim = t.ndim
        dims_to_add = max(0, 4 - ndim) if ndim <= 2 else max(0, 5 - ndim)
        for _ in range(dims_to_add):
            x = x.unsqueeze(0)

        # Prepare pad_width for torch.nn.functional.pad
        pad_widths = []
        for s, ts in reversed(list(zip(t.shape, target_shape, strict=False))):
            total_pad = ts - s
            before = total_pad // 2
            after = total_pad - before
            pad_widths.extend([before, after])

        result = torch.nn.functional.pad(x, pad_widths, mode="reflect")

        # Remove the extra leading dims
        for _ in range(dims_to_add):
            result = result.squeeze(0)
    else:
        result = torch.zeros(target_shape, dtype=t.dtype, device=t.device)
    # Center the original tensor inside the larger box.
    slices = tuple(
        slice((ts - s) // 2, (ts - s) // 2 + s)
        for s, ts in zip(t.shape, target_shape, strict=False)
    )
    result[slices] = t
    return result


def prepare_halfmaps_for_fft(
    map1: torch.Tensor,
    map2: torch.Tensor,
    pad: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, tuple, tuple]:
    """Pad to FFT-friendly size and compute ``rfftn`` on both half-maps.

    Parameters
    ----------
    map1 : torch.Tensor
        First half-map as a float32 2D or 3D tensor.
    map2 : torch.Tensor
        Second half-map as a float32 2D or 3D tensor.
    pad : int
        Number of voxels to mirror-pad on each side.
        This is the edge-padding.
    device : str or torch.device
        Device for computation (e.g. ``"cpu"`` or ``"cuda"`` or ``torch.device(...)``).

    Returns
    -------
    fft1 : torch.Tensor
        Complex rfftn output of first half-map.
    fft2 : torch.Tensor
        Complex rfftn output of second half-map.
    padded_image_shape : tuple
        Real-space shape of the padded grid **before** ``rfftn`` (same meaning as
        ``image_shape`` in torch-grid-utils / torch-fourier-filter).
    fft_crop : tuple of slices
        Slices to crop back to original size.
    """
    pad = int(pad)
    if map1.shape != map2.shape:
        msg = f"Half-map shapes must match: {map1.shape} vs {map2.shape}"
        raise ValueError(msg)
    if map1.ndim not in (2, 3):
        msg = f"Expected 2D or 3D maps, got shape {map1.shape}"
        raise ValueError(msg)

    map1 = map1.to(dtype=torch.float32, device=device)
    map2 = map2.to(dtype=torch.float32, device=device)

    # Edge pad by pad (max window radius), then extend each side with
    # torch_grid_utils.next_fft_size for efficient rFFTn. fft_crop maps irFFT back
    # to the radius-padded region only.
    padded_shape = tuple(s + 2 * pad for s in map1.shape)
    padded_image_shape = tuple(next_fft_size(s) for s in padded_shape)
    fft_crop = tuple(
        slice((fft_s - user_s) // 2, (fft_s - user_s) // 2 + user_s)
        for fft_s, user_s in zip(padded_image_shape, padded_shape, strict=False)
    )

    # Pad to padded_shape, then zero-pad to padded_image_shape
    map1_padded = pad_tensor(map1, padded_shape, mirror_fill=True)
    map2_padded = pad_tensor(map2, padded_shape, mirror_fill=True)
    map1_padded = pad_tensor(map1_padded, padded_image_shape, mirror_fill=False)
    map2_padded = pad_tensor(map2_padded, padded_image_shape, mirror_fill=False)

    fft1 = torch.fft.rfftn(map1_padded, dim=list(range(map1.ndim)))
    fft2 = torch.fft.rfftn(map2_padded, dim=list(range(map2.ndim)))

    return fft1, fft2, padded_image_shape, fft_crop


def calculate_shells(
    apix: float,
    spatial_frequencies: np.ndarray,
    spacing_filter: float,
) -> np.ndarray:
    """Calculate bandpass filter shells for a list of spatial frequencies.

    Parameters
    ----------
    apix : float
        Pixel/voxel size in Angstrom.
    spatial_frequencies : np.ndarray
        1D array of spatial frequencies in 1/Angstrom (i.e. 1/resolution).
    spacing_filter : float
        Fractional shell half-width relative to Nyquist.

    Returns
    -------
    shells : np.ndarray
        Array of shape (N, 2), columns are [low_cutoff, high_cutoff] in 1/Angstrom.
    """
    # Nyquist in normalized frequency (cycles/pixel) is 0.5. spacing_filter is
    # fractional shell half-width around each target 1/resolution (1/A).
    nyq = 0.5
    norm_freqs = spatial_frequencies * apix
    low = (norm_freqs - nyq * spacing_filter) / apix
    high = (norm_freqs + nyq * spacing_filter) / apix
    return np.stack([low, high], axis=1)


def apply_bandpass_and_invert(
    fft_map: torch.Tensor,
    bandpass: torch.Tensor,
    padded_image_shape: tuple,
    fft_crop: tuple,
) -> torch.Tensor:
    """Apply a bandpass filter to an rfftn-transformed map and inverse-transform.

    Parameters
    ----------
    fft_map : torch.Tensor
        Complex tensor, output of torch.fft.rfftn.
    bandpass : torch.Tensor
        Bandpass filter.
    padded_image_shape : tuple
        Real-space output shape for ``irfftn`` (same padded grid as before ``rfftn``).
    fft_crop : tuple
        Slices to crop irfftn output back to radius-padded shape.

    Returns
    -------
    result : torch.Tensor
        Real-space tensor cropped to radius-padded shape.
    """
    ndim = fft_map.ndim
    # rFFT last axis is half the width; truncate bandpass to match fft_map.
    bp_rfft = bandpass[..., : fft_map.shape[-1]]
    filtered = fft_map * bp_rfft
    result = torch.fft.irfftn(filtered, s=padded_image_shape, dim=list(range(ndim)))
    return cast("torch.Tensor", result[fft_crop])
