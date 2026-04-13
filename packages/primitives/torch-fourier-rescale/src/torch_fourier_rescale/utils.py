import numbers

import numpy as np
import torch
from torch.nn import functional as F


def normalize_spacing(spacing: int | float | tuple, ndim: int) -> tuple[float, ...]:
    """Normalize spacing input to a tuple of the correct dimension."""
    if isinstance(spacing, int | float | numbers.Real):
        return (float(spacing),) * ndim
    else:
        return tuple(float(s) for s in spacing)


def calculate_target_shape_from_spacing(
    source_shape: tuple[int, ...],
    source_spacing: tuple[float, ...],
    target_spacing: tuple[float, ...],
) -> tuple[int, ...]:
    """Calculate target shape from source shape and spacing ratio."""
    return tuple(
        int(np.round(src_sh * (src_sp / tgt_sp)))
        for src_sh, src_sp, tgt_sp in zip(source_shape, source_spacing, target_spacing)
    )


def calculate_new_spacing(
    source_spacing: tuple[float, ...],
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[float, ...]:
    """Calculate the actual spacing after rescaling."""
    return tuple(
        src_sp * (src_sh / tgt_sh)
        for src_sp, src_sh, tgt_sh in zip(source_spacing, source_shape, target_shape)
    )


def fourier_rescale_dimension(
    dft: torch.Tensor,
    dim: int,
    source_dim_length: int,
    target_dim_length: int,
    is_rfft: bool = False,
) -> torch.Tensor:
    """Resize a single dimension of a DFT by padding or cropping.

    Parameters
    ----------
    dft : torch.Tensor
        The DFT tensor to resize
    dim : int
        The dimension to resize (negative indexing supported)
    source_dim_length : int
        The original size of the dimension in real space (needed for odd/even handling)
    target_dim_length : int
        The target size of the dimension in real space
    is_rfft : bool
        Whether this is the rfft dimension (positive frequencies only)

    Returns
    -------
    torch.Tensor
        The resized DFT
    """
    if target_dim_length == source_dim_length:
        return dft

    # Normalize dimension index
    if dim < 0:
        dim = dft.ndim + dim

    current_dft_size = dft.shape[dim]

    if is_rfft:
        # For rfft dimension, we only have positive frequencies
        target_dft_size = (target_dim_length // 2) + 1

        if target_dft_size < current_dft_size:
            # Crop: Keep only the required positive frequencies
            indices = [slice(None)] * dft.ndim
            indices[dim] = slice(0, target_dft_size)
            return dft[tuple(indices)]
        else:
            # Pad: Add zeros for higher frequencies
            pad_size = target_dft_size - current_dft_size
            pad_spec = [0, 0] * (dft.ndim - dim - 1) + [0, pad_size] + [0, 0] * dim
            return F.pad(dft, pad_spec, mode="constant", value=0)
    else:
        # For regular FFT dimensions (full frequency spectrum)
        if target_dim_length < source_dim_length:
            # Crop: Remove frequencies symmetrically from both ends
            total_crop = current_dft_size - target_dim_length
            crop_start = total_crop // 2
            crop_end = total_crop - crop_start

            indices = [slice(None)] * dft.ndim
            if crop_end > 0:
                indices[dim] = slice(crop_start, -crop_end)
            else:
                indices[dim] = slice(crop_start, None)
            return dft[tuple(indices)]
        else:
            # Pad: Add zeros symmetrically
            total_pad = target_dim_length - current_dft_size
            pad_start = total_pad // 2
            pad_end = total_pad - pad_start

            # Adjust for odd source size
            if source_dim_length % 2 == 1:
                pad_end = pad_end - 1

            # Create padding specification (PyTorch F.pad expects pairs from last to first dim)
            pad_spec = [0, 0] * (dft.ndim - dim - 1)
            pad_spec.extend([pad_start, pad_end])
            pad_spec.extend([0, 0] * dim)

            return F.pad(dft, pad_spec, mode="constant", value=0)
