from typing import Sequence

import einops
import numpy as np
import torch


def coordinate_grid(
    image_shape: Sequence[int],
    center: torch.Tensor | tuple[float, ...] | None = None,
    norm: bool = False,
    device: torch.device | None = None,
) -> torch.FloatTensor:
    """Get a dense grid of array coordinates from grid dimensions.

    For input `image_shape` of `(d, h, w)`, this function produces a
    `(d, h, w, 3)` grid of coordinates. Coordinate order matches the order of
    dimensions in `image_shape`.

    Parameters
    ----------
    image_shape: Sequence[int]
        Shape of the image for which coordinates should be returned.
    center: torch.Tensor | tuple[float, ...] | None
        Array of center points relative to which coordinates will be calculated.
        If `None`, default to the array origin `[0, ...]` of zero in all dimensions.
    norm: bool
        Whether to compute the Euclidean norm of the coordinate grid.
    device: torch.device
        PyTorch device on which to put the coordinate grid.

    Returns
    -------
    grid: torch.LongTensor
        `(*image_shape, image_ndim)` array of coordinates if `norm` is `False`
        else `(*image_shape, )`.
    """
    grid = torch.tensor(
        np.indices(image_shape),
        device=device,
        dtype=torch.float32
    )  # (coordinates, *image_shape)
    grid = einops.rearrange(grid, 'coords ... -> ... coords')
    ndim = len(image_shape)
    if center is not None:
        center = torch.as_tensor(center, dtype=grid.dtype, device=grid.device)
        center = torch.atleast_1d(center)
        center, ps = einops.pack([center], pattern='* coords')
        ones = ' '.join('1' * ndim)
        axis_ids = ' '.join(_unique_characters(ndim))
        center = einops.rearrange(center, f"b coords -> b {ones} coords")
        grid = grid - center
        [grid] = einops.unpack(grid, packed_shapes=ps, pattern=f'* {axis_ids} coords')
    if norm is True:
        grid = einops.reduce(grid ** 2, '... coords -> ...', reduction='sum') ** 0.5
    return grid


def _unique_characters(n: int) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz"
    return chars[:n]


def image_center(
    image_shape: tuple[int, ...],
    device: torch.device | None = None,
) -> torch.LongTensor:
    from torch_grid_utils.fftfreq_grid import dft_center
    center = dft_center(
        image_shape=image_shape,
        rfft=False,
        fftshift=True,
        device=device
    )
    return center
