"""Utilities for extracting indices of a grid of patches from an image."""

from collections.abc import Sequence
from typing import cast

import einops
import torch

from ._patch_grid_centers import _patch_centers_1d


def patch_grid_indices(
    image_shape: tuple[int, int] | tuple[int, int, int],
    patch_shape: tuple[int, int] | tuple[int, int, int],
    patch_step: tuple[int, int] | tuple[int, int, int],
    distribute_patches: bool = True,
    device: torch.device = None,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Extract indices of a grid of patches from an image.

    Parameters
    ----------
    image_shape: tuple[int, int] | tuple[int, int, int]
        (h, w) or (d, h, w) shape of the image.
    patch_shape: tuple[int, int] | tuple[int, int, int]
        (patch_h, patch_w) or (patch_d, patch_h, patch_w) of patches to be extracted.
    patch_step: tuple[int, int] | tuple[int, int, int]
        The target distance between patch centers in dimensions `h` and `w`.
    distribute_patches: bool
        Whether to distribute patches across the entire dimension length (`True`)
        or leave a gap at the end of each dimension (`False`).
    device: torch.device, optional
        Device for computation. Default is None, which uses the device of the
        input image.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]
        `(grid_h, grid_w)` array of indices of patch centers
        in image dimensions `h` and `w`.
    """
    parameters_are_valid = len(image_shape) == len(patch_shape) and len(
        image_shape
    ) == len(patch_step)
    if parameters_are_valid is False:
        raise ValueError(
            "image shape, patch length and patch step are not the same length."
        )
    ndim = len(image_shape)
    if ndim == 2:
        return _patch_indices_2d(
            image_shape=image_shape,
            patch_shape=cast(tuple[int, int], patch_shape),
            patch_step=cast(tuple[int, int], patch_step),
            distribute_patches=distribute_patches,
            device=device,
        )
    elif ndim == 3:
        return _patch_indices_3d(
            image_shape=image_shape,
            patch_shape=cast(tuple[int, int, int], patch_shape),
            patch_step=cast(tuple[int, int, int], patch_step),
            distribute_patches=distribute_patches,
            device=device,
        )
    else:
        raise NotImplementedError("only 2D and 3D patches currently supported")


def _patch_centers_to_indices_1d(
    patch_centers: torch.Tensor, patch_length: int, device: torch.device = None
) -> torch.Tensor:
    """
    Convert patch centers to indices.

    Parameters
    ----------
    patch_centers: torch.Tensor
        (..., patch_length) array of patch centers.
    patch_length: int
        Length of the patch.
    device: torch.device, optional
        Device for computation.
        Default is None, which uses the device of the input patch centers.

    Returns
    -------
    torch.Tensor
        (..., patch_length) array of patch indices.
    """
    displacements = torch.arange(patch_length, device=device) - patch_length // 2
    patch_centers = einops.rearrange(patch_centers, "... -> ... 1")
    return patch_centers + displacements  # (..., patch_shape)


def _patch_indices_2d(
    image_shape: Sequence[int],
    patch_shape: tuple[int, int],
    patch_step: tuple[int, int],
    distribute_patches: bool = True,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract indices of a 2D grid of patches from an image.

    Parameters
    ----------
    image_shape: Sequence[int]
        (h, w) shape of the image.
    patch_shape: tuple[int, int]
        `(patch_h, patch_w)` of patches to be extracted.
    patch_step: tuple[int, int]
        The target distance between patch centers in dimensions `h` and `w`.
    distribute_patches: bool
        Whether to distribute patches across the entire dimension length (`True`)
        or leave a gap at the end of each dimension (`False`).
    device: torch.device, optional
        Device for computation. Default is None, which uses the device of the
        input image.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        `(grid_h, grid_w)` array of indices of patch centers
        in image dimensions `h` and `w`.
    """
    centers = [
        _patch_centers_1d(
            dim_length=_dim_length,
            patch_length=_patch_length,
            patch_step=_patch_step,
            distribute_patches=distribute_patches,
            device=device,
        )
        for _dim_length, _patch_length, _patch_step in zip(
            image_shape[-2:], patch_shape, patch_step
        )
    ]
    idx_h, idx_w = (
        _patch_centers_to_indices_1d(
            patch_centers=per_dim_centers,
            patch_length=patch_length,
            device=device,
        )
        for per_dim_centers, patch_length in zip(centers, patch_shape)
    )
    idx_h = einops.rearrange(idx_h, "ph h -> ph 1 h 1")
    idx_w = einops.rearrange(idx_w, "pw w -> 1 pw 1 w")
    return idx_h, idx_w


def _patch_indices_3d(
    image_shape: Sequence[int],
    patch_shape: tuple[int, int, int],
    patch_step: tuple[int, int, int],
    distribute_patches: bool = True,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract indices of a 3D grid of patches from an image.

    Parameters
    ----------
    image_shape: Sequence[int]
        (d, h, w) shape of the image.
    patch_shape: tuple[int, int, int]
        `(patch_d, patch_h, patch_w)` of patches to be extracted.
    patch_step: tuple[int, int, int]
        The target distance between patch centers in dimensions `d`, `h` and `w`.
    distribute_patches: bool
        Whether to distribute patches across the entire dimension length (`True`)
        or leave a gap at the end of each dimension (`False`).
    device: torch.device, optional
        Device for computation. Default is None, which uses the device of the
        input image.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        `(grid_d, grid_h, grid_w)` array of indices of patch centers
        in image dimensions `d`, `h` and `w`.
    """
    centers = [
        _patch_centers_1d(
            dim_length=_dim_length,
            patch_length=_patch_length,
            patch_step=_patch_step,
            distribute_patches=distribute_patches,
            device=device,
        )
        for _dim_length, _patch_length, _patch_step in zip(
            image_shape[-3:], patch_shape, patch_step
        )
    ]
    idx_d, idx_h, idx_w = (
        _patch_centers_to_indices_1d(
            patch_centers=per_dim_centers,
            patch_length=patch_length,
            device=device,
        )
        for per_dim_centers, patch_length in zip(centers, patch_shape)
    )
    idx_d = einops.rearrange(idx_d, "pd d -> pd 1 1 d 1 1")
    idx_h = einops.rearrange(idx_h, "ph h -> 1 ph 1 1 h 1")
    idx_w = einops.rearrange(idx_w, "pw w -> 1 1 pw 1 1 w")
    return idx_d, idx_h, idx_w
