"""Utilities for extracting centers of a grid of patches from an image."""

from collections.abc import Sequence
from typing import cast

import einops
import torch


def patch_grid_centers(
    image_shape: tuple[int, int] | tuple[int, int, int],
    patch_shape: tuple[int, int] | tuple[int, int, int],
    patch_step: tuple[int, int] | tuple[int, int, int],
    distribute_patches: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Extract centers of a grid of patches from an image.

    Parameters
    ----------
    image_shape: tuple[int, int] | tuple[int, int, int]
        (h, w) or (d, h, w) shape of the image.
    patch_shape: tuple[int, int] | tuple[int, int, int]
        `(patch_h, patch_w)` or `(patch_d, patch_h, patch_w)`
        of patches to be extracted.
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
    torch.Tensor
        (..., grid_h, grid_w, 2) array of coordinates of patch centers
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
        return _patch_centers_2d(
            image_shape=image_shape,
            patch_shape=cast("tuple[int, int]", patch_shape),
            patch_step=cast("tuple[int, int]", patch_step),
            distribute_patches=distribute_patches,
            device=device,
        )
    elif ndim == 3:
        return _patch_centers_3d(
            image_shape=image_shape,
            patch_shape=cast("tuple[int, int, int]", patch_shape),
            patch_step=cast("tuple[int, int, int]", patch_step),
            distribute_patches=distribute_patches,
            device=device,
        )
    else:
        raise NotImplementedError("only 2D and 3D patches currently supported")


def _patch_centers_1d(
    dim_length: int,
    patch_length: int,
    patch_step: int,
    distribute_patches: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Extract centers of a 1D grid of patches from an image.

    Parameters
    ----------
    dim_length: int
        Length of the dimension.
    patch_length: int
        Length of the patch.
    patch_step: int
        The target distance between patch centers.
    distribute_patches: bool
        Whether to distribute patches across the entire dimension length (`True`)
        or leave a gap at the end of each dimension (`False`).
    device: torch.device, optional
        Device for computation. Default is None, which uses the device of the
        input image.

    Returns
    -------
    torch.Tensor
        (..., patch_length) array of patch centers.
    """
    min_bound = patch_length // 2
    max_bound = dim_length - min_bound - 1
    if max_bound < min_bound:
        max_bound = min_bound
    patch_centers = torch.arange(
        min_bound, max_bound + 1, step=patch_step, device=device
    )
    if distribute_patches is True:
        delta = max_bound - patch_centers[-1]
        shifts = torch.linspace(0, delta, steps=len(patch_centers), device=device)
        patch_centers += torch.round(shifts).long()
    return patch_centers


def _patch_centers_2d(
    image_shape: Sequence[int],
    patch_shape: tuple[int, int],
    patch_step: tuple[int, int],
    distribute_patches: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Extract centers of a 2D grid of patches from an image.

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
    distribute_patches: bool
    device: torch.device, optional
        Device for computation. Default is None, which uses the device of the
        input image.

    Returns
    -------
    torch.Tensor
        (..., grid_h, grid_w, 2) array of coordinates of patch centers
        in image dimensions `h` and `w`.
    """
    pc_h, pc_w = (
        _patch_centers_1d(
            dim_length=dim_length,
            patch_length=patch_length,
            patch_step=step,
            distribute_patches=distribute_patches,
            device=device,
        )
        for dim_length, patch_length, step in zip(
            image_shape[-2:], patch_shape, patch_step
        )
    )
    n_ph, n_pw = len(pc_h), len(pc_w)
    phh = einops.repeat(pc_h, "ph -> ph pw", pw=n_pw)
    pww = einops.repeat(pc_w, "pw -> ph pw", ph=n_ph)
    patch_centers = einops.rearrange([phh, pww], "hw h w -> h w hw")
    return patch_centers


def _patch_centers_3d(
    image_shape: Sequence[int],
    patch_shape: tuple[int, int, int],
    patch_step: tuple[int, int, int],
    distribute_patches: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Extract centers of a 3D grid of patches from an image.

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
    torch.Tensor
        (..., grid_d, grid_h, grid_w, 3) array of coordinates of patch centers
        in image dimensions `d`, `h` and `w`.
    """
    pc_d, pc_h, pc_w = (
        _patch_centers_1d(
            dim_length=dim_length,
            patch_length=patch_length,
            patch_step=step,
            distribute_patches=distribute_patches,
            device=device,
        )
        for dim_length, patch_length, step in zip(
            image_shape[-3:], patch_shape, patch_step
        )
    )
    n_pd, n_ph, n_pw = len(pc_d), len(pc_h), len(pc_w)
    pdd = einops.repeat(pc_d, "pd -> pd ph pw", ph=n_ph, pw=n_pw)
    phh = einops.repeat(pc_h, "ph -> pd ph pw", pd=n_pd, pw=n_pw)
    pww = einops.repeat(pc_w, "pw -> pd ph pw", pd=n_pd, ph=n_ph)
    patch_centers = einops.rearrange([pdd, phh, pww], "dhw d h w -> d h w dhw")
    return patch_centers
