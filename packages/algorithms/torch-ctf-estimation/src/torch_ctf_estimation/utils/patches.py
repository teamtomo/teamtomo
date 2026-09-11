"""Patch extraction and power spectra for CTF estimation."""

from __future__ import annotations

import einops
import torch
from torch_grid_utils.patch_grid import patch_grid


def extract_ctf_patches(
    image: torch.Tensor,
    patch_sidelength: int,
    defocus_grid_resolution: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
    """Extract overlapping patches or a single whole-image patch.

    Parameters
    ----------
    image : torch.Tensor
        Prepared image of shape ``(t, h, w)``.
    patch_sidelength : int
        Patch size in pixels. Values ``< 0`` select whole-image mode.
    defocus_grid_resolution : tuple[int, int, int]
        ``(nt, nh, nw)``. Whole-image mode requires ``nh == nw == 1``.

    Returns
    -------
    patches : torch.Tensor
        Shape ``(t, gh, gw, ph, pw)``.
    patch_centers : torch.Tensor
        Patch centers in pixel coordinates, shape ``(t, gh, gw, 3)``.
    image_sidelength_for_1d : int
        Sidelength to pass to 1D estimators.
    use_whole_image : bool
        Whether whole-image mode was used.
    """
    _t, h, w = image.shape
    use_whole_image = patch_sidelength < 0
    if not use_whole_image and (h < patch_sidelength or w < patch_sidelength):
        raise ValueError(
            f"Rescaled image size ({h}, {w}) is smaller than patch_sidelength "
            f"({patch_sidelength}). Use a larger image, a smaller "
            "patch_sidelength, or less aggressive rescaling (e.g. "
            "pixel_spacing_angstroms closer to the internal target spacing)."
        )

    if use_whole_image:
        nt, nh, nw = defocus_grid_resolution
        if nh != 1 or nw != 1:
            raise ValueError(
                "When using whole-image mode (patch_sidelength < 0), "
                "defocus_grid_resolution must have nh=1 and nw=1, "
                f"got (nt={nt}, nh={nh}, nw={nw})."
            )
        patches, patch_centers = patch_grid(
            images=image,
            patch_shape=(1, h, w),
            patch_step=(1, h, w),
        )
        image_sidelength_for_1d = min(h, w)
    else:
        patches, patch_centers = patch_grid(
            images=image,
            patch_shape=(1, patch_sidelength, patch_sidelength),
            patch_step=(1, patch_sidelength // 2, patch_sidelength // 2),
        )
        image_sidelength_for_1d = patch_sidelength

    patches = einops.rearrange(patches, "t gh gw 1 ph pw -> t gh gw ph pw")
    return patches, patch_centers, image_sidelength_for_1d, use_whole_image


def compute_patch_power_spectra(
    patches: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rFFT power spectra of patches and the mean spectrum.

    Parameters
    ----------
    patches : torch.Tensor
        Shape ``(t, gh, gw, ph, pw)``.

    Returns
    -------
    patch_ps : torch.Tensor
        Per-patch power spectra, shape ``(t, gh, gw, ph, pw_rfft)``.
    mean_ps : torch.Tensor
        Mean power spectrum, shape ``(ph, pw_rfft)``.
    """
    patch_ps = torch.abs(torch.fft.rfftn(patches, dim=(-2, -1))) ** 2
    mean_ps = einops.reduce(patch_ps, "... ph pw -> ph pw", reduction="mean")
    return patch_ps, mean_ps


def normalised_patch_positions(
    patch_centers: torch.Tensor,
    image_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Normalize patch centers to ``[0, 1]`` using ``(t-1, h-1, w-1)``.

    Parameters
    ----------
    patch_centers : torch.Tensor
        Pixel-space centers, shape ``(t, gh, gw, 3)``.
    image_shape : tuple[int, int, int]
        ``(t, h, w)`` of the prepared image.

    Returns
    -------
    torch.Tensor
        Normalised positions, same shape as ``patch_centers``.
    """
    t, h, w = image_shape
    lengths = torch.tensor(
        [max(t - 1, 1), max(h - 1, 1), max(w - 1, 1)],
        dtype=patch_centers.dtype,
        device=patch_centers.device,
    )
    return patch_centers / lengths
