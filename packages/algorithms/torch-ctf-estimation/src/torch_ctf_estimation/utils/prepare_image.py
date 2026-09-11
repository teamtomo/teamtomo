"""Prepare a real-space image for CTF estimation."""

from __future__ import annotations

import einops
import torch
from torch_fourier_rescale import fourier_rescale_2d

from torch_ctf_estimation.utils.normalize import normalize_image


def prepare_image_for_ctf(
    image: torch.Tensor,
    pixel_spacing_angstroms: float,
    target_pixel_spacing_angstroms: float = 3.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, float]:
    """Pack, normalize, and Fourier-rescale an image for CTF fitting.

    Parameters
    ----------
    image : torch.Tensor
        ``(t, h, w)`` or ``(h, w)`` real-space image.
    pixel_spacing_angstroms : float
        Source pixel spacing in Angstroms.
    target_pixel_spacing_angstroms : float, optional
        Internal fitting spacing. The image is never upsampled:
        ``new_spacing = max(target, source)``. Default 3.0.
    device : torch.device | None, optional
        Device for the output tensor. If None, uses ``image.device``.

    Returns
    -------
    image : torch.Tensor
        Prepared image of shape ``(t, h, w)`` on ``device``.
    pixel_spacing_used : float
        Pixel spacing after rescaling.
    """
    if device is not None:
        image = image.to(device)
    image = image.float()
    image, _ = einops.pack([image], pattern="* h w")
    image = normalize_image(image)
    new_spacing = max(target_pixel_spacing_angstroms, pixel_spacing_angstroms)
    image, _ = fourier_rescale_2d(
        image=image,
        source_spacing=pixel_spacing_angstroms,
        target_spacing=new_spacing,
    )
    return image, new_spacing
