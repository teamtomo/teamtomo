"""Prepare a real-space image for CTF estimation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import einops
from torch_fourier_rescale import fourier_rescale_2d

from torch_ctf_estimation.utils.normalize import normalize_image

if TYPE_CHECKING:
    import torch


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
        Requested fitting spacing. The image is never upsampled:
        ``max(target, source)`` is requested. Default 3.0.
    device : torch.device | None, optional
        Device for the output tensor. If None, uses ``image.device``.

    Returns
    -------
    image : torch.Tensor
        Prepared image of shape ``(t, h, w)`` on ``device``.
    pixel_spacing_used : float
        Pixel spacing actually achieved by the rescale. The output size is
        rounded to whole pixels, so this can differ slightly from the requested
        spacing (e.g. 4096 px at 1 Å becomes 1365 px at 3.00073 Å). The mean of
        the h and w spacings is returned, with a warning if they differ by more
        than 1e-3 (relative).
    """
    if device is not None:
        image = image.to(device)
    image = image.float()
    image, _ = einops.pack([image], pattern="* h w")
    image = normalize_image(image)
    requested_spacing = max(target_pixel_spacing_angstroms, pixel_spacing_angstroms)
    image, (spacing_h, spacing_w) = fourier_rescale_2d(
        image=image,
        source_spacing=pixel_spacing_angstroms,
        target_spacing=requested_spacing,
    )
    spacing_h, spacing_w = float(spacing_h), float(spacing_w)
    if abs(spacing_h - spacing_w) > 1e-3 * max(spacing_h, spacing_w):
        warnings.warn(
            f"Rescaled pixel spacing differs between h ({spacing_h:.6f} Å) and w "
            f"({spacing_w:.6f} Å); using their mean for isotropic CTF fitting.",
            stacklevel=2,
        )
    new_spacing = 0.5 * (spacing_h + spacing_w)
    return image, new_spacing
