"""Volume preprocessing: voxel-size normalisation and masking."""

from __future__ import annotations

import torch
from torch_fourier_rescale import fourier_rescale_3d
from torch_grid_utils import fftfreq_grid


def normalise_voxel_sizes(
    reference: torch.Tensor,
    mobile: torch.Tensor,
    reference_pixel_size: float,
    mobile_pixel_size: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Rescale *mobile* to match the voxel size of *reference*.

    The reference is never modified; only mobile is resampled.

    Parameters
    ----------
    reference : torch.Tensor
        ``(d, h, w)`` reference volume.
    mobile : torch.Tensor
        ``(d, h, w)`` mobile volume.
    reference_pixel_size : float
        Voxel size of *reference* in Angstroms.
    mobile_pixel_size : float
        Voxel size of *mobile* in Angstroms.

    Returns
    -------
    reference : torch.Tensor
        Unchanged ``(d, h, w)`` reference.
    mobile_rescaled : torch.Tensor
        Mobile resampled to the reference voxel size.
    common_pixel_size : float
        The common voxel size (= *reference_pixel_size*).
    """
    if abs(reference_pixel_size - mobile_pixel_size) < 1e-6:
        return reference, mobile, reference_pixel_size

    mobile_rescaled, _ = fourier_rescale_3d(
        mobile,
        source_spacing=mobile_pixel_size,
        target_spacing=reference_pixel_size,
    )
    return reference, mobile_rescaled, reference_pixel_size


def crop_or_pad_to_shape(
    volume: torch.Tensor,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Center-crop and/or zero-pad *volume* so its last three dims equal *target_shape*.

    Crop is applied first (in case the volume is both larger in some dims and
    smaller in others), then zero-padding is applied to any remaining deficit.

    Parameters
    ----------
    volume : torch.Tensor
        ``(..., d, h, w)`` input tensor.
    target_shape : tuple[int, int, int]
        Target ``(d, h, w)``.

    Returns
    -------
    torch.Tensor
        Tensor with last three dimensions equal to *target_shape*.
    """
    import torch.nn.functional as F

    # --- crop ---
    slices: list[slice] = [slice(None)] * (volume.ndim - 3)
    for current, target in zip(volume.shape[-3:], target_shape, strict=True):
        if current > target:
            start = (current - target) // 2
            slices.append(slice(start, start + target))
        else:
            slices.append(slice(None))
    result = volume[tuple(slices)]

    # --- pad ---
    cd, ch, cw = result.shape[-3:]
    td, th, tw = target_shape
    pad_w = tw - cw
    pad_h = th - ch
    pad_d = td - cd
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pw_l, pw_r = pad_w // 2, pad_w - pad_w // 2
        ph_l, ph_r = pad_h // 2, pad_h - pad_h // 2
        pd_l, pd_r = pad_d // 2, pad_d - pad_d // 2
        result = F.pad(result, (pw_l, pw_r, ph_l, ph_r, pd_l, pd_r))

    return result


def make_spherical_mask(
    shape: tuple[int, int, int],
    radius_fraction: float = 0.45,
    edge_width_fraction: float = 0.05,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Soft cosine-edge spherical mask.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape ``(d, h, w)``.
    radius_fraction : float
        Sphere radius as a fraction of the half-box size.  Default 0.45.
    edge_width_fraction : float
        Width of the cosine soft-edge as a fraction of the half-box size.
    device : torch.device or None
        Target device.

    Returns
    -------
    mask : torch.Tensor
        ``(d, h, w)`` float32 mask with values in ``[0, 1]``.
    """
    freq_grid = fftfreq_grid(
        image_shape=shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=device,
    )  # (d, h, w), values in [0, 0.5] (normalised spatial frequency = radius / box)
    r = freq_grid * 2.0  # rescale so box-inscribed sphere has radius 1

    inner = radius_fraction - edge_width_fraction / 2
    outer = radius_fraction + edge_width_fraction / 2
    mask = torch.zeros_like(r)
    mask[r <= inner] = 1.0
    in_edge = (r > inner) & (r <= outer)
    mask[in_edge] = 0.5 * (
        1 + torch.cos(torch.pi * (r[in_edge] - inner) / edge_width_fraction)
    )
    return mask


def _normalise_volume(volume: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Zero-mean, unit-variance normalisation within *mask* (or globally)."""
    if mask is not None:
        n = mask.sum().clamp(min=1)
        mean = (volume * mask).sum() / n
        std = ((((volume - mean) * mask) ** 2).sum() / n).sqrt().clamp(min=1e-8)
    else:
        mean = volume.mean()
        std = volume.std().clamp(min=1e-8)
    return (volume - mean) / std
