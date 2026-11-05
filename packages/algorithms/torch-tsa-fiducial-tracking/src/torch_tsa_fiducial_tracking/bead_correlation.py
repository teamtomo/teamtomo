"""Correlation of tilt images with a gold-bead template."""

import torch
import torch.nn.functional as F
from torch_tilt_series import preprocess_tilt_series_images


def bead_template(
    shape: tuple[int, int], diameter: float, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """A dark disc with a soft edge, centered at `(h // 2, w // 2)`.

    Parameters
    ----------
    shape : tuple[int, int]
        `(h, w)` of the template.
    diameter : float
        Bead diameter in pixels.
    device : torch.device | str
        Device of the returned tensor.

    Returns
    -------
    torch.Tensor
        `(h, w)` template: -1 inside the bead, 0 outside.
    """
    h, w = shape
    y = torch.arange(h, device=device).float() - h // 2
    x = torch.arange(w, device=device).float() - w // 2
    r = torch.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    return -torch.sigmoid((diameter / 2 - r) / 0.5)


def bead_correlation_maps(images: torch.Tensor, diameter: float) -> torch.Tensor:
    """Correlate each image with a dark bead template.

    Images are band-pass filtered (from a quarter of the bead frequency up to
    Nyquist) and correlated with `bead_template`. Each map is divided by its
    robust standard deviation, so values are comparable between tilts: a
    clear bead scores well above 5.

    Parameters
    ----------
    images : torch.Tensor
        `(b, h, w)` tilt images, raw contrast (beads dark).
    diameter : float
        Bead diameter in pixels.

    Returns
    -------
    torch.Tensor
        `(b, h, w)` correlation maps; a bead centered at pixel (i, j) gives a
        peak at (i, j).
    """
    b, h, w = images.shape
    filtered = preprocess_tilt_series_images(
        images, low=min(0.25 / diameter, 0.1), high=0.5, falloff=0.01
    )
    template = bead_template((h, w), diameter, device=images.device)
    template = template - template.mean()
    template = torch.fft.ifftshift(template, dim=(-2, -1))
    correlation = torch.fft.irfft2(
        torch.fft.rfft2(filtered) * torch.conj(torch.fft.rfft2(template)), s=(h, w)
    )
    flat = correlation.reshape(b, -1)
    median = flat.median(dim=-1, keepdim=True).values
    mad = (flat - median).abs().median(dim=-1, keepdim=True).values
    normalized = (flat - median) / (1.4826 * mad).clamp(min=1e-12)
    return normalized.reshape(b, h, w)


def sample_maps(maps: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Bilinear samples of per-tilt maps at per-tilt positions.

    Parameters
    ----------
    maps : torch.Tensor
        `(b, h, w)` maps.
    positions : torch.Tensor
        `(..., b, 2)` yx array coordinates, one per map.

    Returns
    -------
    torch.Tensor
        `(..., b)` sampled values; 0 outside the maps.
    """
    b, h, w = maps.shape
    leading = positions.shape[:-2]
    p = positions.reshape(-1, b, 2).permute(1, 0, 2)  # (b, n, 2)
    grid = torch.stack(
        [2 * p[..., 1] / (w - 1) - 1, 2 * p[..., 0] / (h - 1) - 1], dim=-1
    )
    sampled = F.grid_sample(
        maps[:, None],
        grid[:, :, None, :],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (b, 1, n, 1)
    return sampled[:, 0, :, 0].permute(1, 0).reshape(*leading, b)
