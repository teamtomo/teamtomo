"""General tilt-series image preprocessing pipeline."""

import einops
import torch
import torch.nn.functional as F
from torch_fourier_filter.bandpass import bandpass_filter
from torch_grid_utils.shapes_2d import rectangle

from torch_tilt_series.utils import normalize_on_central_crop, subtract_plane


def preprocess_tilt_series_images(
    images: torch.Tensor,
    low: float = 0.0,
    high: float = 0.5,
    falloff: float = 0.0,
    bandpass_padding: int = 128,
    subtract_background: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Preprocess tilt-series images for reconstruction.

    Applies, per image: linear background plane subtraction, a bandpass
    filter, and central-crop normalization - similar to the tilt-image
    preprocessing used in Warp.

    To suppress edge artifacts from filtering non-periodic image content,
    the bandpass filter is applied on a mirror-padded copy of each image
    with a soft rectangular mask over the padded border, then cropped back
    to the original size.

    Parameters
    ----------
    images: torch.Tensor
        `(..., h, w)` tilt images.
    low: float
        High-pass cutoff frequency, as a fraction of Nyquist (0-0.5).
        The comparison against `low` is strict, so the default 0.0 already
        excludes the DC (zero-frequency/mean) term.
    high: float
        Low-pass cutoff frequency, as a fraction of Nyquist (0-0.5).
        Default 0.5 (Nyquist). Frequencies are radial (Euclidean norm of the
        per-axis frequencies), so this also clips the diagonal/corner
        frequencies above the axis-aligned Nyquist radius, rather than
        passing the full rfft grid.
    falloff: float
        Width, as a fraction of Nyquist, of the cosine falloff at each
        cutoff. Default 0.0: hard edges, i.e. the filter is binary
        (0.0/1.0) with no soft transition.
    bandpass_padding: int
        Pixels of mirror padding added on each side of each image before
        filtering, and the soft edge width of the rectangular mask applied
        over that padding. The padding is only used internally for the
        bandpass filter step and is cropped back off before returning, so
        the returned images are not padded. Mirror-padding requires
        `bandpass_padding` to be smaller than each spatial dimension, so it
        is silently clamped down to fit small images.
    subtract_background: bool
        If True (default), fit-and-subtract a linear background plane per
        image (see `torch_tilt_series.utils.subtract_plane`) before
        filtering.
    normalize: bool
        If True (default), normalize on the central crop (see
        `torch_tilt_series.utils.normalize_on_central_crop`) after
        filtering.

    Returns
    -------
    images: torch.Tensor
        `(..., h, w)` preprocessed tilt images.
    """
    if subtract_background:
        images = subtract_plane(images)

    h, w = images.shape[-2:]
    pad = max(0, min(bandpass_padding, h - 1, w - 1))
    images_flat, ps = einops.pack([images], "* h w")

    padded = F.pad(images_flat, [pad, pad, pad, pad], mode="reflect")
    mask = rectangle(
        dimensions=(h, w),
        image_shape=padded.shape[-2:],
        smoothing_radius=pad,
        device=images.device,
    )
    padded = padded * mask

    filt = bandpass_filter(
        low=low,
        high=high,
        falloff=falloff,
        image_shape=padded.shape[-2:],
        rfft=True,
        fftshift=False,
        device=images.device,
    )
    padded = torch.fft.irfft2(torch.fft.rfft2(padded) * filt, s=padded.shape[-2:])

    result = padded[..., pad : pad + h, pad : pad + w]
    [result] = einops.unpack(result, ps, "* h w")

    if normalize:
        result = normalize_on_central_crop(result)

    return result
