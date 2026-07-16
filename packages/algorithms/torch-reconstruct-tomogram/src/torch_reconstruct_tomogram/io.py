"""Load and normalize tilt-series image stacks."""

import numpy as np
import torch
from torch_tilt_series import TiltSeries


def _writable(data):
    if isinstance(data, np.ndarray) and not data.flags.writeable:
        data = data.copy()
    return data


def load_tilt_series_images(
    tilt_series: TiltSeries,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Load the raw tilt images matching a TiltSeries' geometry.

    Reads `tilt_series.image_path` and selects/orders rows via
    `tilt_series.image_indices` so the result lines up 1:1 with
    `tilt_series.tilt_angles` etc. Returns unnormalized float32 pixel data at
    `tilt_series.pixel_spacing`; see `normalize_on_central_crop` for the
    central-crop normalization used elsewhere in teamtomo .
    `reconstruct_subvolume`/`reconstruct_tomogram`'s `output_pixel_spacing`
    handles reconstructing at a different pixel size than the raw data's.
    """
    import mrcfile

    if tilt_series.image_path is None:
        raise ValueError(
            "tilt_series.image_path is not set -> construct the TiltSeries "
            "via a torch_tilt_series loader (e.g. from_aretomo_output, "
            "from_etomo_directory) that resolves it, or load images yourself."
        )

    images = mrcfile.read(tilt_series.image_path)
    if tilt_series.image_indices is not None:
        images = images[tilt_series.image_indices.numpy()]

    if device is None:
        device = tilt_series.device
    return torch.as_tensor(_writable(images), device=device).float()


def normalize_on_central_crop(images: torch.Tensor) -> torch.Tensor:
    """Normalize each image to zero mean / unit variance on its central 25% crop."""
    h, w = images.shape[-2:]
    h_crop = slice(int(0.375 * h), int(0.625 * h))
    w_crop = slice(int(0.375 * w), int(0.625 * w))
    crop = images[..., h_crop, w_crop]
    mean = crop.mean(dim=(-2, -1), keepdim=True)
    std = crop.std(dim=(-2, -1), keepdim=True, correction=0)
    return (images - mean) / std
