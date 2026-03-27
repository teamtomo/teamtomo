import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage as ndi
from einops import reduce, rearrange


def calculate_resampling_factor(source: float, target: float) -> float:
    """Calculate the resampling factor for two different sampling rates."""
    return float(source / target)


def normalise_2d(image: torch.Tensor) -> torch.Tensor:
    """Normalise 2D image(s) to mean=0 std=1."""
    mean = reduce(image, "... h w -> ... 1 1", reduction="mean")
    std = torch.std(image, dim=[-2, -1])
    std = rearrange(std, "... -> ... 1 1")
    return (image - mean) / std


def central_crop_2d(image: torch.Tensor, percentage: float = 25) -> torch.Tensor:
    """Get a central crop of (a batch of) 2D image(s).

    Parameters
    ----------
    image: torch.Tensor
        `(b, h, w)` or `(h, w)` array of 2D images.
    percentage: float
        percentage of image height and width for cropped region.
    Returns
    -------
    cropped_image: torch.Tensor
        `(b, h, w)` or `(h, w)` array of cropped 2D images.
    """
    h, w = image.shape[-2], image.shape[-1]
    mh, mw = h // 2, w // 2
    dh, dw = int(h * (percentage / 100 / 2)), int(w * (percentage / 100 / 2))
    hf, wf = mh - dh, mw - dw
    hc, wc = mh + dh, mw + dw
    return image[..., hf:hc, wf:wc]


def estimate_background_std(image: torch.Tensor, mask: torch.Tensor):
    """Estimate the standard deviation of the background from a central crop.

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` array containing dataset for which background standard deviation will be estimated.
    mask: torch.Tensor of 0 or 1
        Binary mask separating foreground and background.
    Returns
    -------
    standard_deviation: float
        estimated standard deviation for the background.
    """
    image = central_crop_2d(image, percentage=25).float()
    mask = central_crop_2d(mask, percentage=25)
    return torch.std(image[mask == 0])


def pixel_count_map_2d(mask: torch.Tensor):
    """Calculate a pixel count map from a binary 2D image.

    https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/60_data_visualization/parametric_maps.html

    Parameters
    ----------
    mask: torch.Tensor
        `(h, w)` binary array

    Returns
    -------

    """
    labels, n = ndi.label(mask.cpu().numpy())
    labels = torch.tensor(labels, dtype=torch.long)
    if n < 100:  # vectorised, not memory efficient
        labels_one_hot = F.one_hot(labels, num_classes=(n + 1))
        counts = reduce(labels_one_hot, "h w c -> c", reduction="sum")
        counts = {label_index: count.item() for label_index, count in enumerate(counts)}
        connected_component_image = np.vectorize(counts.__getitem__)(labels)
        connected_component_image = torch.tensor(connected_component_image)
    else:
        connected_component_image = torch.zeros_like(labels)
        for label_id in range(n + 1):
            connected_component_mask = labels == label_id
            n_connected_components = torch.sum(connected_component_mask)
            connected_component_image[connected_component_mask] = n_connected_components
    return connected_component_image
