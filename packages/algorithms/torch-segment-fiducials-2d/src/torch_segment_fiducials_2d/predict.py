import warnings
from pathlib import Path
from typing import Optional, Tuple

import einops
import torch
from lightning import Trainer
from torch_fourier_rescale import fourier_rescale_2d

from torch_segment_fiducials_2d.model import get_latest_checkpoint, ResidualUNet18
from torch_segment_fiducials_2d.utils import rescale_2d_bicubic, probabilities_to_mask
from torch_segment_fiducials_2d.constants import TRAINING_PIXEL_SIZE


def predict_fiducial_mask(
    image: torch.Tensor,
    pixel_spacing: float,
    probability_threshold: float,
    model_checkpoint_file: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict fiducial masks for a batch of arbitrarily sized images.

    Parameters
    ----------
    image: torch.Tensor
        `(..., h, w)` array containing image dataset.
    pixel_spacing: float
        Isotropic pixel spacing in Angstroms per pixel.

    Returns
    -------
    mask: torch.Tensor
        `(..., h, w)` masks of each detected fiducial.
    """
    # grab original image dimensions
    h, w = image.shape[-2:]

    # predict probabilities and transform into mask
    probabilities = predict_fiducial_probabilities(
        image=image,
        pixel_spacing=pixel_spacing,
        model_checkpoint_file=model_checkpoint_file,
    )
    mask = probabilities_to_mask(
        probabilities=probabilities,
        threshold=probability_threshold,
        # connected_pixel_count_threshold=(PIXELS_PER_FIDUCIAL // 4),
        connected_pixel_count_threshold=0,
    )
    return mask


def predict_fiducial_probabilities(
    image: torch.Tensor,
    pixel_spacing: float,
    model_checkpoint_file: Optional[Path] = None,
) -> torch.Tensor:
    """Predict fiducial probabilities for a set of images.

    Parameters
    ----------
    image: torch.Tensor
        `(..., h, w)` array containing image(s).
    pixel_spacing: float
        Isotropic pixel spacing in Angstroms per pixel.
    model_checkpoint_file: Optional[Path]
        Model checkpoint file to load.

    Returns
    -------
    probabilities: torch.Tensor
        `(..., h, w)` array containing the probability of each pixel belonging to a fiducial.
    """
    # cast to tensor
    image = torch.as_tensor(image, dtype=torch.float32)

    # prepare model
    if model_checkpoint_file is None:
        model_checkpoint_file = get_latest_checkpoint()
    model = ResidualUNet18.load_from_checkpoint(
        model_checkpoint_file, map_location="cpu"
    )
    model.eval()

    # (..., h, w) -> (b, h, w)
    image, ps = einops.pack([image], pattern="* h w")

    # run inference, one image at a time
    probabilities = [
        _predict_fiducial_probabilities_single_image(
            image=_image,
            pixel_spacing=pixel_spacing,
            model=model,
        )
        for _image in image
    ]
    probabilities = torch.stack(probabilities, dim=0)  # (b, h, w)

    # (b, h, w) -> (..., h, w)
    [probabilities] = einops.unpack(probabilities, packed_shapes=ps, pattern="* h w")
    return probabilities


def _predict_fiducial_probabilities_single_image(
    image: torch.Tensor,  # (h, w)
    pixel_spacing: float,
    model: ResidualUNet18,
) -> torch.Tensor:
    # grab original image dims
    h, w = image.shape[-2:]

    # downscale image
    image, _ = fourier_rescale_2d(
        image, source_spacing=pixel_spacing, target_spacing=TRAINING_PIXEL_SIZE
    )
    image = einops.rearrange(image, "h w -> 1 h w")

    # predict using tiled pred
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="lightning")
        [probabilities] = Trainer(accelerator="auto").predict(model, image)

    # upscale output
    probabilities = rescale_2d_bicubic(probabilities, size=(h, w))
    probabilities = torch.clamp(probabilities, min=0, max=1)
    return probabilities
