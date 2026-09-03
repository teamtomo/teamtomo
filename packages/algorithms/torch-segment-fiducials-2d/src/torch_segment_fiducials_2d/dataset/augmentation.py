import random
from typing import List, Tuple

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF


def random_square_crop_at_scale(
    *images: torch.Tensor, scale_range=Tuple[float, float]
) -> List[torch.Tensor]:
    """Apply the same random square crop to multiple images."""
    crop_parameters = T.RandomResizedCrop.get_params(
        img=images[0], scale=scale_range, ratio=[1, 1]
    )
    return [TF.crop(image, *crop_parameters) for image in images]


def random_crop_at_size(
    *images: torch.Tensor, size=Tuple[int, int]
) -> List[torch.Tensor]:
    """Apply the same random crop to multiple images."""
    crop_parameters = T.RandomCrop.get_params(img=images[0], output_size=size)
    return [TF.crop(image, *crop_parameters) for image in images]


def random_flip(*images: torch.Tensor, p=0.5) -> List[torch.Tensor]:
    """Apply the same random flip to multiple images."""
    result = list(images)
    if random.random() > 1 - p:
        result = [TF.hflip(image) for image in result]
    if random.random() > 1 - p:
        result = [TF.vflip(image) for image in result]
    return result


def random_rotation(*images: torch.Tensor, p=0.5) -> List[torch.Tensor]:
    """Apply the same random rotation to multiple images."""
    angle = random.uniform(-180, 180)
    result = list(images)
    if random.random() > 1 - p:
        result = [
            TF.rotate(image, angle=angle, interpolation=TF.InterpolationMode.BILINEAR)
            for image in result
        ]
    return result
