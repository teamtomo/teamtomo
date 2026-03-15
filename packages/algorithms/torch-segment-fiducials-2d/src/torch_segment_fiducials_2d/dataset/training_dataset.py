from os import PathLike
from pathlib import Path
from typing import Tuple

import einops
import imageio
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from torch_segment_fiducials_2d.dataset.augmentation import (
    random_flip,
    random_square_crop_at_scale,
)
from torch_segment_fiducials_2d.constants import TRAINING_IMAGE_DIMENSIONS


class FiducialSegmentationDataset(Dataset):
    """Fiducial segmentation dataset.

    https://zenodo.org/record/7104305
    - Images are in subfolders of root_dir called 'images' and 'masks'
    - dataset on disk are resampled to 8 Å/px
    - dataset are cropped in memory to (512, 512) then normalised to mean=0,
    std=1 before being passed to the network
    - train/eval mode activated via methods of the same name
    """

    def __init__(self, directory: PathLike, train: bool = True, download: bool = True):
        self.dataset_directory = Path(directory)
        self.image_directory = self.dataset_directory / "images"
        self.mask_directory = self.dataset_directory / "masks"

        if download is True:
            self.dataset_directory.mkdir(exist_ok=True, parents=True)
            if self.data_directory_is_empty:
                self._download_data()
        else:  # basic sanity check only
            if (
                len(self.image_files) != len(self.mask_files)
                or len(self.image_files) == 0
            ):
                raise FileNotFoundError(
                    "masks and images directories must contain the same number of images"
                )
        self.train() if train is True else self.eval()
        self._validation_crop_parameters = None

    @property
    def data_directory_is_empty(self):
        return len(list(self.dataset_directory.iterdir())) == 0

    def train(self):
        self._is_training = True

    def eval(self):
        self._is_training = False

    @property
    def image_files(self):
        return sorted(self.image_directory.glob("*.tif"))

    @property
    def mask_files(self):
        return sorted(self.mask_directory.glob("*.tif"))

    def check_files(self, *files: Path):
        for file in files:
            if file.exists() and file.is_file():
                continue
            else:
                raise FileNotFoundError(f"{file} not found")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx].name
        image_file = self.image_directory / filename
        mask_file = self.mask_directory / filename
        self.check_files(image_file, mask_file)

        image = torch.tensor(imageio.imread(image_file), dtype=torch.float32)
        mask = torch.tensor(imageio.imread(mask_file), dtype=torch.float32)
        image, mask = self.preprocess(image, mask)  # (1, h, w), (h, w)
        return image, mask

    def preprocess(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # add channel dim for torchvision transforms
        image = einops.rearrange(image, "h w -> 1 h w")
        mask = einops.rearrange(mask, "h w -> 1 h w")
        if self._is_training:
            image, mask = self.augment(image, mask)
        else:
            image, mask = self.crop_for_validation(image, mask)
        image = self.normalise(image)
        image = image.float().contiguous()
        mask = mask.long().contiguous()
        mask = einops.rearrange(mask, "1 h w -> h w")
        return image, mask

    def augment(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_scale = np.prod(TRAINING_IMAGE_DIMENSIONS) / np.prod(image.shape)
        image, mask = random_square_crop_at_scale(
            image, mask, scale_range=[0.75 * target_scale, 1.33 * target_scale]
        )
        image = TF.resize(
            image,
            size=TRAINING_IMAGE_DIMENSIONS,
            interpolation=TF.InterpolationMode.BICUBIC,
        )
        mask = TF.resize(
            mask,
            size=TRAINING_IMAGE_DIMENSIONS,
            interpolation=TF.InterpolationMode.NEAREST,
        )
        image, mask = random_flip(image, mask, p=0.5)
        return image, mask

    def crop_for_validation(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._validation_crop_parameters is None:
            self._validation_crop_parameters = T.RandomCrop.get_params(
                image, output_size=TRAINING_IMAGE_DIMENSIONS
            )
        image = TF.crop(image, *self._validation_crop_parameters)
        mask = TF.crop(mask, *self._validation_crop_parameters)
        return image, mask

    def normalise(self, image: torch.Tensor) -> torch.Tensor:
        mean, std = torch.mean(image), torch.std(image)
        torch.nan_to_num(image, nan=mean)
        return (image - mean) / std
