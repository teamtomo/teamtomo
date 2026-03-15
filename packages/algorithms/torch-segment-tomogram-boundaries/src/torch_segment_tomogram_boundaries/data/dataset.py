# src/torch_segment_tomogram_boundaries/data/dataset.py

from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from torch_segment_tomogram_boundaries import config
from torch_segment_tomogram_boundaries.data.weight_maps import (
    generate_boundary_weight_map,
)

TransformType = Optional[Callable[[np.ndarray, np.ndarray], Dict[str, np.ndarray]]]


class PTFileDataset(Dataset):
    def __init__(self, pt_file_paths: List[Path], transform: TransformType = None):
        self.pt_file_paths = pt_file_paths
        self.transform = transform
        self.to_tensor = ToTensorV2()

    def __len__(self) -> int:
        return len(self.pt_file_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pt_path = self.pt_file_paths[idx]
        data = torch.load(pt_path, map_location="cpu")

        image_np = data["image"].numpy().transpose(1, 2, 0)
        label_np = data["label"].numpy().squeeze()

        if self.transform:
            transformed = self.transform(image=image_np, mask=label_np)
            image_np = transformed["image"]
            label_np = transformed["mask"]

        label_binary_np = (label_np > 0.5).astype(np.float32)

        if config.USE_GAUSSIAN_LABEL_SMOOTHING and config.GAUSSIAN_LABEL_SIGMA > 0:
            soft_label_np = gaussian_filter(
                label_binary_np, sigma=config.GAUSSIAN_LABEL_SIGMA
            )
            soft_label_np = np.clip(soft_label_np, 0.0, 1.0)
        else:
            soft_label_np = label_binary_np

        weight_map_np = generate_boundary_weight_map(
            label_binary_np,
            boundary_width=config.WEIGHT_MAP_BOUNDARY_WIDTH,
        )

        image_tensor = self.to_tensor(image=image_np)["image"]
        label_tensor = self.to_tensor(image=label_binary_np)["image"].float()
        soft_label_tensor = self.to_tensor(image=soft_label_np)["image"].float()
        weight_map_tensor = self.to_tensor(image=weight_map_np)["image"]

        return {
            "image": image_tensor.float(),
            "label": label_tensor,
            "soft_label": soft_label_tensor,
            "weight_map": weight_map_tensor.float(),
        }
