import numpy as np
import torch

from torch_subpixel_crop import subpixel_crop_3d
from skimage import data

image = torch.tensor(data.binary_blobs(length=128, n_dim=3)).float()
positions = torch.tensor(np.random.uniform(low=0, high=127, size=(100, 3))).float()

crops = subpixel_crop_3d(
    image=image,
    positions=positions,
    sidelength=32
)

# (100, 32, 32, 32)
print(crops.shape)