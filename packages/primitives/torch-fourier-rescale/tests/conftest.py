import einops
import numpy as np
import pytest
import torch


@pytest.fixture
def circle():
    coordinate_grid = einops.rearrange(np.indices((28, 28)), "yx h w -> h w yx")
    coordinate_grid -= np.array([14, 14])
    return (torch.tensor(np.sum(coordinate_grid**2, axis=-1) ** 0.5) < 6).float()


@pytest.fixture
def sphere():
    coordinate_grid = einops.rearrange(
        np.indices((28, 28, 28)), "zyx d h w -> d h w zyx"
    )
    coordinate_grid -= np.array([14, 14, 14])
    return (torch.tensor(np.sum(coordinate_grid**2, axis=-1) ** 0.5) < 4).float()
