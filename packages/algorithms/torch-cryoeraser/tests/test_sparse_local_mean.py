import numpy as np
import pytest
import torch

from torch_cryoeraser.sparse_local_mean import estimate_local_mean


@pytest.mark.parametrize("shape", [(512, 512), (256, 1024), (1024, 256)])
def test_estimate_local_mean_recovers_smooth_background(shape):
    """The spline fit reproduces a smooth background, including non-square images.

    The knots for each axis must be paired with that axis's coordinates; with
    them swapped, a non-square image gets a knot grid stretched to the other
    dimension and the fit error grows several-fold.
    """
    h, w = shape
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    background = np.sin(3 * yy) + 0.5 * np.cos(4 * xx) + yy * xx
    image = torch.tensor(background, dtype=torch.float32)

    np.random.seed(0)  # pixels for the fit are sampled with np.random
    local_mean = estimate_local_mean(image, resolution=(5, 5))

    assert local_mean.shape == image.shape
    rms_error = np.sqrt(((local_mean.numpy() - background) ** 2).mean())
    assert rms_error < 0.005
