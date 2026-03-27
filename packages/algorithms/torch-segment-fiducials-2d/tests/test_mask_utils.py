import warnings

import torch

warnings.filterwarnings(action="ignore", category=UserWarning, module="tiler")
from torch_segment_fiducials_2d.utils import probabilities_to_mask


def test_probabilities_to_mask():
    # single image
    probs = torch.rand((512, 512))
    mask = probabilities_to_mask(
        probs, threshold=0.5, connected_pixel_count_threshold=0
    )
    assert mask.shape == (512, 512)
    assert torch.allclose(mask, probs > 0.5)

    # arbitrary stack of images
    probs = torch.rand((1, 2, 3, 512, 512))
    mask = probabilities_to_mask(
        probs, threshold=0.5, connected_pixel_count_threshold=0
    )
    assert mask.shape == (1, 2, 3, 512, 512)
    assert torch.allclose(mask, probs > 0.5)
