import torch

from torch_tiltxcorr.utils import taper_image_edges


def test_taper_image_edges_keeps_center_and_zeros_corners():
    images = torch.ones(2, 40, 40)
    tapered = taper_image_edges(images)
    assert tapered.shape == images.shape
    # flat central region (well within the 90% inner width) is untouched
    assert torch.allclose(tapered[:, 15:25, 15:25], torch.ones(2, 10, 10))
    # the very corners sit outside the taper's smoothing radius -> zeroed
    assert torch.allclose(tapered[:, 0, 0], torch.zeros(2))
    assert torch.allclose(tapered[:, -1, -1], torch.zeros(2))


def test_taper_image_edges_broadcasts_over_leading_dims():
    torch.manual_seed(0)
    images = torch.rand(3, 5, 32, 32)
    tapered = taper_image_edges(images)
    assert tapered.shape == images.shape
    assert torch.allclose(tapered[1, 2], taper_image_edges(images[1, 2]))
