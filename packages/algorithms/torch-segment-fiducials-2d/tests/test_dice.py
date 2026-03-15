import torch
import torch.nn.functional as F
from einops import rearrange
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="tiler")
from torch_segment_fiducials_2d.model.dice import dice_coefficient


def test_dice_coefficient():
    """Check dice coefficient calculation is correct for single images."""
    a = torch.zeros((10, 10))
    a[5:, 5:] = 1
    a[5:, :5] = 2

    b = torch.zeros((10, 10))
    b[6:, 6:] = 1
    b[6:, :3] = 2

    a = rearrange(F.one_hot(a.long(), num_classes=3), "h w c -> c h w")
    b = rearrange(F.one_hot(b.long(), num_classes=3), "h w c -> c h w")
    dice = dice_coefficient(a, b, batched_input=False)
    assert torch.allclose(dice, torch.tensor([0.82, 0.78, 0.64]), atol=1e-2)


def test_dice_coefficient_batched_input():
    """Batched input should return mean over batch dim."""
    a = torch.zeros((2, 10, 10))
    a[:, 5:, 5:] = 1
    a[:, 5:, :5] = 2

    b = torch.zeros((2, 10, 10))
    b[:, 6:, 6:] = 1
    b[:, 6:, :3] = 2

    a = rearrange(F.one_hot(a.long(), num_classes=3), "b h w c -> b c h w")
    b = rearrange(F.one_hot(b.long(), num_classes=3), "b h w c -> b c h w")
    dice = dice_coefficient(a, b, batched_input=True)
    assert torch.allclose(dice, torch.tensor([0.82, 0.78, 0.64]), atol=1e-2)
