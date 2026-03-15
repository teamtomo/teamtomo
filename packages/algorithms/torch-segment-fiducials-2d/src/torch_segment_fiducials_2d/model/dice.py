import torch
import torch.nn.functional as F
from einops import rearrange, reduce


def dice_coefficient(a, b, batched_input: bool):
    """Multiclass dice coefficient between images.

    Images `a` and `b` should be ordered `b c h w` with `images_are_batched=True`
    or `c h w` with `images_are_batched=False`. The values in each channel
    should be either binary `F.one_hot()` or the result of calling
    `F.softmax()` on logits.

    Parameters
    ----------
    a: torch.Tensor
        `(b, c, h, w)` or `(c, h, w)` array. Values should be one-hot encoded or
        the result of calling `F.softmax()` on logits.
    b: torch.Tensor
        `(b, c, h, w)` or `(c, h, w)` array. Values should be one-hot encoded or
        the result of calling `F.softmax()` on logits.
    batched_input: bool
        Whether the inputs are treated as batches.

    Returns
    -------
    dice_coefficient: torch.Tensor
        `(c, ) array of dice coefficients per channel.
    """
    if batched_input is False:  # add batch dim
        a = rearrange(a, "... -> 1 ...")
        b = rearrange(b, "... -> 1 ...")
    intersection = a * b
    intersection = reduce(intersection, "b c ... -> b c", reduction="sum")
    a = reduce(a, "b c ... -> b c", reduction="sum")
    b = reduce(b, "b c ... -> b c", reduction="sum")
    eps = 1e-6  # avoid potential divide by 0
    dice_coefficient = (2 * intersection + eps) / (a + b + eps)
    return reduce(dice_coefficient, "b c ... -> c ...", reduction="mean")


def dice_score(ground_truth: torch.Tensor, logits: torch.Tensor):
    """Average multiclass dice score over all classes.

    Parameters
    ----------
    ground_truth: torch.Tensor
        `(b, h, w)` array of label images.
    logits: torch.Tensor
        `(b, c, h, w)` array of logits.

    Returns
    -------
    loss: torch.Tensor
    """
    c = logits.shape[-3]
    ground_truth_one_hot = F.one_hot(ground_truth, num_classes=c).float()
    ground_truth_one_hot = rearrange(ground_truth_one_hot, "b h w c -> b c h w")
    probabilities = torch.softmax(logits, dim=1)
    dice = dice_coefficient(probabilities, ground_truth_one_hot, batched_input=True)
    return torch.mean(dice)


def dice_loss(ground_truth: torch.Tensor, logits: torch.Tensor):
    """Optimisation target based on the dice coefficient

    Parameters
    ----------
    ground_truth: torch.Tensor
        `(b, h, w)` array of label images.
    logits: torch.Tensor
        `(b, c, h, w)` array of logits.

    Returns
    -------
    loss: torch.Tensor
    """
    return 1 - dice_score(ground_truth=ground_truth, logits=logits)
