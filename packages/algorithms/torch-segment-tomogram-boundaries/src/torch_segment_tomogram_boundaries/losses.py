"""Loss functions for tomographic boundary segmentation.

This module provides specialized loss functions designed for boundary detection
in 3D tomographic data, including weighted losses, boundary-focused losses,
and flexible loss combination utilities.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


class SimpleDiceLoss(nn.Module):
    """
    Simplified MONAI DiceLoss wrapper.
    """

    def __init__(self, from_logits: bool = True):
        super().__init__()
        self.dice_loss = DiceLoss(
            sigmoid=from_logits, squared_pred=True, reduction="mean"
        )
        self.name = "DiceLoss"

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.dice_loss(pred_logits, target)


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss with pixel-wise weighting.

    This loss applies different weights to different pixels based on a
    pre-computed weight map, allowing emphasis on important regions
    (e.g., boundaries) while de-emphasizing others.

    Parameters
    ----------
    from_logits : bool, default=True
        Whether input predictions are logits (True) or probabilities (False).
    """

    def __init__(self, from_logits: bool = True, label_smoothing: float = 0.0):
        super().__init__()
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.name = "WeightedBCE"

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            target_smooth = (
                target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            )
        else:
            target_smooth = target

        if self.from_logits:
            loss = F.binary_cross_entropy_with_logits(
                pred_logits, target_smooth.float(), reduction="none"
            )
        else:
            loss = F.binary_cross_entropy(
                torch.sigmoid(pred_logits), target_smooth.float(), reduction="none"
            )

        if weight_map is not None:
            weighted_loss = loss * weight_map
            return weighted_loss.mean()
        else:
            return loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary-focused loss that emphasizes boundary pixels over regions.

    This loss automatically detects boundary pixels using morphological operations
    and applies higher weight to boundary regions while maintaining coverage
    of non-boundary areas.

    Parameters
    ----------
    from_logits : bool, default=True
        Whether input predictions are logits (True) or probabilities (False).
    boundary_weight_factor : float, default=10.0
        Multiplier for boundary pixel loss relative to region pixels.
    """

    def __init__(self, from_logits: bool = True, boundary_weight_factor: float = 10.0):
        super().__init__()
        self.from_logits = from_logits
        self.boundary_weight_factor = boundary_weight_factor
        self.name = "BoundaryLoss"

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.from_logits:
            loss_pixels = F.binary_cross_entropy_with_logits(
                pred_logits, target.float(), reduction="none"
            )
        else:
            loss_pixels = F.binary_cross_entropy(
                torch.sigmoid(pred_logits), target.float(), reduction="none"
            )
        max_pool = F.max_pool2d(target.float(), kernel_size=3, stride=1, padding=1)
        min_pool = -F.max_pool2d(-target.float(), kernel_size=3, stride=1, padding=1)
        boundary_mask = (max_pool - min_pool).bool()
        non_boundary_mask = ~boundary_mask
        loss_boundary = (loss_pixels * boundary_mask.float()).sum()
        num_boundary = boundary_mask.sum().clamp(min=1.0)
        mean_loss_boundary = loss_boundary / num_boundary
        loss_region = (loss_pixels * non_boundary_mask.float()).sum()
        num_region = non_boundary_mask.sum().clamp(min=1.0)
        mean_loss_region = loss_region / num_region
        return self.boundary_weight_factor * mean_loss_boundary + mean_loss_region


def gradient2d_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the L1 loss between the gradients of the prediction and the target."""
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device
    ).view(1, 1, 3, 3)

    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)

    return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)


class WeightedHuberWithGradientLoss(nn.Module):
    """
    A combined loss function that includes a weighted Huber loss and a 2D gradient loss.
    """

    def __init__(
        self, from_logits: bool = True, eps: float = 1.0, gradient_weight: float = 10.0
    ):
        super().__init__()
        self.from_logits = from_logits
        self.eps = eps
        self.gradient_weight = gradient_weight
        self.name = "WeightedHuberWithGradientLoss"

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.from_logits:
            y_pred = torch.sigmoid(pred_logits)
        else:
            y_pred = pred_logits

        y_true = target.float()

        num_positive = torch.sum(y_true, dim=(1, 2, 3))
        num_negative = (y_true.size(1) * y_true.size(2) * y_true.size(3)) - num_positive

        positive_weights = (y_true.size(1) * y_true.size(2) * y_true.size(3)) / (
            num_positive + self.eps
        )
        negative_weights = (y_true.size(1) * y_true.size(2) * y_true.size(3)) / (
            num_negative + self.eps
        )

        positive_weights = positive_weights.view(y_true.shape[0], 1, 1, 1)
        negative_weights = negative_weights.view(y_true.shape[0], 1, 1, 1)

        importance_mask = torch.where(
            y_true > 0, positive_weights, 2 * negative_weights
        )

        huber_loss = importance_mask * F.huber_loss(y_pred, y_true, reduction="none")

        grad_loss = self.gradient_weight * gradient2d_loss(y_pred, y_true)

        loss = huber_loss.mean() + grad_loss
        return loss


class CombinedLoss(nn.Module):
    """
    Flexible multi-component loss function with weighted combination.

    This loss allows combining multiple loss functions (e.g., Dice + BCE)
    with custom weights to leverage benefits of different loss types.

    Parameters
    ----------
    losses_dict : Dict[str, Tuple[nn.Module, float]]
        Dictionary mapping loss names to (loss_function, weight) tuples.
    from_logits : bool, default=True
        Whether input predictions are logits (True) or probabilities (False).
    """

    def __init__(
        self, losses_dict: Dict[str, Tuple[nn.Module, float]], from_logits: bool = True
    ):
        super().__init__()
        self.losses = nn.ModuleDict(
            {name: loss for name, (loss, weight) in losses_dict.items()}
        )
        self.weights = {name: weight for name, (loss, weight) in losses_dict.items()}
        self.from_logits = from_logits
        self.name = "+".join(losses_dict.keys())

    def forward(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        weight_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=pred_logits.device)
        for name, loss_fn in self.losses.items():
            # Pass weight_map to all component losses, they will handle None appropriately
            component_loss = loss_fn(pred_logits, target, weight_map)
            total_loss += self.weights[name] * component_loss
        return total_loss


def get_loss_function(loss_config: dict) -> nn.Module:
    """
    Factory function to create loss functions from configuration.

    Parameters
    ----------
    loss_config : dict
        Configuration dictionary with keys:
        - 'name': Loss function name or combination (e.g., 'dice+bce')
        - 'weights': List of weights for combined losses (if applicable)
        - 'params': Dictionary of parameters for loss functions

    Returns
    -------
    nn.Module
        Configured loss function ready for training.

    Raises
    ------
    ValueError
        If unknown loss name or mismatched weights for combined losses.

    Examples
    --------
    >>> config = {"name": "weighted_bce"}
    >>> loss_fn = get_loss_function(config)

    >>> config = {"name": "dice+bce", "weights": [0.5, 0.5]}
    >>> combined_loss = get_loss_function(config)
    """
    name = loss_config["name"].lower()
    from_logits = True  # All our model outputs are logits
    params = loss_config.get("params", {})

    # Extract label smoothing parameter
    label_smoothing = params.get("label_smoothing", 0.0)

    loss_lib = {
        "bce": nn.BCEWithLogitsLoss(),
        "weighted_bce": WeightedBCELoss(
            from_logits=from_logits, label_smoothing=label_smoothing
        ),
        "boundary": BoundaryLoss(from_logits=from_logits),
        "dice": SimpleDiceLoss(from_logits=from_logits),
        "weighted_huber_with_gradient": WeightedHuberWithGradientLoss(
            from_logits=from_logits
        ),
    }

    if "+" not in name:
        if name in loss_lib:
            return loss_lib[name]
        raise ValueError(f"Unknown loss function: {name}")

    # Handle combined losses
    loss_components_names = name.split("+")
    loss_weights = loss_config.get("weights")
    if not loss_weights or len(loss_components_names) != len(loss_weights):
        raise ValueError(
            f"Loss weights must be provided for combined loss '{name}' and match the number of components."
        )

    losses_to_combine = {}
    for comp_name, weight in zip(loss_components_names, loss_weights):
        if comp_name in loss_lib:
            losses_to_combine[comp_name] = (loss_lib[comp_name], weight)
        else:
            raise ValueError(f"Unknown component '{comp_name}' in combined loss.")

    return CombinedLoss(losses_to_combine, from_logits=from_logits)
