import warnings
from typing import Tuple, Optional, Callable, Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from tiler import Tiler, Merger
from torch import Tensor, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .dice import dice_loss, dice_score
from .model_parts import Conv3x3, Down, Up
from ..utils import normalise_2d
from ..constants import TRAINING_IMAGE_DIMENSIONS


class ResidualUNet18(L.LightningModule):
    """U-Net with ResNet18 style encoder."""

    in_channels: int = 1
    num_classes: int = 2

    def __init__(self, batch_size: int = 4, learning_rate: float = 1e-05):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.validation_dice_score = 0
        self.validation_step_outputs = []

        self.base_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.down1 = Down(32, 64, n_residual_blocks=2, stride=2)
        self.down2 = Down(64, 128, n_residual_blocks=2, stride=2)
        self.down3 = Down(128, 256, n_residual_blocks=2, stride=2)
        self.down4 = Down(256, 512, n_residual_blocks=2, stride=2)
        self.flat_conv = nn.Sequential(
            Conv3x3(512, 512, stride=1), nn.ReLU(inplace=True)
        )
        self.up4 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up1 = Up(64, 32)
        self.out_conv = nn.Conv2d(32, self.num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image: Tensor) -> Tensor:
        x1 = self.base_layer(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        out = self.down4(x4)
        out = self.flat_conv(out)
        out = self.up4(out, x4)
        out = self.up3(out, x3)
        out = self.up2(out, x2)
        out = self.up1(out, x1)
        out = self.out_conv(out)
        return out

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        images, labels = batch
        logits = self(images)  # (b, c, h, w)
        cross_entropy = F.cross_entropy(logits, labels)
        dice = dice_loss(ground_truth=labels, logits=logits)
        loss = cross_entropy + dice
        self.log(name="training loss", value=loss)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        images, labels = batch
        logits = self(images)  # (b, c, h, w)
        dice = dice_score(ground_truth=labels, logits=logits)
        self.log("validation dice score", dice)
        self.validation_step_outputs.append(dice)
        return dice

    def predict_step(
        self,
        image: torch.Tensor,  # (h, w)
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """Tiled prediction."""
        tiler = Tiler(
            data_shape=image.shape,
            tile_shape=TRAINING_IMAGE_DIMENSIONS,
            overlap=0.35,
            mode="reflect",
        )
        merger = Merger(tiler)
        image = image.cpu().numpy()
        for idx, tiles in tiler.iterate(image, batch_size=self.batch_size):
            tiles = torch.as_tensor(tiles, dtype=torch.float, device=self.device)
            tiles = normalise_2d(tiles)
            tiles = rearrange(tiles, "b h w -> b 1 h w")
            prediction = self(tiles)
            probabilities = F.softmax(prediction, dim=1)[:, 1, ...]
            probabilities = probabilities.detach().cpu().numpy()
            merger.add_batch(
                batch_id=idx, batch_size=self.batch_size, data=probabilities
            )
        return torch.tensor(merger.merge(unpad=True))

    def on_validation_epoch_end(self):
        mean_dice_score = torch.mean(torch.as_tensor(self.validation_step_outputs))
        self.validation_dice_score = mean_dice_score
        self.log(name="validation dice score", value=mean_dice_score)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.RMSprop(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-8,
            momentum=0.9,
        )
        self.scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2)
        return optimizer

    def optimizer_step(
        self,
        epoch_idx: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Optional[Callable[[], Any]] = None,
        **kwargs,
    ) -> None:
        self.log("learning rate", optimizer.param_groups[0]["lr"])
        if batch_idx == 0:  # call the scheduler after each validation
            self.scheduler.step(self.validation_dice_score)
            print(
                "\n"
                f"validation dice: {self.validation_dice_score}, "
                f"best: {self.scheduler.best}, "
                f"num_bad_epochs: {self.scheduler.num_bad_epochs}"
                "\n"
            )  # for debugging
        super().optimizer_step(
            epoch_idx, batch_idx, optimizer, optimizer_closure, **kwargs
        )
