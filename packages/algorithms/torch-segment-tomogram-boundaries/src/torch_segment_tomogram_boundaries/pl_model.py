# src/torch_segment_tomogram_boundaries/pl_model.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from torch_segment_tomogram_boundaries import config


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        learning_rate: float = 1e-3,
        target_shape: tuple = None,
    ):
        super().__init__()
        # Ensure all important parameters are saved
        self.save_hyperparameters(ignore=["model", "loss_function"])
        self.model = model
        self.criterion = loss_function

    def setup(self, stage: str):
        pass

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage: str):
        image, label, weight_map = batch["image"], batch["label"], batch["weight_map"]
        target_for_loss = batch.get("soft_label", label)
        batch_size = image.size(0)
        model_output = self(image)
        # Handle both single output and tuple output (when using aux_params)
        if isinstance(model_output, tuple):
            pred_logits = model_output[0]  # Main segmentation output
        else:
            pred_logits = model_output
        effective_weight_map = weight_map
        if stage == "val" and not getattr(
            config, "USE_WEIGHT_MAP_FOR_VALIDATION", True
        ):
            effective_weight_map = None
        loss = self.criterion(pred_logits, target_for_loss, effective_weight_map)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Log the learning rate to the progress bar
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        return loss, pred_logits

    def training_step(self, batch, batch_idx):
        loss, pred_logits = self._common_step(batch, batch_idx, "train")
        if batch_idx == 0 and self.trainer.is_global_zero:
            pred_probs = torch.sigmoid(pred_logits)
            self._log_images(batch, pred_probs, "Train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred_logits = self._common_step(batch, batch_idx, "val")
        pred_probs = torch.sigmoid(pred_logits)
        pred_binary = (pred_probs > 0.5).float()
        dice = self.dice_coefficient(pred_binary, batch["label"])
        self.log(
            "val_dice",
            dice,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch["image"].size(0),
            sync_dist=True,
        )
        if batch_idx == 0 and self.trainer.is_global_zero:
            self._log_images(batch, pred_probs, "Validation")
        return loss

    def _log_images(self, batch: dict, pred_probs: torch.Tensor, stage_name: str):
        if self.logger is None or not hasattr(self.logger.experiment, "add_image"):
            return
        image, label = batch["image"], batch["label"]
        num_images_to_log = min(8, image.size(0))
        grid_params = {"padding": 2, "pad_value": 1.0, "nrow": 4}

        input_grid = torchvision.utils.make_grid(
            image[:num_images_to_log, 0:1, :, :], **grid_params, normalize=True
        )
        self.logger.experiment.add_image(
            f"{stage_name}/Input (Tomogram)", input_grid, self.current_epoch
        )

        label_grid = torchvision.utils.make_grid(
            label[:num_images_to_log].to(torch.float32), **grid_params
        )
        self.logger.experiment.add_image(
            f"{stage_name}/Ground Truth", label_grid, self.current_epoch
        )

        pred_prob_grid = torchvision.utils.make_grid(
            pred_probs[:num_images_to_log].to(torch.float32), **grid_params
        )
        self.logger.experiment.add_image(
            f"{stage_name}/Prediction Probabilities", pred_prob_grid, self.current_epoch
        )

    def dice_coefficient(self, pred, target, smooth=1e-5):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return torch.mean((2.0 * intersection + smooth) / (union + smooth))

    def configure_optimizers(self):
        optimizer_config = config.OPTIMIZER_CONFIG
        optimizer_name = optimizer_config["name"]
        optimizer_params = optimizer_config["params"]
        optimizer_params["lr"] = self.hparams.learning_rate

        optimizer = getattr(torch.optim, optimizer_name)(
            self.parameters(), **optimizer_params
        )

        if not config.USE_LR_SCHEDULER:
            return optimizer

        # New unified Warm-up and Decay Logic
        if hasattr(config, "WARMUP_EPOCHS") and config.WARMUP_EPOCHS > 0:
            warmup_epochs = config.WARMUP_EPOCHS
            max_epochs = config.MAX_EPOCHS

            # Main decay scheduler
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs - warmup_epochs,
                eta_min=config.SCHEDULER_CONFIG["params"].get("eta_min", 1e-7),
            )

            # Sequential scheduler
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1e-6,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    main_scheduler,
                ],
                milestones=[warmup_epochs],
            )

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }

        else:  # Fallback to original logic if no warm-up is configured
            scheduler_config = config.SCHEDULER_CONFIG
            scheduler_name = scheduler_config["name"]
            scheduler_params = scheduler_config["params"]

            # Special handling for CosineAnnealingLR T_max if not specified
            if (
                scheduler_name == "CosineAnnealingLR"
                and "T_max" not in scheduler_params
            ):
                scheduler_params["T_max"] = config.MAX_EPOCHS

            scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
                optimizer, **scheduler_params
            )

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }

            scheduler_monitor = scheduler_config.get("monitor")
            if scheduler_monitor:
                lr_scheduler_config["monitor"] = scheduler_monitor
            if scheduler_name == "ReduceLROnPlateau":
                lr_scheduler_config["reduce_on_plateau"] = True

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }
