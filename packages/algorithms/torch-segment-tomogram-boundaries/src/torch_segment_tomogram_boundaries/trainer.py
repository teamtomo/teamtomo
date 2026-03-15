"""PyTorch Lightning training pipeline for tomographic boundary segmentation.

This module provides the main training interface that orchestrates data loading,
model setup, callback configuration, and training execution using PyTorch Lightning.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch

# Note: keep callback imports grouped for clarity
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from torch_segment_tomogram_boundaries import config, constants
from torch_segment_tomogram_boundaries.data.dataloading import SegmentationDataModule
from torch_segment_tomogram_boundaries.losses import get_loss_function
from torch_segment_tomogram_boundaries.models import create_unet
from torch_segment_tomogram_boundaries.pl_model import SegmentationModel
from torch_segment_tomogram_boundaries.utils.common import save_config_snapshot


class TomoSlabTrainer:
    """
    Complete training pipeline for tomographic boundary segmentation models.

    This class provides a high-level interface for training deep learning models
    on tomographic data. It encapsulates the entire training workflow including
    data preparation, model configuration, callback setup, and training execution
    using PyTorch Lightning.

    The trainer supports:
    - Configurable MONAI-based model architectures
    - Multiple loss functions with automatic selection
    - Standard early stopping and checkpointing callbacks
    - Multi-GPU training and mixed precision
    - Comprehensive logging and checkpointing

    Attributes
    ----------
    model_arch : str
        Architecture name (e.g., 'Unet', 'UnetPlusPlus').
    model_encoder : str
        Encoder backbone name (e.g., 'resnet18', 'efficientnet-b0').
    encoder_weights : str or None
        Pre-trained weights ('imagenet', None).
    encoder_depth : int
        Encoder depth (typically 5 for ResNet).
    decoder_channels : List[int]
        Decoder channel configuration.
    decoder_attention_type : str
        Attention mechanism type ('scse', 'cbam', None).
    classes : int
        Number of output classes (1 for binary segmentation).
    in_channels : int
        Number of input channels (currently 1 for single normalized intensity).
    loss_config : Dict[str, Any]
        Loss function configuration.
    learning_rate : float
        Initial learning rate.
    global_rank : int
        Process rank for distributed training.
    """

    def __init__(
        self,
        loss_config: Dict[str, Any] = config.LOSS_CONFIG,
        learning_rate: float = config.LEARNING_RATE,
        train_data_dir: Path = config.TRAIN_DATA_DIR,
        val_data_dir: Path = config.VAL_DATA_DIR,
        ckpt_save_dir: Path = config.CKPT_SAVE_PATH,
    ) -> None:
        """
        Initialize the TomoSlabTrainer with model and training configurations.

        Parameters
        ----------
        model_arch : str
            Architecture name for the segmentation model (e.g., 'Unet', 'UnetPlusPlus').
        model_encoder : str
            Encoder backbone name (e.g., 'resnet18', 'resnet34', 'efficientnet-b0').
        encoder_weights : str or None
            Pre-trained weights for encoder ('imagenet', None).
        encoder_depth : int
            Depth of the encoder (typically 5 for ResNet architectures).
        decoder_channels : List[int]
            Number of channels in each decoder block, from deepest to shallowest.
        decoder_attention_type : str
            Type of attention mechanism in decoder ('scse', 'cbam', None).
        classes : int
            Number of output classes (1 for binary segmentation).
        in_channels : int
            Number of input channels (1 for normalized intensity images).
        loss_config : Dict[str, Any]
            Loss function configuration with 'name' and optional 'weights'.
        learning_rate : float
            Initial learning rate for the optimizer.
        train_data_dir : Path
            Directory containing training data files.
        val_data_dir : Path
            Directory containing validation data files.
        """
        self.loss_config = loss_config
        self.learning_rate = learning_rate
        self.train_data_dir = Path(train_data_dir)
        self.val_data_dir = Path(val_data_dir)
        self.ckpt_save_dir = Path(ckpt_save_dir)
        self.global_rank = int(os.environ.get("GLOBAL_RANK", 0))

    def _setup_datamodule(self) -> SegmentationDataModule:
        """
        Set up the PyTorch Lightning DataModule for training and validation.

        Returns
        -------
        SegmentationDataModule
            Configured data module with train/val datasets.

        Raises
        ------
        FileNotFoundError
            If training or validation data files are not found.
        """
        train_files = sorted(list(self.train_data_dir.glob("*.pt")))
        val_files = sorted(list(self.val_data_dir.glob("*.pt")))
        if not train_files or not val_files:
            raise FileNotFoundError(
                "Training or validation data not found. Run the TrainingDataGenerator first."
            )
        return SegmentationDataModule(
            train_pt_files=train_files,
            val_pt_files=val_files,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
        )

    def _setup_model(self) -> SegmentationModel:
        """
        Create and configure the PyTorch Lightning model wrapper.

        Returns
        -------
        SegmentationModel
            Configured Lightning module with model, loss function, and hyperparameters.
        """
        base_model = create_unet(**config.MODEL_CONFIG)
        loss_fn = get_loss_function(self.loss_config)
        if self.global_rank == 0:
            loss_name = getattr(loss_fn, "name", loss_fn.__class__.__name__)
            print(f"Using Loss Function: {loss_name}")
            print(
                "Using enhanced augmentations without cropping for consistent train/val pipeline"
            )
        return SegmentationModel(
            model=base_model,
            loss_function=loss_fn,
            learning_rate=self.learning_rate,
            target_shape=constants.TARGET_VOLUME_SHAPE,
        )

    def _setup_callbacks(self) -> List[pl.Callback]:
        """
        Configure PyTorch Lightning callbacks for training.

        Returns
        -------
        List[pl.Callback]
            List of configured callbacks including checkpointing, early stopping,
            and learning rate monitoring.
        """
        checkpointer = ModelCheckpoint(
            monitor=constants.MONITOR_METRIC,
            mode=constants.MONITOR_MODE,
            filename=f"best-{{epoch}}-{{{constants.MONITOR_METRIC}:.4f}}",
            save_top_k=config.CHECKPOINT_SAVE_TOP_K,
            verbose=(self.global_rank == 0),
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        progress_bar = TQDMProgressBar(refresh_rate=10)
        callbacks = [progress_bar, checkpointer, lr_monitor]

        if self.global_rank == 0:
            print("Using standard EarlyStopping callback.")
        callbacks.append(
            EarlyStopping(
                monitor=constants.MONITOR_METRIC,
                mode=constants.MONITOR_MODE,
                patience=config.STANDARD_EARLY_STOPPING_PATIENCE,
                min_delta=config.EARLY_STOP_MIN_DELTA,
                verbose=(self.global_rank == 0),
            )
        )
        return callbacks

    def fit(
        self,
        extra_callbacks: Optional[List[pl.Callback]] = None,
        extra_trainer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Execute the complete training pipeline.

        This method orchestrates the entire training process including:
        - Data module setup and validation
        - Model instantiation with loss function
        - Callback and logger configuration
        - PyTorch Lightning trainer setup
        - Model training execution

        The trained model checkpoints will be saved to 'lightning_logs/' directory.
        """
        if self.global_rank == 0:
            print("--- Running on main process (rank 0). Verbose output enabled. ---")
        torch.set_float32_matmul_precision("high")

        datamodule = self._setup_datamodule()
        pl_model = self._setup_model()

        if self.global_rank == 0:
            print("--- Configuring Logger and Callbacks ---")
        experiment_name = "unet-monai"
        experiment_details = f"loss-{self.loss_config['name'].replace('+', '_')}"
        logger = TensorBoardLogger(
            save_dir=self.ckpt_save_dir, name=f"{experiment_name}-{experiment_details}"
        )

        if self.global_rank == 0:
            snapshot_path = save_config_snapshot(Path(logger.log_dir))
            print(f"Config snapshot saved to: {snapshot_path}")

        callbacks = self._setup_callbacks()
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        use_ddp = torch.cuda.is_available() and config.DEVICES and config.DEVICES != 1

        trainer_config = {
            "max_epochs": config.MAX_EPOCHS,
            "accelerator": config.ACCELERATOR,
            "devices": config.DEVICES,
            "precision": config.PRECISION,
            "log_every_n_steps": constants.LOG_EVERY_N_STEPS,
            "check_val_every_n_epoch": constants.CHECK_VAL_EVERY_N_EPOCH,
            "logger": logger,
            "callbacks": callbacks,
            "strategy": "ddp_find_unused_parameters_false" if use_ddp else "auto",
            "gradient_clip_val": 0.5,
            "gradient_clip_algorithm": "norm",
            "enable_model_summary": True,
        }
        if extra_trainer_kwargs:
            trainer_config.update(extra_trainer_kwargs)

        self.trainer = pl.Trainer(**trainer_config)

        if self.global_rank == 0:
            print("--- Starting Training ---")
            print(f"To view logs, run: tensorboard --logdir={logger.save_dir}")

        self.trainer.fit(pl_model, datamodule=datamodule)

        if self.trainer.is_global_zero:
            print("--- Training Finished ---")
            if (
                hasattr(self.trainer.checkpoint_callback, "best_model_path")
                and self.trainer.checkpoint_callback is not None
            ):
                print(
                    f"Best model checkpoint saved at: {self.trainer.checkpoint_callback.best_model_path}"
                )


def train(
    train_data_dir: Path,
    val_data_dir: Path,
    ckpt_save_dir: Path = config.CKPT_SAVE_PATH,
    learning_rate: float = config.LEARNING_RATE,
    max_epochs: int = config.MAX_EPOCHS,
    **trainer_kwargs,
):
    """
    High-level function to train a tomogram segmentation model.

    This function provides a simple interface for training, handling the setup
    of the model, data, and trainer based on the provided parameters and
    project configurations.

    Parameters
    ----------
    train_data_dir : Path
        Directory containing training data files (*.pt).
    val_data_dir : Path
        Directory containing validation data files (*.pt).
    ckpt_save_dir : Path, optional
        Directory to save model checkpoints and logs, by default config.CKPT_SAVE_PATH.
    learning_rate : float, optional
        Initial learning rate for the optimizer, by default config.LEARNING_RATE.
    max_epochs : int, optional
        Maximum number of training epochs, by default config.MAX_EPOCHS.
    **trainer_kwargs :
        Additional keyword arguments to be passed to the PyTorch Lightning Trainer.
        This allows for overriding settings like 'accelerator', 'devices', etc.
    """
    pl.seed_everything(42, workers=True)

    trainer = TomoSlabTrainer(
        learning_rate=learning_rate,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        ckpt_save_dir=ckpt_save_dir,
    )

    # Combine max_epochs with other trainer arguments
    all_trainer_kwargs = {"max_epochs": max_epochs, **trainer_kwargs}
    trainer.fit(extra_trainer_kwargs=all_trainer_kwargs)
