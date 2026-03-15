"""Configuration parameters for tomographic boundary segmentation.

This module centralizes all configuration parameters including model architecture,
training hyperparameters, data paths, loss functions, and augmentation settings.
All parameters can be overridden programmatically for experiments.
"""

import os
from pathlib import Path

import torch

# --- DATA ---
BASE_DATA_PATH = Path(os.environ.get("TORCH_TOMO_SLAB_DATA", "data/"))

# Input data directories for script 01 and 02
TOMOGRAM_DIR: Path = BASE_DATA_PATH / "data_in" / "volumes"

# Output data directories from scripts
MASKS_DIR: Path = BASE_DATA_PATH / "data_in" / "boundary_mask_volumes"
PREPARED_DATA_BASE_DIR: Path = BASE_DATA_PATH / "prepared_data"
TRAIN_DATA_DIR: Path = PREPARED_DATA_BASE_DIR / "train"
VAL_DATA_DIR: Path = PREPARED_DATA_BASE_DIR / "val"

# --- TRAINING HYPERPARAMETERS ---
LEARNING_RATE: float = 1e-4
MAX_EPOCHS: int = 50
PRECISION: str = "32"
WARMUP_EPOCHS: int = 0

# --- MODEL ARCHITECTURE ---
MODEL_CONFIG: dict[str, any] = {
    "in_channels": 1,
    "out_channels": 1,
    "channels": (48, 96, 192, 384),
    "strides": (2, 2, 2),
    "num_res_units": 2,
    "dropout": 0.1,
}

# Control whether validation keeps per-pixel weighting; disabling aligns val_loss with
# unweighted BCE so the metric reflects global accuracy instead of boundary emphasis.
USE_WEIGHT_MAP_FOR_VALIDATION: bool = False

# --- Loss Function Configuration ---
# 'name': Can be a single loss or multiple losses combined with '+'.
# 'weights': A list of weights for combined losses. Must match the number of losses.
# 'params': A nested dictionary for loss-specific hyperparameters (now unused).
LOSS_CONFIG: dict[str, any] = {
    "name": "weighted_bce",  # Options: 'dice', 'bce', 'dice+bce', 'boundary', 'weighted_bce'.
    "weights": [0.5, 0.5],  # Only used for combined losses like 'dice+bce'
    "params": {
        "label_smoothing": 0.1,  # Re-enabled smoothing to soften boundary transitions
        "gradient_weight": 20,
    },
}

# Optional Gaussian smoothing applied to labels before computing losses.
USE_GAUSSIAN_LABEL_SMOOTHING: bool = True
GAUSSIAN_LABEL_SIGMA: float = 3
WEIGHT_MAP_BOUNDARY_WIDTH: int = 4

# --- DATALOADER & AUGMENTATION ---
BATCH_SIZE: int = 8
NUM_WORKERS: int = 16

# --- EARLY STOPPING ---
STANDARD_EARLY_STOPPING_PATIENCE: int = 10
EARLY_STOP_MIN_DELTA: float = 1e-3

# --- PL TRAINER & INFRASTRUCTURE ---
ACCELERATOR: str = "auto"
# Force single GPU for Optuna trials to avoid distributed training issues
DEVICES: int = (
    1
    if os.environ.get("OPTUNA_TRIAL")
    else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
)

# --- CHECKPOINTING ---
CKPT_SAVE_PATH: Path = BASE_DATA_PATH / "checkpoints"
CHECKPOINT_SAVE_TOP_K: int = 1

# --- OPTIMIZER ---
OPTIMIZER_CONFIG: dict[str, any] = {
    "name": "AdamW",
    "params": {
        "lr": LEARNING_RATE,
        "weight_decay": 1e-1,  # Strong L2 regularization
        "eps": 1e-7,  # More stable epsilon for AdamW
    },
}

# --- SCHEDULER ---
USE_LR_SCHEDULER: bool = True
SCHEDULER_CONFIG: dict[str, any] = {
    "name": "ReduceLROnPlateau",
    "params": {
        "factor": 0.5,
        "patience": 4,
        "mode": "max",
        "threshold": 1e-3,
        "threshold_mode": "rel",
        "cooldown": 0,
        "min_lr": 1e-7,
    },
    "monitor": "val_dice",
}
