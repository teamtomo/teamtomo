"""Project constants and file path configurations.

This module defines file paths, data processing constants, and other fixed
parameters used throughout the tomographic segmentation pipeline.
"""

VALIDATION_FRACTION: float = 0.2
# --- Data Processing Constants ---
# Final crop size for training transforms
FINAL_CROP_SIZE = 512

# Resize volume-labels to these dims
TARGET_VOLUME_SHAPE = (256, 512, 512)

# Number of 2D slices to extract from each 3D volume in script 02
NUM_SECTIONS_PER_VOLUME = 150

MONITOR_METRIC: str = "val_dice"
MONITOR_MODE: str = "max"
LOG_EVERY_N_STEPS: int = 10
CHECK_VAL_EVERY_N_EPOCH: int = 1

# --- AUGMENTATION PARAMETERS ---
AUGMENTATION_CONFIG: dict[str, any] = {
    # Dimensional consistency parameters (NEW - for training stability)
    "RESIZE_BUFFER_HEIGHT": 288,  # 12.5% larger than target (256 * 1.125)
    "RESIZE_BUFFER_WIDTH": 576,  # 12.5% larger than target (512 * 1.125)
    "TARGET_HEIGHT": 256,  # Final crop height (guarantees consistent output)
    "TARGET_WIDTH": 512,  # Final crop width (guarantees consistent output)
    # Geometric augmentation parameters (UPDATED - more conservative)
    "ROTATE_LIMIT": 20,  # Reduced from 20 to 8 degrees for stability
    "AFFINE_SCALE_RANGE": (0.95, 1.05),  # Conservative scaling (was aggressive 0.8-1.2)
    # Intensity augmentation parameters (UPDATED - more conservative)
    "BRIGHTNESS_CONTRAST_LIMIT": 0.2,
    "GAMMA_LIMIT": (50, 70),
    "NOISE_VAR_LIMIT": (5, 15),  # Reduced from (5, 10) for cleaner augmentations
    "BLUR_LIMIT": 3,  # Gaussian blur kernel limit (unchanged)
    # Dropout/occlusion parameters (UNCHANGED - proven effective)
    "COARSE_DROPOUT_HOLES": (3, 8),  # Number of holes range
    "COARSE_DROPOUT_SIZE": (0.03, 0.08),  # Hole size range
    "GRID_DROPOUT_RATIO": 0.1,  # Grid dropout coverage
}
