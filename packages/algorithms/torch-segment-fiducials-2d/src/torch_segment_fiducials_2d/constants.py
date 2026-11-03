"""Constants describing the training data of the pretrained model."""

import numpy as np

TRAINING_IMAGE_DIMENSIONS = (512, 512)
TRAINING_PIXEL_SIZE = 8  # Å/px
PIXELS_PER_FIDUCIAL = (np.pi * 50**2) / (TRAINING_PIXEL_SIZE**2)
