"""Reconstruct a tomogram from an IMOD (ETOMO) alignment.

Loads the alignment from an ETOMO directory, reconstructs the tomogram and writes
it to an MRC file. Replace the paths and pixel spacing with your own.
"""

from pathlib import Path

import mrcfile
import torch
from torch_tilt_series import TiltSeries

from torch_reconstruct_tomogram import reconstruct_tomogram

# Path to ETOMO project directory
ETOMO_DIR = Path("/path/to/etomo/dir")

# Choose device: "cpu" or "cuda"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pixel spacing in Angstroms (required)
PIXEL_SPACING = 6.192

# Load tilt series alignment.
tilt_series = TiltSeries.from_etomo_directory(
    etomo_dir=ETOMO_DIR,
    pixel_spacing=PIXEL_SPACING,
    device=DEVICE,
)

# Reconstruct tomogram
volume_shape = (512, 512, 512)
sidelength = 128
tomogram = reconstruct_tomogram(tilt_series, volume_shape, sidelength)

# Save as MRC file
output_path = ETOMO_DIR / "torch_tomogram_reconstruction.mrc"
mrcfile.write(
    output_path,
    tomogram.cpu().numpy(),
    overwrite=True,
    voxel_size=PIXEL_SPACING,
)
