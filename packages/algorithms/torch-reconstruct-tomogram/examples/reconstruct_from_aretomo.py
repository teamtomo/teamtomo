"""Reconstruct a tomogram from an AreTomo alignment.

Loads the alignment from an AreTomo `.aln` file, reconstructs the tomogram and
writes it to an MRC file. Replace the paths and pixel spacing with your own.
"""

from pathlib import Path

import mrcfile
import torch
from torch_tilt_series import TiltSeries

from torch_reconstruct_tomogram import reconstruct_tomogram

# Paths to AreTomo alignment file and raw tilt stack
ALN_PATH = Path("/path/to/your/alignment.aln")
TILT_STACK_PATH = Path("/path/to/your/tilt_stack.mrc")

# Choose device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pixel spacing in Angstroms (required)
# AreTomo .aln files contain shifts in pixels, this converts them to Angstroms
PIXEL_SPACING = 6.192

# Load tilt series alignment.
tilt_series = TiltSeries.from_aretomo_output(
    aln_path=ALN_PATH,
    pixel_spacing=PIXEL_SPACING,
    image_path=TILT_STACK_PATH,
    device=DEVICE,
)

# Reconstruct tomogram
volume_shape = (512, 512, 512)
sidelength = 128
tomogram = reconstruct_tomogram(tilt_series, volume_shape, sidelength)

# Save as MRC file
output_path = ALN_PATH.parent / "torch_tomogram_reconstruction.mrc"
mrcfile.write(
    output_path,
    tomogram.cpu().numpy(),
    overwrite=True,
    voxel_size=PIXEL_SPACING,
)
