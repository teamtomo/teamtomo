"""Load a tilt series from AreTomo output.

Reads the alignment from an AreTomo `.aln` file and projects a 3D point into every
tilt image. Needs the optional IO dependencies: `pip install torch-tilt-series[io]`.
Replace the paths and pixel spacing with your own.
"""

from pathlib import Path

import torch

from torch_tilt_series import TiltSeries

# Paths to AreTomo alignment file and raw tilt stack
ALN_PATH = Path("/path/to/your/alignment.aln")
TILT_STACK_PATH = Path("/path/to/your/tilt_stack.mrc")

# Choose device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pixel spacing in Angstroms (required)
# AreTomo .aln files contain shifts in pixels, this converts them to Angstroms
PIXEL_SPACING = 6.192

# Load tilt series from AreTomo output
tilt_series = TiltSeries.from_aretomo_output(
    aln_path=ALN_PATH,
    pixel_spacing=PIXEL_SPACING,
    image_path=TILT_STACK_PATH,
    device=DEVICE,
)

# Project a 3D point (zyx, Angstroms, relative to tomogram center) into each
# tilt -> 2D detector positions (yx, Angstroms, relative to detector center)
points_zyx = torch.tensor([[0.0, 0.0, 0.0]], device=DEVICE)
projected_yx = tilt_series.project_points(points_zyx)

# tilt_series.image_path / tilt_series.image_indices describe where the
# matching raw tilt images live and how to select/order them; TiltSeries
# itself never loads image data. See torch-reconstruct-tomogram for loading,
# normalizing, and extracting subtilt-series / reconstructing subvolumes.
