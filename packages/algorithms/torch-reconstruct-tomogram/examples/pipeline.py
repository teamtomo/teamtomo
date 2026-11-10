"""From a tilt series to subvolumes and a tomogram.

Walks through the whole package: load an alignment, project points, extract
subtilt-series, reconstruct subvolumes and the full tomogram (also at a chosen
output pixel size), and save the result. Replace the paths and pixel spacing with
your own.
"""

from pathlib import Path

import mrcfile
import torch
from torch_tilt_series import TiltSeries

from torch_reconstruct_tomogram import (
    extract_particle_tilt_series,
    reconstruct_subvolume,
    reconstruct_tomogram,
)

# Choose device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pixel spacing in Angstroms (required)
PIXEL_SPACING = 1.548

# 1. Load a tilt series
# From AreTomo output (.aln file + tilt stack):
ALN_PATH = Path("/path/to/your/alignment.aln")
TILT_STACK_PATH = Path("/path/to/your/tilt_stack.mrc")
tilt_series = TiltSeries.from_aretomo_output(
    aln_path=ALN_PATH,
    pixel_spacing=PIXEL_SPACING,
    image_path=TILT_STACK_PATH,
    device=DEVICE,
)

# Or from an ETOMO directory:
# tilt_series = TiltSeries.from_etomo_directory(
#     etomo_dir=Path("/path/to/etomo/dir"),
#     pixel_spacing=PIXEL_SPACING,
#     device=DEVICE,
# )

print("projection_matrices", tuple(tilt_series.projection_matrices.shape))

# Move between devices if needed
tilt_series.to(DEVICE)

# 2. Project 3D points into the tilt images
# Points are zyx, in Angstroms, relative to the tomogram center. Projected
# 2D points are also in Angstroms, relative to the detector center.
points_zyx = torch.tensor([[0.0, 0.0, 0.0], [100.0, -50.0, 200.0]], device=DEVICE)
projected_yx = tilt_series.project_points(points_zyx)  # (n_points, n_tilts, 2)

# 3. Extract a subtilt-series around each point
# Loads + normalizes images from tilt_series.image_path internally.
particle_tilt_series = extract_particle_tilt_series(
    tilt_series, points_zyx, sidelength=64, return_rfft=False
)  # (n_points, n_tilts, 64, 64)

# 4. Reconstruct subvolumes at the points
subvolumes = reconstruct_subvolume(
    tilt_series, points_zyx, sidelength=64
)  # (n_points, 64, 64, 64)

# 5. Reconstruct the full tomogram
volume_shape = (256, 512, 512)
sidelength = 128
tomogram = reconstruct_tomogram(tilt_series, volume_shape, sidelength, batch_size=None)

# Or reconstruct at desired pixel size: output_pixel_spacing targets an arbitrary
# output voxel size independent of the raw data's, at both the local
# (reconstruct_subvolume) and global (reconstruct_tomogram) level.

OUTPUT_PIXEL_SPACING = 10.0
binned_shape = tuple(
    round(s * PIXEL_SPACING / OUTPUT_PIXEL_SPACING) for s in volume_shape
)
tomogram_binned = reconstruct_tomogram(
    tilt_series,
    volume_shape=binned_shape,
    sidelength=sidelength,
    output_pixel_spacing=OUTPUT_PIXEL_SPACING,
    batch_size=None,
)

# 6. Save the result
output_path = ALN_PATH.parent / "torch_tomogram_reconstruction.mrc"
mrcfile.write(
    output_path,
    tomogram.cpu().numpy(),
    overwrite=True,
    voxel_size=PIXEL_SPACING,
)
