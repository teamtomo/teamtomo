import etomofiles
import mrcfile
import numpy as np
from pathlib import Path
from torch_tomogram import Tomogram


# Read etomo alignment data
ETOMO_DIR = Path("/path/to/etomo/dir")

df = etomofiles.read(ETOMO_DIR)
# Filter out excluded tilts
df = df.loc[~df['excluded']].reset_index(drop=True)

# Get IMOD xf components from dataframe
# df_to_xf(df, yx=True) returns (n_tilts, 2, 3) array
# Each matrix is [[A22, A21, DY], [A12, A11, DX]] (ready for torch-tomogram yz) 
xf = etomofiles.df_to_xf(df, yx=True)
m, shifts = xf[:, :, :2], xf[:, :, 2]
# Convert IMOD's backward projection model to torch-tomogram's forward model
# IMOD: image -> sample
#   > the 2d matrix from the .xf file represents a 2d transform to align
#   > the image with the tilt-axis
# torch-tomogram: sample -> image
#   > the shifts are applied after rotation and projection and shift the
#   > projected sample to the image position
#
#  Roation matrix are orthogonal, so inversion = transposition :
#  np.einsum('nij,nj->ni', np.linalg.inv(m), shifts) = np.einsum('nji,nj->ni', m, shifts) 
#
#  Negate shifts for forward projection model
corrected_shifts = -np.einsum('nji,nj->ni', m, shifts)
corrected_shifts = np.ascontiguousarray(corrected_shifts)

# Load and normalize tilt stack
tilt_stack_path = ETOMO_DIR / df.image_path[0].replace('[0]', '')
tilt_stack_full = mrcfile.read(tilt_stack_path)
# Filter stack to only included tilts
tilt_stack = tilt_stack_full[df.idx_tilt.to_numpy()]
#Zero-mean, unit-variance normalization per image
tilt_stack -= np.mean(tilt_stack, axis=(-2, -1), keepdims=True)
tilt_stack /= np.std(tilt_stack, axis=(-2, -1), keepdims=True)

# Create tomogram object and reconstruct
tilt_series = Tomogram(
    images=tilt_stack,
    tilt_angles=df.tlt,
    tilt_axis_angle=df.tilt_axis_angle,
    sample_translations=corrected_shifts.copy()
)
tomogram = tilt_series.reconstruct_tomogram((100, 480, 380), 128)

mrcfile.write(ETOMO_DIR / 'tt_rec.mrc', tomogram.numpy(), overwrite=True, voxel_size=10)
