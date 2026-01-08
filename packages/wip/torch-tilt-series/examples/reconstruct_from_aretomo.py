import alnfile
import mrcfile
import numpy as np
from pathlib import Path
from torch_tilt_series import Tomogram

# Paths
ALN_PATH = Path('/Path/to/your/alnfile.aln')
TILT_STACK_PATH = Path('/Path/to/your/raw/tilt_stack.mrc')

# Read AreTomo alignment data
df = alnfile.read(ALN_PATH)

# Get shifts for reconstruction 
# In aretomo, tx and ty already represent the shifts in the forward projection model (sample -> image) 
corrected_shifts_t_xy = df[['tx', 'ty']].to_numpy()
corrected_shifts_t_yx = corrected_shifts_t_xy[:, ::-1].copy() #yx

# Load tilt stack
tilt_stack_full = mrcfile.read(TILT_STACK_PATH)

included_indices = df['sec'].values - 1  # 0-indexed section indices
tilt_stack = tilt_stack_full[included_indices]

# Normalize
tilt_stack -= np.mean(tilt_stack, axis=(-2, -1), keepdims=True)
tilt_stack /= np.std(tilt_stack, axis=(-2, -1), keepdims=True) 

# Build tomogram and reconstruct
tilt_series = Tomogram(
    images=tilt_stack,
    tilt_angles=df['tilt'].to_numpy(),
    tilt_axis_angle=df['rot'].to_numpy(), 
    sample_translations=corrected_shifts_xf.copy()
) 

tomogram = tilt_series.reconstruct_tomogram((100, 480, 320), 128)

# Save tomogram
out_path = ALN_PATH.parent / 'tt_rec.mrc'
mrcfile.write(out_path, tomogram.numpy(), overwrite=True, voxel_size=10)
