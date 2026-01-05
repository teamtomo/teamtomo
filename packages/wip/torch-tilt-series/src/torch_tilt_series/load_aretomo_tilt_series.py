from pathlib import Path

import alnfile
import mrcfile
import numpy as np
import pandas as pd
import torch

from torch_tilt_series import TiltSeries

def load_aretomo_tilt_series(
    aln_file: str | Path,
    pixel_spacing: float,
    device: torch.device | str = "cpu",
) -> "TiltSeries":
    """Initialize TiltSeries from AreTomo dataframe."""
    # load aln metadata
    df = alnfile.read(aln_file)

    # Extract XY shifts and convert to angstroms
    aln_shifts_yx = df[["ty", "tx"]].to_numpy()
    aln_shifts_yx_ang = aln_shifts_yx * pixel_spacing

    # Load tilt stack as float32
    tilt_series_path = df["image_path"].iloc[0]
    tilt_stack_full = mrcfile.read(tilt_series_path)
    tilt_stack_full = tilt_stack_full.astype(np.float32)

    # Subset to valid tilts
    idx_valid = df["sec"].values - 1  # Convert from 1-indexed to 0-indexed
    tilt_stack = tilt_stack_full[idx_valid]

    # Normalize on central 25% crop to avoid edge artifacts
    h, w = tilt_stack.shape[-2:]
    h_crop = slice(int(0.375 * h), int(0.625 * h))
    w_crop = slice(int(0.375 * w), int(0.625 * w))
    crop = tilt_stack[:, h_crop, w_crop]
    tilt_stack -= np.mean(crop, axis=(-2, -1), keepdims=True)
    tilt_stack /= np.std(crop, axis=(-2, -1), keepdims=True)

    return TiltSeries(
        images=tilt_stack,
        tilt_angles=df["tilt"].to_numpy(),
        tilt_axis_angle=df["rot"].to_numpy(),
        sample_translations=aln_shifts_yx_ang,
        pixel_spacing=pixel_spacing,
        device=device,
    )