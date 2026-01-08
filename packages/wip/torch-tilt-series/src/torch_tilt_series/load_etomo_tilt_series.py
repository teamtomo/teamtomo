from pathlib import Path

import etomofiles
import mrcfile
import numpy as np
import torch

from .tilt_series import TiltSeries

def load_imod_tilt_series(
    etomo_directory: str | Path,
    pixel_spacing: float,
    device: torch.device | str = "cpu",
) -> "TiltSeries":
    """Initialize TiltSeries from ETOMO directory."""
    df = etomofiles.read(etomo_directory)

    # Filter out excluded tilts
    df = df.loc[~df['excluded']].reset_index(drop=True)
    df = df.loc[~df["excluded"]].reset_index(drop=True)

    # Get shifts from IMOD xf metadata
    # c.f. https://bio3d.colorado.edu/imod/doc/man/xfmodel.html
    # > Each linear transformation in a transform file is specified by a line
    #   with six numbers:
    #   A11 A12 A21 A22 DX DY where the coordinate (X, Y) is transformed to
    #   (X', Y') by:
    #    X' = A11 * X + A12 * Y + DX
    #    Y' = A21 * X + A22 * Y + DY
    #
    # Using etomofiles.df_to_xf(df, yx=True)
    # we get an (n_tilts, 2, 3) numpy array for transforming yx coords
    # Each matrix is [[A22, A21, DY],
    #                 [A12, A11, DX]]
    xf = etomofiles.df_to_xf(df, yx=True)
    m, xf_shifts = xf[:, :, :2], xf[:, :, 2]

    # Convert IMOD's backward projection model to forward model
    # Rotation matrices are orthogonal, so inversion = transposition
    # Negate shifts for forward projection model
    corrected_shifts = -1 * np.einsum("nji,nj->ni", m, xf_shifts)
    corrected_shifts = np.ascontiguousarray(corrected_shifts)

    # Convert shifts from pixels to Angstroms
    corrected_shifts_ang = corrected_shifts * pixel_spacing

    # Load tilt stack
    tilt_stack_path = df["image_path"][0].replace("[0]", "")
    tilt_series = mrcfile.read(tilt_stack_path)

    # Subset to only valid tilts
    idx_valid = df["idx_tilt"].to_numpy()
    tilt_series = tilt_series[idx_valid]
    tilt_series = tilt_series.astype(np.float32)

    # Normalize tilt series on central 25% crop to avoid edge artifacts
    h, w = tilt_series.shape[-2:]
    hs = slice(int(0.375 * h), int(0.625 * h))
    ws = slice(int(0.375 * w), int(0.625 * w))
    crop = tilt_series[:, hs, ws]
    tilt_series -= np.mean(crop, axis=(-2, -1), keepdims=True)
    tilt_series /= np.std(crop, axis=(-2, -1), keepdims=True)

    return TiltSeries(
        tilt_images=tilt_series,
        stage_tilt_angles=df["tlt"].to_numpy(),
        tilt_axis_angle=df["tilt_axis_angle"].to_numpy(),
        image_shifts_angstroms=corrected_shifts_ang,
        pixel_spacing=pixel_spacing,
        device=device,
    )