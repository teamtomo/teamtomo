"""Data I/O for deformation field operations."""

from pathlib import Path
from typing import Union

import pandas as pd
import torch


def write_deformation_field_to_csv(
    deformation_field: torch.Tensor, output_path: Union[str, Path]
) -> None:
    """
    Write deformation field to CSV format.

    Parameters
    ----------
    deformation_field: torch.Tensor
        Deformation field with shape (2, t, h, w)
        Channel 0 is y_shift, channel 1 is x_shift
    output_path: Union[str, Path]
        Path to output CSV file

    Returns
    -------
    None
    """
    # Get dimensions
    _, t, h, w = deformation_field.shape

    # Prepare data lists
    time_indices = []
    height_indices = []
    width_indices = []
    y_shifts = []
    x_shifts = []

    # Iterate through all time points and spatial positions
    for t_idx in range(t):
        for h_idx in range(h):
            for w_idx in range(w):
                # Get shifts
                y_shift = deformation_field[0, t_idx, h_idx, w_idx].item()
                x_shift = deformation_field[1, t_idx, h_idx, w_idx].item()

                # Append to lists
                time_indices.append(t_idx)
                height_indices.append(h_idx)
                width_indices.append(w_idx)
                y_shifts.append(y_shift)
                x_shifts.append(x_shift)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "t": time_indices,
            "h": height_indices,
            "w": width_indices,
            "y_shift": y_shifts,
            "x_shift": x_shifts,
        }
    )

    # Write to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Deformation field written to {output_path}")
    print(
        f"CSV contains {len(df)} rows with {t} time points and "
        f"{h * w} spatial positions per time point"
    )


def read_deformation_field_from_csv(
    csv_path: Union[str, Path], device: torch.device = None
) -> torch.Tensor:
    """
    Read deformation field from CSV format.

    Parameters
    ----------
    csv_path: Union[str, Path]
        Path to input CSV file
    device: torch.device, optional
        Device for output tensor

    Returns
    -------
    deformation_field: torch.Tensor
        Deformation field with shape (2, t, h, w)
    """
    if device is None:
        device = torch.device("cpu")

    # Read CSV
    df = pd.read_csv(csv_path)

    print(f"Reading deformation field from {csv_path}")
    print(f"CSV contains {len(df)} rows")

    # Get unique indices to determine dimensions
    unique_t = sorted(df["t"].unique())
    unique_h = sorted(df["h"].unique())
    unique_w = sorted(df["w"].unique())

    t = len(unique_t)
    h = len(unique_h)
    w = len(unique_w)

    print(f"Detected dimensions: t={t}, h={h}, w={w}")

    # Initialize tensor
    deformation_field = torch.zeros((2, t, h, w), device=device, dtype=torch.float32)

    # Create mapping from indices to tensor positions
    t_to_idx = {t_val: idx for idx, t_val in enumerate(unique_t)}
    h_to_idx = {h_val: idx for idx, h_val in enumerate(unique_h)}
    w_to_idx = {w_val: idx for idx, w_val in enumerate(unique_w)}

    # Fill tensor
    for _, row in df.iterrows():
        t_idx = t_to_idx[row["t"]]
        h_idx = h_to_idx[row["h"]]
        w_idx = w_to_idx[row["w"]]
        y_shift = row["y_shift"]
        x_shift = row["x_shift"]

        # Fill deformation field
        deformation_field[0, t_idx, h_idx, w_idx] = y_shift  # Channel 0 = y_shift
        deformation_field[1, t_idx, h_idx, w_idx] = x_shift  # Channel 1 = x_shift

    print(f"Successfully loaded deformation field with shape {deformation_field.shape}")
    print(
        f"Deformation field range: "
        f"y=[{deformation_field[0].min():.3f}, {deformation_field[0].max():.3f}], "
        f"x=[{deformation_field[1].min():.3f}, {deformation_field[1].max():.3f}]"
    )

    return deformation_field
