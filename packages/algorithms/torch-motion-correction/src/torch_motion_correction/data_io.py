"""Compatibility shims — logic now lives on DeformationField in types.py."""

from pathlib import Path
from typing import Union

import torch

from torch_motion_correction.deformation_field import DeformationField


def write_deformation_field_to_csv(
    deformation_field: torch.Tensor, output_path: Union[str, Path]
) -> None:
    """Write a deformation field tensor to CSV."""
    DeformationField(data=deformation_field).to_csv(output_path)


def read_deformation_field_from_csv(
    csv_path: Union[str, Path], device: torch.device | None = None
) -> torch.Tensor:
    """Read a deformation field from CSV. Returns raw tensor."""
    return DeformationField.from_csv(csv_path, device=device).data
