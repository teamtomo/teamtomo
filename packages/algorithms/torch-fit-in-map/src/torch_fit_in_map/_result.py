"""AlignmentResult dataclass returned by all alignment functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class AlignmentResult:
    """Result of a volume alignment operation.

    Parameters
    ----------
    rotation_matrix : torch.Tensor
        ``(3, 3)`` rotation matrix in zyx convention.  This is the matrix M to
        pass as ``matrices`` to
        ``affine_transform_image_3d(mobile, T(c) @ M_4x4 @ T(-c), ...,
        zyx_matrices=True)`` to rotate the mobile volume onto the reference.
    translation_pixels : torch.Tensor
        ``(3,)`` translation vector in zyx pixels.  After applying the rotation,
        translate the mobile by ``-translation_pixels`` to align it with the
        reference (pull convention: pass ``T(-translation_pixels)`` to
        ``affine_transform_image_3d``).
    score : float
        Peak normalised cross-correlation score (higher is better, max 1.0 for
        identical volumes).
    simulated_volume : torch.Tensor or None
        ``(d, h, w)`` simulated density volume generated during
        ``fit_map_in_pdb`` / ``fit_pdb_in_map``; ``None`` unless
        ``save_simulated=True`` was set.
    translation_angstroms : torch.Tensor or None
        ``(3,)`` translation in Angstroms.  Populated only when
        ``pixel_size_angstroms`` is supplied to the alignment function.
    """

    rotation_matrix: torch.Tensor
    translation_pixels: torch.Tensor
    score: float
    simulated_volume: torch.Tensor | None = field(default=None, repr=False)
    translation_angstroms: torch.Tensor | None = field(default=None, repr=False)
