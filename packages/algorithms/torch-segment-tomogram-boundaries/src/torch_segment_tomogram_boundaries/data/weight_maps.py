from typing import Optional

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure


def generate_boundary_weight_map(
    mask: np.ndarray,
    high_weight: float = 2.5,
    base_weight: Optional[float] = None,
    boundary_width: int = 1,
) -> np.ndarray:
    """Generate a normalized boundary-focused weight map.

    Parameters
    ----------
    mask : np.ndarray
        Binary ground-truth mask (0 background, 1 foreground).
    high_weight : float, default=2.5
        Weight assigned to pixels lying on the boundary band.
    base_weight : float, optional
        Base weight for non-boundary pixels. When ``None`` the function uses
        ``high_weight / 10`` which keeps the previous heuristic intact.
    boundary_width : int, default=1
        Thickness (in pixels) of the boundary band inside the object. A value of
        1 mirrors the original behaviour where a single erosion defines the edge.

    Returns
    -------
    np.ndarray
        Normalized weight map with the same spatial size as ``mask``.
    """
    # Ensure mask is binary (0 or 1)
    binary_mask = (mask > 0).astype(np.uint8)

    if base_weight is None:
        base_weight = max(high_weight / 10.0, 1e-6)

    # Derive a predictable boundary band using morphological erosion
    if boundary_width > 0:
        structure = generate_binary_structure(rank=binary_mask.ndim, connectivity=1)
        eroded_mask = binary_erosion(
            binary_mask, structure=structure, iterations=boundary_width, border_value=0
        )
        boundary = (binary_mask.astype(bool) & ~eroded_mask).astype(np.uint8)
    else:
        boundary = np.zeros_like(binary_mask, dtype=np.uint8)

    # Create the weight map
    weight_map = np.full(mask.shape, base_weight, dtype=np.float32)
    weight_map[boundary == 1] = high_weight

    # Normalize so the mean stays close to 1 and gradients remain stable
    mean_val = weight_map.mean()
    if mean_val > 0:
        weight_map = weight_map / mean_val

    return weight_map
