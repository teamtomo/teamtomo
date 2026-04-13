"""Refine an initial tilt axis angle."""

import einops
import torch
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]
from torch_affine_utils.transforms_2d import R

# teamtomo torch functionality
from torch_fourier_slice import project_2d_to_1d  # type: ignore[import-untyped]
from torch_grid_utils import circle  # type: ignore[import-untyped]


def refine_tilt_axis_angle(
    tilt_series: torch.Tensor,
    tilt_axis_angle: float = 0.0,
    search_range: float = 10.0,
    max_iter: int = 20,
) -> float:
    """Refine the tilt axis angle for electron tomography data.

    Uses common line projections and scipy's Brent's method (bounded) to find
    the optimal tilt axis angle that minimizes differences between projections
    across the tilt series.

    Parameters
    ----------
    tilt_series : torch.Tensor
        Tensor containing the tilt series images with shape [n_tilts, height, width].
    tilt_axis_angle : float, default=0.0
        Initial guess for the tilt axis angle in degrees.
    search_range : float, default=10.0
        Search range around the initial angle in degrees (±search_range).
    max_iter : int, default=10
        Maximum iterations for Brent's method optimizer.

    Returns
    -------
    float
        The optimized tilt axis angle in degrees.

    Notes
    -----
    The function works by:
    1. Projecting images perpendicular to the tilt axis to extract common lines
    2. Comparing these projections across different tilts
    3. Minimizing differences between projections using Brent's method

    Common line projections are normalized and weighted according to a
    projected spherical mask to emphasize regions of interest.
    """
    n_tilts, h, w = tilt_series.shape
    device = tilt_series.device
    size = min(h, w)

    # use a spherical real-space alignment mask, quick way to get around #4
    alignment_mask = circle(
        radius=size // 3,
        smoothing_radius=size // 6,
        image_shape=(h, w),
        device=device,
    )
    masked_tilt_series = tilt_series * alignment_mask

    # generate a weighting for the common line ROI by projecting the mask
    mask_weights = project_2d_to_1d(
        alignment_mask,
        torch.eye(2, device=device),  # angle does not matter for circle
    )
    mask_weights = mask_weights / mask_weights.max()  # normalise to 0 and 1

    def objective(angle: float) -> float:
        """Objective function: compute loss for given tilt axis angle."""
        # The common line is the projection perpendicular to the
        # tilt-axis, hence add 90 degrees to project along the x-axis
        angle_tensor = torch.tensor(
            [angle + 90.0] * n_tilts,
            dtype=torch.float32,
            device=device,
        )
        M = R(angle_tensor, yx=False)
        M = M[:, :2, :2]  # we only need the rotation matrix

        projections = torch.cat(
            [  # indexing with [[i]] does not drop the dimension
                project_2d_to_1d(masked_tilt_series[i], M[[i]]) for i in range(n_tilts)
            ]
        )
        projections = projections - einops.reduce(
            projections, "tilt w -> tilt 1", reduction="mean"
        )
        projections = projections / torch.std(projections, dim=(-1), keepdim=True)
        projections = projections * mask_weights  # weight the common lines

        squared_differences = (
            projections - einops.rearrange(projections, "b d -> b 1 d")
        ) ** 2
        loss = einops.reduce(squared_differences, "b1 b2 d -> 1", reduction="sum")
        return float(loss.item())

    # Define search bounds
    angle_min = tilt_axis_angle - search_range
    angle_max = tilt_axis_angle + search_range

    # Run Brent's method optimization
    result = minimize_scalar(
        objective,
        bounds=(angle_min, angle_max),
        method="bounded",
        options={"maxiter": max_iter},
    )

    return float(result.x)
