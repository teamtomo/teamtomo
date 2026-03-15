"""Utility functions associated with backend functions."""

import torch
import einops


def normalize_template_projection(
    projections: torch.Tensor,
    small_shape: tuple[int, int],
    large_shape: tuple[int, int],
) -> torch.Tensor:
    r"""Subtract mean of edge values and set variance to 1 (in large shape).

    This function uses the fact that variance of a sequence, Var(X), is scaled by the
    relative size of the small (unpadded) and large (padded with zeros) space. Some
    negligible error is introduced into the variance (~1e-4) due to this routine.

    Let $X$ be the large, zero-padded projection and $x$ the small projection each
    with sizes $(H, W)$ and $(h, w)$, respectively. The mean of the zero-padded
    projection in terms of the small projection is:
    .. math::
        \begin{align}
            \mu(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} X_{ij} \\
            \mu(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{h} \sum_{j=1}^{w} X_{ij} + 0 \\
            \mu(X) &= \frac{h \cdot w}{H \cdot W} \mu(x)
        \end{align}
    The variance of the zero-padded projection in terms of the small projection can be
    obtained by:
    .. math::
        \begin{align}
            Var(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} (X_{ij} -
                \mu(X))^2 \\
            Var(X) &= \frac{1}{H \cdot W} \left(\sum_{i=1}^{h}
                \sum_{j=1}^{w} (X_{ij} - \mu(X))^2 +
                \sum_{i=h+1}^{H}\sum_{i=w+1}^{W} \mu(X)^2 \right) \\
            Var(X) &= \frac{1}{H \cdot W} \sum_{i=1}^{h} \sum_{j=1}^{w} (X_{ij} -
                \mu(X))^2 + (H-h)(W-w)\mu(X)^2
        \end{align}

    Parameters
    ----------
    projections : torch.Tensor
        `(..., h, w)` real-space projections of the template (in small space).
    small_shape : tuple[int, int]
        `(h, w)` shape of the template (in real space).
    large_shape : tuple[int, int]
        `(h_im, w_im)` shape of the image (in real space).

    Returns
    -------
    projections: torch.Tensor
        `(..., h, w)` edge-mean subtracted projections
         normalized so variance of zero-padded projection would be 1.
    """
    h, w = small_shape
    h_im, w_im = large_shape

    # Extract edges while preserving batch dimensions
    top_edge = projections[..., 0, :]  # shape: (..., w)
    bottom_edge = projections[..., -1, :]  # shape: (..., w)
    left_edge = projections[..., 1:-1, 0]  # shape: (..., h-2)
    right_edge = projections[..., 1:-1, -1]  # shape: (..., h-2)
    edge_pixels = torch.concatenate(
        [top_edge, bottom_edge, left_edge, right_edge], dim=-1
    )  # shape: (..., w + w + h-2 + h-2)

    # Subtract the edge pixel mean and calculate variance of small, unpadded projection
    edge_mean = einops.reduce(edge_pixels, "... b -> ...", reduction="mean")
    edge_mean = einops.rearrange(edge_mean, "... -> ... 1 1")
    projections -= edge_mean

    # Fast calculation of mean/var using Torch + appropriate scaling.
    relative_size = h * w / (h_im * w_im)
    per_image_mean = einops.reduce(projections, "... h w -> ...", reduction="mean")
    per_image_mean *= relative_size**2

    # First term of the variance calculation
    diff = projections - einops.rearrange(per_image_mean, "... -> ... 1 1")
    per_image_variance = einops.reduce(
        diff**2, pattern="... h w -> ...", reduction="sum"
    )

    # Add the second term of the variance calculation
    per_image_variance += (h_im - h) * (w_im - w) * per_image_mean**2
    per_image_variance /= h_im * w_im
    per_image_variance = einops.rearrange(per_image_variance, "... -> ... 1 1")

    return projections / torch.sqrt(per_image_variance)
