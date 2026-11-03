"""Real-space backprojection and its adjoint (splatting) for a TiltSeries."""

import torch
import torch.nn.functional as F
from torch_tilt_series import TiltSeries


def affine_projection(tilt_series: TiltSeries) -> tuple[torch.Tensor, torch.Tensor]:
    """`project(p) = matrices @ p + offsets`, tomogram zyx -> detector yx (A).

    Returns `(n_tilts, 2, 3)` matrices and `(n_tilts, 2)` offsets, evaluated by
    projecting the origin and three unit vectors.
    """
    scale = 1000.0
    device = tilt_series.device
    points = torch.cat(
        [torch.zeros(1, 3, device=device), scale * torch.eye(3, device=device)]
    )
    projected = tilt_series.project_points(points)  # (4, t, 2)
    offsets = projected[0]
    matrices = (projected[1:] - offsets) / scale  # (3, t, 2)
    return matrices.permute(1, 2, 0), offsets


def voxel_positions(
    matrices: torch.Tensor,
    offsets: torch.Tensor,
    volume_shape: tuple[int, int, int],
    image_shape: tuple[int, int],
    pixel_spacing: float,
    z_center: float = 0.0,
) -> torch.Tensor:
    """Detector positions of every voxel center in one tilt.

    The volume has `pixel_spacing` voxels centered on the tomogram center
    (voxel `(d // 2, h // 2, w // 2)`); detector positions are array
    coordinates in images of `image_shape` whose center pixel is
    `(h // 2, w // 2)`.

    Parameters
    ----------
    matrices : torch.Tensor
        `(2, 3)` linear part of this tilt's projection (`affine_projection`).
    offsets : torch.Tensor
        `(2, )` detector position of the tomogram center (Angstroms).
    volume_shape : tuple[int, int, int]
        `(d, h, w)` of the volume.
    image_shape : tuple[int, int]
        `(h, w)` of the images.
    pixel_spacing : float
        Voxel and pixel spacing in Angstroms.
    z_center : float
        Tomogram z (Angstroms) of the volume's center slice `d // 2`.

    Returns
    -------
    torch.Tensor
        `(d, h, w, 2)` yx array coordinates.
    """
    device = matrices.device
    z, y, x = (
        (torch.arange(n, device=device).float() - n // 2) * pixel_spacing
        for n in volume_shape
    )
    z = z + z_center
    m = matrices / pixel_spacing  # A -> px
    center = torch.tensor(
        [image_shape[0] // 2, image_shape[1] // 2], device=device
    ).float()
    c = offsets / pixel_spacing + center
    return (
        m[:, 0] * z[:, None, None, None]
        + m[:, 1] * y[None, :, None, None]
        + m[:, 2] * x[None, None, :, None]
        + c
    )


def backproject(image: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Sample one image (bilinear, zero outside) at `(d, h, w, 2)` positions."""
    hi, wi = image.shape
    d, h, w, _ = positions.shape
    grid = torch.stack(
        [
            2 * positions[..., 1] / (wi - 1) - 1,
            2 * positions[..., 0] / (hi - 1) - 1,
        ],
        dim=-1,
    ).reshape(1, d * h, w, 2)
    sampled = F.grid_sample(
        image[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(d, h, w)


def splat(
    volume: torch.Tensor, positions: torch.Tensor, image_shape: tuple[int, int]
) -> torch.Tensor:
    """Project a volume: add each voxel into the image at its position.

    Bilinear splatting, the adjoint of `backproject`.
    """
    hi, wi = image_shape
    values = volume.reshape(-1)
    p = positions.reshape(-1, 2)
    y0 = torch.floor(p[:, 0])
    x0 = torch.floor(p[:, 1])
    wy = p[:, 0] - y0
    wx = p[:, 1] - x0
    image = torch.zeros(hi * wi, device=volume.device, dtype=volume.dtype)
    for dy, dx, weight in (
        (0, 0, (1 - wy) * (1 - wx)),
        (0, 1, (1 - wy) * wx),
        (1, 0, wy * (1 - wx)),
        (1, 1, wy * wx),
    ):
        yy = (y0 + dy).long()
        xx = (x0 + dx).long()
        valid = (yy >= 0) & (yy < hi) & (xx >= 0) & (xx < wi)
        image.index_add_(0, (yy * wi + xx)[valid], (values * weight)[valid])
    return image.reshape(hi, wi)
