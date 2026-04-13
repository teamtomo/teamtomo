"""Common three-dimensional shapes for masking or filtering."""

import einops
import torch

from .coordinate_grid import coordinate_grid
from .fftfreq_grid import dft_center
from .geometry import _angle_between_vectors
from .soft_edge import add_soft_edge_3d


def sphere(
    radius: float,
    image_shape: tuple[int, int, int] | int,
    center: tuple[float, float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct a 3D spherical mask.

    Parameters
    ----------
    radius: float
        Radius of the sphere in pixels.
    image_shape: tuple[int, int, int] | int
        Shape `(d, h, w)` of 3D image(s) for the sphere mask.
    center: tuple[float, float, float] | None
        `(d, h, w)` coordinates of the center of the sphere. If `None`, the center of
        the image will be used.
    smoothing_radius: float
        Radius of the soft edge to be added around the sphere.
    device: torch.device | None
        Device on which to create the mask. If `None`, the device of the output will be
        the same as the default torch device.

    Returns
    -------
    sphere_mask: torch.Tensor
        `(d, h, w)` array with values in [0, 1].
    """
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape, image_shape)
    if center is None:
        center = dft_center(image_shape, rfft=False, fftshift=True)
    distances = coordinate_grid(
        image_shape=image_shape,
        center=center,
        norm=True,
        device=device,
    )
    mask = torch.zeros_like(distances, dtype=torch.bool)
    mask[distances < radius] = 1
    return add_soft_edge_3d(mask, smoothing_radius=smoothing_radius)


def cuboid(
    dimensions: tuple[float, float, float],
    image_shape: tuple[int, int, int] | int,
    center: tuple[float, float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct a 3D cuboidal mask.

    Parameters
    ----------
    dimensions: tuple[float, float, float]
        `(d, h, w)` dimensions of the cuboid in pixels.
    image_shape: tuple[int, int, int] | int
        Shape `(d, h, w)` of 3D image(s) for the cuboid mask.
    center: tuple[float, float, float] | None
        `(d, h, w)` coordinates of the center of the cuboid. If `None`, the center of
        the image will be used.
    smoothing_radius: float
        Radius of the soft edge to be added around the cuboid.
    device: torch.device | None
        Device on which to create the mask. If `None`, the device of the output will be
        the same as the default torch device.

    Returns
    -------
    cuboid_mask: torch.Tensor
        `(d, h, w)` array with values in [0, 1].
    """
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape, image_shape)
    if center is None:
        center = dft_center(image_shape, rfft=False, fftshift=True)
    coordinates = coordinate_grid(
        image_shape=image_shape,
        center=center,
        device=device,
    )
    dd, dh, dw = dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2
    depth_mask = torch.logical_and(coordinates[..., 0] > -dd, coordinates[..., 0] < dd)
    height_mask = torch.logical_and(coordinates[..., 1] > -dh, coordinates[..., 1] < dh)
    width_mask = torch.logical_and(coordinates[..., 2] > -dw, coordinates[..., 2] < dw)
    mask = einops.rearrange(
        [depth_mask, height_mask, width_mask], "dhw d h w -> d h w dhw"
    )
    mask = torch.all(mask, dim=-1)
    return add_soft_edge_3d(mask, smoothing_radius=smoothing_radius)


def cube(
    sidelength: float,
    image_shape: tuple[int, int, int] | int,
    center: tuple[float, float, float] | None = None,
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct a 3D cubic mask.

    Parameters
    ----------
    sidelength: float
        Length of the sides of the cube in pixels.
    image_shape: tuple[int, int, int] | int
        Shape `(d, h, w)` of 3D image(s) for the cube mask.
    center: tuple[float, float, float] | None
        `(d, h, w)` coordinates of the center of the cube. If `None`, the center of
        the image will be used.
    smoothing_radius: float
        Radius of the soft edge to be added around the cube.
    device: torch.device | None
        Device on which to create the mask. If `None`, the device of the output will be
        the same as the default torch device.

    Returns
    -------
    cube_mask: torch.Tensor
        `(d, h, w)` array with values in [0, 1].
    """
    cube = cuboid(
        dimensions=(sidelength, sidelength, sidelength),
        image_shape=image_shape,
        center=center,  # type: ignore[arg-type]
        smoothing_radius=smoothing_radius,
        device=device,
    )
    return cube


def cone(
    aperture: float,
    image_shape: tuple[int, int, int] | int,
    principal_axis: tuple[float, float, float] = (1, 0, 0),
    smoothing_radius: float = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Construct a 3D cone-shaped mask.

    Parameters
    ----------
    aperture: float
        Aperture of the cone in degrees.
    image_shape: tuple[int, int, int] | int
        Shape `(d, h, w)` of 3D image(s) for the cone mask.
    principal_axis: tuple[float, float, float]
        Vector defining the principal axis of the cone.
    smoothing_radius: float
        Radius of the soft edge to be added around the cone.
    device: torch.device | None
        Device on which to create the mask. If `None`, the device of the output will be
        the same as the default torch device.

    Returns
    -------
    cone_mask: torch.Tensor
        `(d, h, w)` array with values in [0, 1].
    """
    if isinstance(image_shape, int):
        image_shape = (image_shape, image_shape, image_shape)
    center = dft_center(image_shape, rfft=False, fftshift=True)
    vectors = coordinate_grid(
        image_shape=image_shape,
        center=center,
        device=device,
    ).float()
    vectors_norm = einops.reduce(vectors**2, "... c -> ... 1", reduction="sum") ** 0.5
    vectors /= vectors_norm
    principal_axis = torch.as_tensor(principal_axis, dtype=vectors.dtype, device=device)
    principal_axis_norm = einops.reduce(
        principal_axis**2,  # type: ignore[operator]
        "... c -> ... 1",
        reduction="sum",
    )
    principal_axis_norm = principal_axis_norm**0.5
    principal_axis /= principal_axis_norm
    angles = _angle_between_vectors(vectors, principal_axis)
    acute_bound = aperture / 2
    obtuse_bound = 180 - acute_bound
    in_cone = torch.logical_or(angles <= acute_bound, angles >= obtuse_bound)
    dc_d, dc_h, dc_w = dft_center(image_shape, rfft=False, fftshift=True)
    in_cone[dc_d, dc_h, dc_w] = True
    return add_soft_edge_3d(in_cone, smoothing_radius=smoothing_radius)
