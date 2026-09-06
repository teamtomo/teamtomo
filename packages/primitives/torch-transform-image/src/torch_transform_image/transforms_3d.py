from typing import Literal, Optional

import einops
import torch
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_3d import Rx, Ry, Rz, T
from torch_grid_utils import coordinate_grid, dft_center
from torch_image_interpolation import sample_image_3d


def affine_transform_image_3d(
    image: torch.Tensor,
    matrices: torch.Tensor,
    interpolation: Literal["nearest", "trilinear"],
    output_shape: Optional[tuple] = None,
    zyx_matrices: bool = False,
) -> torch.Tensor:
    # grab image dimensions
    if output_shape:
        d, h, w = output_shape
    else:
        d, h, w = image.shape[-3:]

    if not zyx_matrices:
        matrices = matrices.clone()  # dont modify the input tensor
        matrices[..., :3, :3] = torch.flip(matrices[..., :3, :3], dims=(-2, -1))
        matrices[..., :3, 3] = torch.flip(matrices[..., :3, 3], dims=(-1,))

    # generate grid of pixel coordinates
    grid = coordinate_grid(image_shape=(d, h, w), device=image.device)

    # apply matrix to coordinates
    grid = homogenise_coordinates(grid)  # (d, h, w, zyxw)
    grid = einops.rearrange(grid, "d h w zyxw -> d h w zyxw 1")
    grid = matrices @ grid
    grid = grid[
        ..., :3, 0
    ]  # dehomogenise coordinates: (..., d, h, w, zyxw, 1) -> (..., d, h, w, zyx)

    # sample image at transformed positions
    result = sample_image_3d(image, coordinates=grid, interpolation=interpolation)
    return result


def rotate_then_shift_image_3d(
    image: torch.Tensor,
    rotate_zyx: list[float | int] | tuple[float | int, ...] = (0, 0, 0),
    shift_zyx: list[float | int] | tuple[float | int, ...] = (0, 0, 0),
    interpolation: Literal["trilinear", "nearest"] = "trilinear",
) -> torch.Tensor:
    """
    This is a wrapper function to easily rotate and shift a 3D image.

    Image is first rotated by the specified number of degrees, according
    to the right hand rule, around the center of the image. Then, image
    is shifted by the specified number of pixels (see note about shift
    conventions below).

    Parameters
    ----------
    image : torch.Tensor
        The image to be shifted/rotated.
    rotate_zyx : list[float] | tuple[float, ...], optional
        The angles in degrees by which to rotate the image according to
        the right hand rule. Positive values rotate the image CCW. Must
        be a list or tuple of length 3 in the order (z, y, x). If
        multiple angles are provided, rotations will be performed in z,
        y, x order.
    shift_zyx : list[float] | tuple[float, ...], optional
        The number of pixels by which to shift the image. Positive
        values shift up/right. Must be a list or tuple of length 3 in
        the form (z, y, x).
    interpolation : Literal["trilinear", "nearest"], optional
        The interpolation method to use.

    Returns
    -------
    torch.Tensor
        The shifted and/or rotated image.

    See Also
    --------
    shift_then_rotate_image_3d transforms_2d.rotate_then_shift_image_2d

    Notes
    -----
    Shift direction assumes the origin (0, 0, 0) of the image is in the
    bottom left (following convention in cryo-EM image processing).
    Matplotlib and plotly display images with y = 0 at the top by
    default so your image may be shifted opposite of what you expect. If
    you want to shift the other direction, just reverse the sign of your
    shift argument.

    """
    image_center = torch.as_tensor(0, device=image.device, dtype=torch.float32)
    if any(rotate_zyx):
        d, h, w = image.shape[-3:]
        image_center = dft_center(
            image_shape=(d, h, w), device=image.device, fftshift=True, rfft=False
        )

    matrix = _build_rotate_shift_matrix_3d(rotate_zyx, shift_zyx, image_center, rotate_first=True)
    return affine_transform_image_3d(
        image=image,
        matrices=matrix,
        interpolation=interpolation,
        zyx_matrices=True,
    )


def shift_then_rotate_image_3d(
    image: torch.Tensor,
    rotate_zyx: list[float | int] | tuple[float | int, ...] = (0, 0, 0),
    shift_zyx: list[float | int] | tuple[float | int, ...] = (0, 0, 0),
    interpolation: Literal["nearest", "trilinear"] = "trilinear",
) -> torch.Tensor:
    """
    This is a wrapper function to easily shift and rotate a 3D image.

    Image is first shifted by the specified number of pixels (see note
    about shift conventions below). Then, image is rotated by the specified
    number of degrees, according to the right hand rule, around the center
    of the image.

    Parameters
    ----------
    image : torch.Tensor
        The image to be shifted/rotated.
    rotate_zyx : list[float] | tuple[float, ...], optional
        The angles in degrees by which to rotate the image according to the
        right hand rule. Positive values rotate the image CCW. Must be a
        list or tuple of length 3 in the order (z, y, x). If
        multiple angles are provided, rotations will be performed in z,
        y, x order.
    shift_zyx : list[float] | tuple[float, ...], optional
        The number of pixels by which to shift the image. Positive values
        shift up/right. Must be a list or tuple of length 3 in the form (z,
        y, x).
    interpolation : Literal["trilinear", "nearest"], optional
        The interpolation method to use.

    Returns
    -------
    torch.Tensor
        The shifted and/or rotated image.

    See Also
    --------
    rotate_then_shift_image_3d
    transforms_2d.shift_then_rotate_image_2d

    Notes
    -----
    Shift direction assumes the origin (0, 0, 0) of the image is in the
    bottom left (following convention in cryo-EM image processing).
    Matplotlib and plotly display images with y = 0 at the top by default
    so your image may be shifted opposite of what you expect. If you want
    to shift the other direction, just reverse the sign of your shift
    argument.

    """
    image_center = torch.as_tensor(0, device=image.device, dtype=torch.float32)
    if any(rotate_zyx):
        d, h, w = image.shape[-3:]
        image_center = dft_center(
            image_shape=(d, h, w), device=image.device, fftshift=True, rfft=False
        )

    matrix = _build_rotate_shift_matrix_3d(rotate_zyx, shift_zyx, image_center, rotate_first=False)
    return affine_transform_image_3d(
        image=image,
        matrices=matrix,
        interpolation=interpolation,
        zyx_matrices=True,
    )

def rotate_image_3d_about_tilt_axis(
    image: torch.Tensor,
    tilt_deg: float | int,
    tilt_axis_angle: float | int = 90.0,
    interpolation: Literal["trilinear", "nearest"] = "trilinear",
) -> torch.Tensor:
    """Rotate a 3D image about an axis lying in the XY plane.

    The tilt axis is oriented in XY at ``tilt_axis_angle`` degrees from +X
    toward +Y (0 → about X, 90 → about Y). ``tilt_deg`` is the rotation about
    that axis (right-hand rule). Implemented as
    ``Rz(φ) @ Rx(θ) @ Rz(-φ)`` with a single resample about the image center.

    Parameters
    ----------
    image : torch.Tensor
        Volume with shape ``(..., d, h, w)``.
    tilt_deg : float | int
        Rotation angle about the tilt axis, in degrees.
    tilt_axis_angle : float | int
        Orientation of the tilt axis in the XY plane, in degrees from +X
        toward +Y. Default ``90`` (Y axis).
    interpolation : {"trilinear", "nearest"}
        Sampling mode passed to :func:`affine_transform_image_3d`.

    Returns
    -------
    torch.Tensor
        Rotated volume, same shape as ``image``. Out-of-bounds samples are
        zero (see :func:`torch_image_interpolation.sample_image_3d`).
    """
    phi = float(tilt_axis_angle)
    theta = float(tilt_deg)
    device = image.device
    d, h, w = image.shape[-3:]
    center = dft_center(
        image_shape=(d, h, w), device=device, fftshift=True, rfft=False
    )
    R = (
        Rz(phi, zyx=True, device=device)
        @ Rx(theta, zyx=True, device=device)
        @ Rz(-phi, zyx=True, device=device)
    )
    matrix = T(center, device=device) @ R @ T(-center, device=device)
    matrix = torch.inverse(matrix)
    return affine_transform_image_3d(
        image=image,
        matrices=matrix,
        interpolation=interpolation,
        zyx_matrices=True,
    )


def _build_rotate_shift_matrix_3d(
        rotate_zyx: list[float | int] | tuple[float | int, ...],
        shift_zyx: list[float | int] | tuple[float | int, ...],
        image_center: torch.Tensor,
        rotate_first: bool,
) -> torch.Tensor:
    if (num_angles := len(rotate_zyx)) != 3:
        e = f"3 angles (zyx) are required but {num_angles} were supplied: {rotate_zyx}."
        raise ValueError(e)
    if (num_shifts := len(shift_zyx)) != 3:
        e = f"3 shifts (zyx) are required but {num_shifts} were supplied: {shift_zyx}."
        raise ValueError(e)

    device = image_center.device
    rotation_matrix = (
            Rx(rotate_zyx[2], zyx=True, device=device)
            @ Ry(rotate_zyx[1], zyx=True, device=device)
            @ Rz(rotate_zyx[0], zyx=True, device=device)
        )
    translation_matrix = T(shift_zyx, device=device)

    if rotate_first:
        inner_matrix = translation_matrix @ rotation_matrix
    else:
        inner_matrix = rotation_matrix @ translation_matrix
    matrix = T(image_center, device=device) @ inner_matrix @ T(-image_center, device=device)
    # Matrix is inverted because it is applied to the coordinate grid,
    # not the image directly.
    return torch.inverse(matrix)
