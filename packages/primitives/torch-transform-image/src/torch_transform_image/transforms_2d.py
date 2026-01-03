from typing import Literal, Optional

import einops
import torch
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_2d import R, T
from torch_grid_utils import coordinate_grid, dft_center
from torch_image_interpolation import sample_image_2d


def affine_transform_image_2d(
        image: torch.Tensor,
        matrices: torch.Tensor,
        interpolation: Literal['nearest', 'bilinear', 'bicubic'],
        output_shape: Optional[tuple] = None,
        yx_matrices: bool = False,
) -> torch.Tensor:
    # grab image dimensions
    if output_shape:
        h, w = output_shape
    else:
        h, w = image.shape[-2:]

    if not yx_matrices:
        matrices = matrices.clone()  # dont modify the input tensor
        matrices[..., :2, :2] = (
            torch.flip(matrices[..., :2, :2], dims=(-2, -1))
        )
        matrices[..., :2, 2] = torch.flip(matrices[..., :2, 2], dims=(-1,))

    # generate grid of pixel coordinates
    grid = coordinate_grid(image_shape=(h, w), device=image.device)

    # apply matrix to coordinates
    grid = homogenise_coordinates(grid)  # (h, w, yxw)
    grid = einops.rearrange(grid, 'h w yxw -> h w yxw 1')
    grid = matrices @ grid
    grid = grid[..., :2, 0]  # dehomogenise coordinates: (..., h, w, yxw, 1) -> (..., h, w, yx)

    # sample image at transformed positions
    result = sample_image_2d(image, coordinates=grid, interpolation=interpolation)
    return result


def rotate_then_shift_image_2d(
    image: torch.Tensor,
    rotate: int | float = 0,
    shift: list[float | int] | tuple[float | int, float | int] = (0, 0),
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic",
) -> torch.Tensor:
    """This is a wrapper function to easily rotate and shift a 2D image.

    The image is rotated CCW around its center by the specified number
    of degrees and then shifted up/left by the specified number of
    pixels (see note about direction conventions below!). Currently,
    only a single shift and a single rotation are allowed.

    Parameters
    ----------
    image : torch.Tensor
        The image to be shifted/rotated.
    angle : int | float, optional
        The angle in degrees by which to rotate the image.
    shift : list[float | int] | tuple[float | int, float | int], optional
        The number of pixels by which to shift the image. Positive
        values shift up/right. Must be a list or tuple of length 2 in
        the form (y, x).
    interpolation : Literal["nearest", "bilinear", "bicubic"], optional
        The interpolation method to use. Default is "bicubic".

    Returns
    -------
    torch.Tensor
        The shifted and/or rotated image.

    Notes
    -----
    The description of operations assumes the origin (0,0) of the image
    is in the lower left (following convention in cryo-EM image
    processing). This is NOT the default for images displayed by
    matplotlib, plotly, etc. so images may be tranformed in the opposite
    direction from expected. If you want to tranform the other
    direction, just reverse the sign of your rotate and shift arguments.
    """
    image_center = 0
    if rotate:
        h, w = image.shape[-2:]
        image_center = dft_center(
            image_shape=(h, w), device=image.device, fftshift=True, rfft=False
            )

    center_tensor = torch.as_tensor(image_center, device=image.device, dtype=torch.float32)
    rotate_tensor = torch.as_tensor(rotate, device=image.device, dtype=torch.float32)
    # Because shift is applied to the coordinate grid, it must be
    # negated to produce a positive (up/right) shift on the image.
    shift_tensor = -torch.as_tensor(shift, device=image.device, dtype=torch.float32)

    if rotate_tensor.numel() > 1 or shift_tensor.numel() > 2:
        raise NotImplementedError(
            "Only single angle and single shift values are supported."
            )

    matrix = T(center_tensor) @ R(rotate_tensor) @ T(shift_tensor) @ T(-center_tensor)

    return affine_transform_image_2d(
        image=image,
        matrices=matrix,
        interpolation=interpolation,
        yx_matrices=True,
    )


def shift_then_rotate_image_2d(
    image: torch.Tensor,
    rotate: int | float = 0,
    shift: list[float | int] | tuple[float | int, float | int] = (0, 0),
    interpolation: Literal["nearest", "bilinear", "bicubic"] = "bicubic",
):
    """This is a wrapper function to easily shift and rotate a 2D image.

    The image is shifted up/left by the specified number of pixels and
    then rotated CCW around its center by the specified number of
    degrees (see note about direction conventions below!). Currently,
    only a single shift and a single rotation are allowed.

    Parameters
    ----------
    image : torch.Tensor
        The image to be shifted/rotated.
    angle : int | float, optional
        The angle in degrees by which to rotate the image.
    shift : list[float | int] | tuple[float | int, float | int], optional
        The number of pixels by which to shift the image. Positive
        values shift up/right. Must be a list or tuple of length 2 in
        the form (y, x).
    interpolation : Literal["nearest", "bilinear", "bicubic"], optional
        The interpolation method to use. Default is "bicubic".

    Returns
    -------
    torch.Tensor
        The shifted and/or rotated image.

    Notes
    -----
    The description of operations assumes the origin (0,0) of the image
    is in the lower left (following convention in cryo-EM image
    processing). This is NOT the default for images displayed by
    matplotlib, plotly, etc. so images may be tranformed in the opposite
    direction from expected. If you want to tranform the other
    direction, just reverse the sign of your rotate and shift arguments.
    """
    image_center = 0
    if rotate:
        h, w = image.shape[-2:]
        image_center = dft_center(
            image_shape=(h, w), device=image.device, fftshift=True, rfft=False
            )

    center_tensor = torch.as_tensor(image_center, device=image.device, dtype=torch.float32)
    rotate_tensor = torch.as_tensor(rotate, device=image.device, dtype=torch.float32)
    # Because shift is applied to the coordinate grid, it must be
    # negated to produce a positive (up/right) shift on the image.
    shift_tensor = -torch.as_tensor(shift, device=image.device, dtype=torch.float32)

    if rotate_tensor.numel() > 1 or shift_tensor.numel() > 2:
        raise NotImplementedError(
            "Only single angle and single shift values are supported."
            )

    matrix = T(center_tensor) @ T(shift_tensor) @ R(rotate_tensor) @ T(-center_tensor)

    return affine_transform_image_2d(
        image=image,
        matrices=matrix,
        interpolation=interpolation,
        yx_matrices=True,
    )
