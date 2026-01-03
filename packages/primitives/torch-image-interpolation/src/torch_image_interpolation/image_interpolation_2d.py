from typing import Literal

import einops
import torch
import torch.nn.functional as F

from torch_image_interpolation import utils
from .grid_sample_utils import array_to_grid_sample


def sample_image_2d(
    image: torch.Tensor,
    coordinates: torch.Tensor,
    interpolation: Literal['nearest', 'bilinear', 'bicubic'] = 'bilinear',
) -> torch.Tensor:
    """Sample a 2D image with a specific interpolation mode.

    Parameters
    ----------
    image: torch.Tensor
        `(h, w)` image or `(c, h, w)` multichannel image.
    coordinates: torch.Tensor
        `(..., 2)` array of coordinates at which `image` should be sampled.
        - Coordinates are ordered `yx` and are positions in the `h` and `w` dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    interpolation: Literal['nearest', 'bilinear', 'bicubic']
        Interpolation mode for image sampling.

    Returns
    -------
    samples: torch.Tensor
        `(..., )` or `(..., c)` array of samples from `image`.
    """
    device = coordinates.device

    if image.ndim not in (2, 3):
        raise ValueError(f'image should have shape (h, w) or (c, h, w), got {image.shape}')

    # keep track of a few image properties
    input_image_is_complex = torch.is_complex(image)
    input_image_is_multichannel = image.ndim == 3

    # coerce single channel to multi-channel
    if input_image_is_multichannel is False:
        image = einops.rearrange(image, 'h w -> 1 h w')

    # set up for sampling with torch.nn.functional.grid_sample
    # shape (..., 2) -> (n, 2)
    coordinates, ps = einops.pack([coordinates], pattern='* yx')
    n_samples = coordinates.shape[0]
    h, w = image.shape[-2:]

    # handle complex input
    if input_image_is_complex:
        # cannot sample complex tensors directly with grid_sample
        # c.f. https://github.com/pytorch/pytorch/issues/67634
        # workaround: treat real and imaginary parts as separate channels
        image = torch.view_as_real(image)
        image = einops.rearrange(image, 'c h w complex -> (complex c) h w')

    # torch.nn.functional.grid_sample is set up for sampling grids
    # here we view our image as a batch of n_samples multi-channel images
    # then sample a batch of (1x1) grids
    # this enables sampling arbitrarily shaped arrays of coords
    image = einops.repeat(image, 'c h w -> b c h w', b=n_samples)
    coordinates = einops.rearrange(coordinates, 'b yx -> b 1 1 yx')  # b h w yx

    # take the samples
    samples = F.grid_sample(
        input=image,
        grid=array_to_grid_sample(coordinates, array_shape=image.shape[-2:]),
        mode=interpolation,
        padding_mode='border',  # this increases sampling fidelity at edges
        align_corners=True,
    )

    # reconstruct complex valued samples if required
    if input_image_is_complex is True:
        samples = einops.rearrange(samples, 'b (complex c) 1 1 -> b c complex', complex=2)
        samples = utils.view_as_complex(samples.contiguous())  # (b, c)
    else:
        samples = einops.rearrange(samples, 'b c 1 1 -> b c')

    # set samples from outside of image to zero explicitly
    coordinates = einops.rearrange(coordinates, 'b 1 1 yx -> b yx')
    image_shape = torch.as_tensor(image.shape[-2:]).to(device)
    inside = torch.logical_and(coordinates >= 0, coordinates <= image_shape - 1)
    inside = torch.all(inside, dim=-1)  # (b,)
    samples[~inside] *= 0

    # pack samples back into the expected shape
    # (b, c) -> (..., c)
    [samples] = einops.unpack(samples, pattern='* c', packed_shapes=ps)

    # ensure output has correct shape
    # - (...) if input image was single channel
    # - (..., c) if input image was multi-channel
    if input_image_is_multichannel is False:
        samples = einops.rearrange(samples, '... 1 -> ...')  # drop channel dim

    return samples


def insert_into_image_2d(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    image: torch.Tensor,
    weights: torch.Tensor | None = None,
    interpolation: Literal['nearest', 'bilinear'] = 'bilinear',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert values into a 2D image with bilinear interpolation.

    Parameters
    ----------
    values: torch.Tensor
        `(...)` or `(..., c)` array of values to be inserted into `image`.
    coordinates: torch.Tensor
        `(..., 2)` array of 2D coordinates for each value in `data`.
        - Coordinates are ordered `yx` and are positions in the `h` and `w` dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    image: torch.Tensor
        `(h, w)` or `(c, h, w)` array containing the image into which data will be inserted.
    weights: torch.Tensor | None
        `(h, w)` array containing weights associated with each pixel in `image`.
        This is useful for tracking weights across multiple calls to this function.
    interpolation: Literal['nearest', 'bilinear']
        Interpolation mode used for adding data points to grid.

    Returns
    -------
    image, weights: tuple[torch.Tensor, torch.Tensor]
        The image and weights after updating with data from `data` at `coordinates`.
    """
    # keep track of a few properties of the inputs
    input_image_is_multichannel = image.ndim == 3
    h, w = image.shape[-2:]

    # validate inputs
    values_shape = values.shape[:-1] if input_image_is_multichannel else values.shape
    coordinates_shape, coordinates_ndim = coordinates.shape[:-1], coordinates.shape[-1]

    if values_shape != coordinates_shape:
        raise ValueError('One coordinate pair is required for each value in data.')
    if coordinates_ndim != 2:
        raise ValueError('Coordinates must be 2D with shape (..., 2).')
    if image.dtype != values.dtype:
        raise ValueError('Image and values must have the same dtype.')

    if weights is None:
        weights = torch.zeros(size=(h, w), dtype=torch.float32, device=image.device)

    # add channel dim to both image and values if input image is not multichannel
    if not input_image_is_multichannel:
        image = einops.rearrange(image, 'h w -> 1 h w')
        values = einops.rearrange(values, '... -> ... 1')

    # linearise data and coordinates
    values, _ = einops.pack([values], pattern='* c')
    coordinates, _ = einops.pack([coordinates], pattern='* yx')
    coordinates = coordinates.float()

    # only keep data and coordinates inside the image
    image_shape = torch.tensor((h, w), device=image.device, dtype=torch.float32)
    upper_bound = image_shape - 1
    idx_inside = (coordinates >= 0) & (coordinates <= upper_bound)
    idx_inside = torch.all(idx_inside, dim=-1)
    values, coordinates = values[idx_inside], coordinates[idx_inside]

    # splat data onto grid
    if interpolation == 'nearest':
        image, weights = _insert_nearest_2d(values, coordinates, image, weights)
    if interpolation == 'bilinear':
        image, weights = _insert_linear_2d(values, coordinates, image, weights)

    # ensure correct output image shape
    # single channel input -> (h, w)
    # multichannel input -> (c, h, w)
    if not input_image_is_multichannel:
        image = einops.rearrange(image, '1 h w -> h w')

    return image, weights


def _insert_nearest_2d(
    data,  # (b, c)
    coordinates,  # (b, yx)
    image,  # (c, h, w)
    weights  # (h, w)
):
    # b is number of data points to insert per channel, c is number of channels
    b, c = data.shape

    # flatten data to insert values for all channels with one call to _index_put()
    data = einops.rearrange(data, 'b c -> b c')

    # find nearest voxel for each coordinate
    coordinates = torch.round(coordinates).long()
    idx_h, idx_w = einops.rearrange(coordinates, 'b yx -> yx b')

    # insert ones into weights image (h, w) at each position
    w = torch.ones(size=(b, 1), device=weights.device, dtype=weights.dtype)

    # setup indices for insertion
    idx_c = torch.arange(c, device=coordinates.device, dtype=torch.long)
    idx_c = einops.rearrange(idx_c, 'c -> 1 c')
    idx_h = einops.rearrange(idx_h, 'b -> b 1')
    idx_w = einops.rearrange(idx_w, 'b -> b 1')

    # insert image data and weights
    image.index_put_(indices=(idx_c, idx_h, idx_w), values=data, accumulate=True)
    weights.index_put_(indices=(idx_h, idx_w), values=w, accumulate=True)
    return image, weights


def _insert_linear_2d(
    data,  # (b, c)
    coordinates,  # (b, yx)
    image,  # (c, h, w)
    weights  # (h, w)
):
    # b is number of data points to insert per channel, c is number of channels
    b, c = data.shape

    # cache corner coordinates for each value to be inserted
    #     C10---C11
    #      |  P  |
    #     C00---C01
    coordinates = einops.rearrange(coordinates, 'b yx -> yx b')
    y0, x0 = torch.floor(coordinates)
    y1, x1 = torch.ceil(coordinates)

    # populate arrays of corner indices
    idx_h = torch.empty(size=(b, 2, 2), dtype=torch.long, device=image.device)
    idx_w = torch.empty(size=(b, 2, 2), dtype=torch.long, device=image.device)

    idx_h[:, 0, 0], idx_w[:, 0, 0] = y0, x0  # C00
    idx_h[:, 0, 1], idx_w[:, 0, 1] = y0, x1  # C01
    idx_h[:, 1, 0], idx_w[:, 1, 0] = y1, x0  # C10
    idx_h[:, 1, 1], idx_w[:, 1, 1] = y1, x1  # C11

    # calculate linear interpolation weights for each corner
    y, x = coordinates
    ty, tx = y - y0, x - x0  # fractional position between corners
    w = torch.empty(size=(b, 2, 2), device=image.device, dtype=weights.dtype)
    w[:, 0, 0] = (1 - ty) * (1 - tx)   # C00
    w[:, 0, 1] = (1 - ty) * tx         # C01
    w[:, 1, 0] = ty * (1 - tx)         # C10
    w[:, 1, 1] = ty * tx               # C11

    # make sure indices broadcast correctly
    idx_c = torch.arange(c, device=coordinates.device, dtype=torch.long)
    idx_c = einops.rearrange(idx_c, 'c -> 1 c 1 1')
    idx_h = einops.rearrange(idx_h, 'b h w -> b 1 h w')
    idx_w = einops.rearrange(idx_w, 'b h w -> b 1 h w')

    # insert weighted data and weight values at each corner across all channels
    # make sure to do atomic adds
    data = einops.rearrange(data, 'b c -> b c 1 1')
    w = einops.rearrange(w, 'b h w -> b 1 h w')
    image.index_put_(
        indices=(idx_c, idx_h, idx_w),
        values=data * w.to(data.dtype),
        accumulate=True
    )
    weights.index_put_(indices=(idx_h, idx_w), values=w, accumulate=True)

    return image, weights
