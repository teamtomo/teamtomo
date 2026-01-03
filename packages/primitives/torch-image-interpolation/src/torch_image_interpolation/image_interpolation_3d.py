from typing import Literal

import einops
import torch
import torch.nn.functional as F

from .grid_sample_utils import array_to_grid_sample
from torch_image_interpolation import utils


def sample_image_3d(
    image: torch.Tensor,
    coordinates: torch.Tensor,
    interpolation: Literal['nearest', 'trilinear'] = 'trilinear',
) -> torch.Tensor:
    """Sample a 3D image with a specific interpolation mode.

    Parameters
    ----------
    image: torch.Tensor
        `(d, h, w)` image or `(c, d, h, w)` multichannel image.
    coordinates: torch.Tensor
        `(..., 3)` array of coordinates at which `image` should be sampled.
        - Coordinates are ordered `zyx` and are positions in the `d`, `h` and `w` dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    interpolation: Literal['nearest', 'trilinear']
        Interpolation mode for image sampling.

    Returns
    -------
    samples: torch.Tensor
        `(..., )` or `(..., c)` array of samples from `image`.
    """
    device = coordinates.device

    if image.ndim not in (3, 4):
        raise ValueError(f'image should have shape (d, h, w) or (c, d, h, w), got {image.shape}')

    # keep track of a few image properties
    input_image_is_complex = torch.is_complex(image)
    input_image_is_multichannel = image.ndim == 4

    # coerce single channel to multi-channel
    if input_image_is_multichannel is False:
        image = einops.rearrange(image, 'd h w -> 1 d h w')

    # setup coordinates for sampling image with torch.nn.functional.grid_sample
    # shape (..., 3) -> (b, 3)
    coordinates, ps = einops.pack([coordinates], pattern='* zyx')
    n_samples = coordinates.shape[0]

    # handle complex input
    if input_image_is_complex:
        # cannot sample complex tensors directly with grid_sample
        # c.f. https://github.com/pytorch/pytorch/issues/67634
        # workaround: treat real and imaginary parts as separate channels
        image = torch.view_as_real(image)
        image = einops.rearrange(image, 'c d h w complex -> (complex c) d h w')

    # torch.nn.functional.grid_sample is set up for sampling grids
    # here we view our volume as a batch of n_samples multi-channel volumes
    # then sample a batch of (1x1x1) grids
    # this enables sampling arbitrarily shaped arrays of coords
    image = einops.repeat(image, 'c d h w -> b c d h w', b=n_samples)
    coordinates = einops.rearrange(coordinates, 'b zyx -> b 1 1 1 zyx')  # b d h w zyx

    # take the samples
    interpolation = 'bilinear' if interpolation == 'trilinear' else interpolation
    samples = F.grid_sample(
        input=image,
        grid=array_to_grid_sample(coordinates, array_shape=image.shape[-3:]),
        mode=interpolation,  # bilinear is trilinear sampling when input is volumetric
        padding_mode='border',  # this increases sampling fidelity at edges
        align_corners=True,
    )

    # reconstruct complex valued samples if required
    if input_image_is_complex is True:
        samples = einops.rearrange(samples, 'b (complex c) 1 1 1 -> b c complex', complex=2)
        samples = utils.view_as_complex(samples.contiguous())  # (b, c)
    else:
        samples = einops.rearrange(samples, 'b c 1 1 1 -> b c')

    # set samples from outside of volume to zero
    coordinates = einops.rearrange(coordinates, 'b 1 1 1 zyx -> b zyx')
    volume_shape = torch.as_tensor(image.shape[-3:]).to(device)
    inside = torch.logical_and(coordinates >= 0, coordinates <= volume_shape - 1)
    inside = torch.all(inside, dim=-1)  # (b, d, h, w)
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


def insert_into_image_3d(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    image: torch.Tensor,
    weights: torch.Tensor | None = None,
    interpolation: Literal['nearest', 'trilinear'] = 'trilinear',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert values into a 3D image with specified interpolation.

    Parameters
    ----------
    values: torch.Tensor
        `(...)` or `(..., c)` array of values to be inserted into `image`.
    coordinates: torch.Tensor
        `(..., 3)` array of 3D coordinates for each value in `data`.
        - Coordinates are ordered `zyx` and are positions in the `d`, `h` and `w` dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    image: torch.Tensor
        `(d, h, w)` or `(c, d, h, w)` array containing the image into which data will be inserted.
    weights: torch.Tensor | None
        `(d, h, w)` array containing weights associated with each voxel in `image`.
        This is useful for tracking weights across multiple calls to this function.
    interpolation: Literal['nearest', 'trilinear']
        Interpolation mode used for adding data points to grid.

    Returns
    -------
    image, weights: tuple[torch.Tensor, torch.Tensor]
        The image and weights after updating with data from `data` at `coordinates`.
    """
    # keep track of a few properties of the inputs
    input_image_is_multichannel = image.ndim == 4
    d, h, w = image.shape[-3:]

    # validate inputs
    values_shape = values.shape[:-1] if input_image_is_multichannel else values.shape
    coordinates_shape, coordinates_ndim = coordinates.shape[:-1], coordinates.shape[-1]

    if values_shape != coordinates_shape:
        raise ValueError('One coordinate triplet is required for each value in data.')
    if coordinates_ndim != 3:
        raise ValueError('Coordinates must be 3D with shape (..., 3).')
    if image.dtype != values.dtype:
        raise ValueError('Image and values must have the same dtype.')

    if weights is None:
        weights = torch.zeros(size=(d, h, w), dtype=torch.float32, device=image.device)

    # add channel dim to both image and values if input image is not multichannel
    if not input_image_is_multichannel:
        image = einops.rearrange(image, 'd h w -> 1 d h w')
        values = einops.rearrange(values, '... -> ... 1')

    # linearise data and coordinates
    values, _ = einops.pack([values], pattern='* c')
    coordinates, _ = einops.pack([coordinates], pattern='* zyx')
    coordinates = coordinates.float()

    # only keep data and coordinates inside the image
    image_shape = torch.tensor((d, h, w), device=image.device, dtype=torch.float32)
    upper_bound = image_shape - 1
    idx_inside = (coordinates >= 0) & (coordinates <= upper_bound)
    idx_inside = torch.all(idx_inside, dim=-1)
    values, coordinates = values[idx_inside], coordinates[idx_inside]

    # splat data onto grid
    if interpolation == 'nearest':
        image, weights = _insert_nearest_3d(values, coordinates, image, weights)
    if interpolation == 'trilinear':
        image, weights = _insert_linear_3d(values, coordinates, image, weights)

    # ensure correct output image shape
    # single channel input -> (d, h, w)
    # multichannel input -> (c, d, h, w)
    if not input_image_is_multichannel:
        image = einops.rearrange(image, '1 d h w -> d h w')

    return image, weights


def _insert_nearest_3d(
    data,  # (b, c)
    coordinates,  # (b, zyx)
    image,  # (c, d, h, w)
    weights  # (d, h, w)
):
    # b is number of data points to insert per channel, c is number of channels
    b, c = data.shape

    # flatten data to insert values for all channels with one call to index_put()
    data = einops.rearrange(data, 'b c -> b c')

    # find nearest voxel for each coordinate
    coordinates = torch.round(coordinates).long()
    idx_z, idx_y, idx_x = einops.rearrange(coordinates, 'b zyx -> zyx b')

    # insert ones into weights image (d, h, w) at each position
    w = torch.ones(size=(b, 1), device=weights.device, dtype=weights.dtype)

    # setup indices for insertion
    idx_c = torch.arange(c, device=coordinates.device, dtype=torch.long)
    idx_c = einops.rearrange(idx_c, 'c -> 1 c')
    idx_z = einops.rearrange(idx_z, 'b -> b 1')
    idx_y = einops.rearrange(idx_y, 'b -> b 1')
    idx_x = einops.rearrange(idx_x, 'b -> b 1')

    # insert image data and weights
    image.index_put_(indices=(idx_c, idx_z, idx_y, idx_x), values=data, accumulate=True)
    weights.index_put_(indices=(idx_z, idx_y, idx_x), values=w, accumulate=True)
    return image, weights


def _insert_linear_3d(
    data,  # (b, c)
    coordinates,  # (b, zyx)
    image,  # (c, d, h, w)
    weights  # (d, h, w)
):
    # b is number of data points to insert per channel, c is number of channels
    b, c = data.shape

    # cache corner coordinates for each value to be inserted
    coordinates = einops.rearrange(coordinates, 'b zyx -> zyx b')
    z0, y0, x0 = torch.floor(coordinates)
    z1, y1, x1 = torch.ceil(coordinates)

    # populate arrays of corner indices
    idx_z = torch.empty(size=(b, 2, 2, 2), dtype=torch.long, device=image.device)
    idx_y = torch.empty(size=(b, 2, 2, 2), dtype=torch.long, device=image.device)
    idx_x = torch.empty(size=(b, 2, 2, 2), dtype=torch.long, device=image.device)

    idx_z[:, 0, 0, 0], idx_y[:, 0, 0, 0], idx_x[:, 0, 0, 0] = z0, y0, x0  # C000
    idx_z[:, 0, 0, 1], idx_y[:, 0, 0, 1], idx_x[:, 0, 0, 1] = z0, y0, x1  # C001
    idx_z[:, 0, 1, 0], idx_y[:, 0, 1, 0], idx_x[:, 0, 1, 0] = z0, y1, x0  # C010
    idx_z[:, 0, 1, 1], idx_y[:, 0, 1, 1], idx_x[:, 0, 1, 1] = z0, y1, x1  # C011
    idx_z[:, 1, 0, 0], idx_y[:, 1, 0, 0], idx_x[:, 1, 0, 0] = z1, y0, x0  # C100
    idx_z[:, 1, 0, 1], idx_y[:, 1, 0, 1], idx_x[:, 1, 0, 1] = z1, y0, x1  # C101
    idx_z[:, 1, 1, 0], idx_y[:, 1, 1, 0], idx_x[:, 1, 1, 0] = z1, y1, x0  # C110
    idx_z[:, 1, 1, 1], idx_y[:, 1, 1, 1], idx_x[:, 1, 1, 1] = z1, y1, x1  # C111

    # calculate trilinear interpolation weights for each corner
    z, y, x = coordinates
    tz, ty, tx = z - z0, y - y0, x - x0  # fractional position between voxel corners
    w = torch.empty(size=(b, 2, 2, 2), device=image.device, dtype=weights.dtype)

    w[:, 0, 0, 0] = (1 - tz) * (1 - ty) * (1 - tx)  # C000
    w[:, 0, 0, 1] = (1 - tz) * (1 - ty) * tx        # C001
    w[:, 0, 1, 0] = (1 - tz) * ty * (1 - tx)        # C010
    w[:, 0, 1, 1] = (1 - tz) * ty * tx              # C011
    w[:, 1, 0, 0] = tz * (1 - ty) * (1 - tx)        # C100
    w[:, 1, 0, 1] = tz * (1 - ty) * tx              # C101
    w[:, 1, 1, 0] = tz * ty * (1 - tx)              # C110
    w[:, 1, 1, 1] = tz * ty * tx                    # C111

    # make sure indices broadcast correctly
    idx_c = torch.arange(c, device=coordinates.device, dtype=torch.long)
    idx_c = einops.rearrange(idx_c, 'c -> 1 c 1 1 1')
    idx_z = einops.rearrange(idx_z, 'b z y x -> b 1 z y x')
    idx_y = einops.rearrange(idx_y, 'b z y x -> b 1 z y x')
    idx_x = einops.rearrange(idx_x, 'b z y x -> b 1 z y x')

    # insert weighted data and weight values at each corner
    data = einops.rearrange(data, 'b c -> b c 1 1 1')
    w = einops.rearrange(w, 'b z y x -> b 1 z y x')
    image.index_put_(
        indices=(idx_c, idx_z, idx_y, idx_x),
        values=data * w.to(data.dtype),
        accumulate=True
    )
    weights.index_put_(indices=(idx_z, idx_y, idx_x), values=w, accumulate=True)

    return image, weights