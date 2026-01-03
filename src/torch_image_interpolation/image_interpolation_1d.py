from typing import Literal

import einops
import torch
import torch.nn.functional as F

from torch_image_interpolation import utils
from torch_image_interpolation.grid_sample_utils import array_to_grid_sample


def sample_image_1d(
        image: torch.Tensor,
        coordinates: torch.Tensor,
        interpolation: Literal['nearest', 'linear', 'cubic'] = 'linear',
) -> torch.Tensor:
    """Sample a 1D image (vector) with a specific interpolation mode.

    Parameters
    ----------
    image: torch.Tensor
        `(w, )` image or `(c, w)` multichannel image.
    coordinates: torch.Tensor
        `(..., )` array of coordinates at which `image` should be sampled.
        - Coordinates are positions in the `w` dimension.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    interpolation: Literal['nearest', 'linear', 'cubic']
        Interpolation mode for image sampling.

    Returns
    -------
    samples: torch.Tensor
        `(..., )` or `(..., c)` array of samples from `image`.
    """
    if image.ndim not in (1, 2):
        raise ValueError(f'image should have shape (w, ) or (c, w), got {image.shape}')

    # keep track of a few image properties
    input_image_is_complex = torch.is_complex(image)
    input_image_is_multichannel = image.ndim == 2

    # coerce single channel to multi-channel
    if input_image_is_multichannel is False:
        image = einops.rearrange(image, 'w -> 1 w')  # (c, w)

    # set up for sampling with torch.nn.functional.grid_sample
    # coordinates: (..., ) -> (n, )
    coordinates, ps = einops.pack([coordinates], pattern='*')
    n_samples = coordinates.shape[0]
    w = image.shape[-1]

    # handle complex input
    if input_image_is_complex:
        # cannot sample complex tensors directly with grid_sample
        # c.f. https://github.com/pytorch/pytorch/issues/67634
        # workaround: treat real and imaginary parts as separate channels
        image = torch.view_as_real(image)
        image = einops.rearrange(image, 'c w complex -> (complex c) w')

    # For grid_sample to work with 1D, we need to add a dummy height dimension
    # and treat it as a 2D image with height=2
    image = einops.repeat(image, 'c w -> c h w', h=2)

    # Create batch of images, one per sample
    image = einops.repeat(image, 'c h w -> b c h w', b=n_samples)

    # Create grid coordinates
    # We need to add a dummy y coordinate (set to 0) to make it work with grid_sample
    dummy_y = torch.zeros_like(coordinates)
    coords_2d = einops.rearrange([dummy_y, coordinates], "yx b -> b 1 1 yx")  # (b, 1, 1, 2)

    # Take the samples
    # We need to convert the coordinates to grid_sample format
    coords_2d = array_to_grid_sample(coords_2d, array_shape=(2, w))
    if interpolation == 'linear':
        mode = 'bilinear'
    elif interpolation == 'cubic':
        mode = 'bicubic'
    else:
        mode = interpolation
    samples = F.grid_sample(
        input=image,
        grid=coords_2d,
        mode=mode,
        padding_mode='border',  # this increases sampling fidelity at edges
        align_corners=True,
    )

    # Reconstruct complex valued samples if required
    if input_image_is_complex is True:
        samples = einops.rearrange(samples, 'b (complex c) 1 1 -> b c complex', complex=2)
        samples = utils.view_as_complex(samples.contiguous())  # (b, c)
    else:
        samples = einops.rearrange(samples, 'b c 1 1 -> b c')

    # Set samples from outside of image to zero explicitly
    inside = torch.logical_and(coordinates >= 0, coordinates <= w - 1)
    samples[~inside] *= 0

    # Pack samples back into the expected shape
    # (b, c) -> (..., c)
    [samples] = einops.unpack(samples, pattern='* c', packed_shapes=ps)

    # Ensure output has correct shape
    # - (...) if input image was single channel
    # - (..., c) if input image was multi-channel
    if input_image_is_multichannel is False:
        samples = einops.rearrange(samples, '... 1 -> ...')  # drop channel dim

    return samples


def insert_into_image_1d(
        values: torch.Tensor,
        coordinates: torch.Tensor,
        image: torch.Tensor,
        weights: torch.Tensor | None = None,
        interpolation: Literal['nearest', 'linear'] = 'linear',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert values into a 1D image with linear interpolation.

    Parameters
    ----------
    values: torch.Tensor
        `(...)` or `(..., c)` array of values to be inserted into `image`.
    coordinates: torch.Tensor
        `(..., 1)` array of 1D coordinates for each value in `data`.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    image: torch.Tensor
        `(w)` or `(c, w)` array containing the image into which data will be inserted.
    weights: torch.Tensor | None
        `(w)` array containing weights associated with each pixel in `image`.
        This is useful for tracking weights across multiple calls to this function.
    interpolation: Literal['nearest', 'linear']
        Interpolation mode used for adding data points to grid.

    Returns
    -------
    image, weights: tuple[torch.Tensor, torch.Tensor]
        The image and weights after updating with data from `data` at `coordinates`.
    """
    # keep track of a few properties of the inputs
    input_image_is_multichannel = image.ndim == 2
    w = image.shape[-1]

    # validate inputs
    values_shape = values.shape[:-1] if input_image_is_multichannel else values.shape

    if values_shape != coordinates.shape:
        raise ValueError('One coordinate is required for each value in data.')
    if image.dtype != values.dtype:
        raise ValueError('Image and values must have the same dtype.')

    if weights is None:
        weights = torch.zeros(size=(w,), dtype=torch.float32, device=image.device)

    # add channel dim to both image and values if input image is not multichannel
    if not input_image_is_multichannel:
        image = einops.rearrange(image, 'w -> 1 w')
        values = einops.rearrange(values, '... -> ... 1')

    # linearise data and coordinates
    values, _ = einops.pack([values], pattern='* c')  # (b, c)
    coordinates, _ = einops.pack([coordinates], pattern='*')  # (b, )
    coordinates = coordinates.float()

    # only keep data and coordinates inside the image
    idx_inside = (coordinates >= 0) & (coordinates <= w - 1)
    values, coordinates = torch.atleast_2d(values[idx_inside]), coordinates[idx_inside]

    # splat data onto grid
    if interpolation == 'nearest':
        image, weights = _insert_nearest_1d(values, coordinates, image, weights)
    if interpolation == 'linear':
        image, weights = _insert_linear_1d(values, coordinates, image, weights)

    # ensure correct output image shape
    # single channel input -> (w)
    # multichannel input -> (c, w)
    if not input_image_is_multichannel:
        image = einops.rearrange(image, '1 w -> w')

    return image, weights


def _insert_nearest_1d(
        data,  # (b, c)
        coordinates,  # (b,)
        image,  # (c, w)
        weights  # (w)
):
    # b is number of data points to insert per channel, c is number of channels
    b, c = data.shape

    # find nearest voxel for each coordinate
    coordinates = torch.round(coordinates).long()

    # insert ones into weights image (w) at each position
    w = torch.ones(size=(b, 1), device=weights.device, dtype=weights.dtype)

    # setup indices for insertion
    idx_c = torch.arange(c, device=coordinates.device, dtype=torch.long)
    idx_c = einops.rearrange(idx_c, 'c -> 1 c')
    idx_x = einops.rearrange(coordinates, 'b -> b 1')

    # insert image data and weights
    image.index_put_(indices=(idx_c, idx_x), values=data, accumulate=True)
    weights.index_put_(indices=(idx_x,), values=w, accumulate=True)
    return image, weights


def _insert_linear_1d(
        data,  # (b, c)
        coordinates,  # (b,)
        image,  # (c, w)
        weights  # (w)
):
    # b is number of data points to insert per channel, c is number of channels
    b, c = data.shape

    # cache corner coordinates for each value to be inserted
    # C0---P---C1
    x0 = torch.floor(coordinates).long()
    x1 = torch.ceil(coordinates).long()

    # calculate linear interpolation weights for each corner
    tx = coordinates - x0.float()  # fractional position between corners
    w = torch.empty(size=(b, 2), device=image.device, dtype=weights.dtype)
    w[:, 0] = 1 - tx  # C0
    w[:, 1] = tx  # C1

    # Set up indices for both corners
    idx_x = torch.stack([x0, x1], dim=1)  # (b, 2)

    # make sure indices broadcast correctly
    idx_c = torch.arange(c, device=coordinates.device, dtype=torch.long)
    idx_c = einops.rearrange(idx_c, 'c -> 1 c 1')
    idx_x = einops.rearrange(idx_x, 'b x -> b 1 x')

    # insert weighted data and weight values at each corner across all channels
    data = einops.rearrange(data, 'b c -> b c 1')
    w = einops.rearrange(w, 'b x -> b 1 x')
    image.index_put_(
        indices=(idx_c, idx_x),
        values=data * w.to(data.dtype),
        accumulate=True
    )
    weights.index_put_(indices=(idx_x,), values=w, accumulate=True)

    return image, weights
