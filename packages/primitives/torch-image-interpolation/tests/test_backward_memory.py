"""Regression tests for the memory cost of the backward pass.

Sampling n points used to broadcast the image to `(n, c, *image_shape)` before
calling `grid_sample`. That is free in the forward pass (a stride-0 view) but
makes the backward pass materialise a dense `(n, c, *image_shape)` input
gradient before the broadcast sums it away, so peak memory and runtime scale
with `n * image_elems` instead of with `n`.

See https://github.com/teamtomo/teamtomo/issues/117
"""

import pytest
import torch
import torch.nn.functional as F

from torch_image_interpolation import (
    sample_image_1d,
    sample_image_2d,
    sample_image_3d,
)


@pytest.fixture
def grid_sample_batch_sizes(monkeypatch):
    """Record the batch size of every `grid_sample` input during the test."""
    seen = []
    real = F.grid_sample

    def spy(input, grid, **kwargs):  # noqa: A002
        seen.append(input.shape[0])
        return real(input, grid, **kwargs)

    monkeypatch.setattr(F, "grid_sample", spy)
    return seen


@pytest.mark.parametrize("n_samples", [1, 64, 4096])
@pytest.mark.parametrize(
    "sample_fn, image_shape, coordinate_ndim",
    [
        (sample_image_1d, (32,), 0),
        (sample_image_2d, (32, 32), 2),
        (sample_image_3d, (32, 32, 32), 3),
    ],
)
def test_image_is_not_broadcast_over_samples(
    sample_fn, image_shape, coordinate_ndim, n_samples, grid_sample_batch_sizes
):
    """The image must be passed to `grid_sample` once, not once per sample."""
    image = torch.rand(image_shape)
    coordinate_shape = (n_samples,) + ((coordinate_ndim,) if coordinate_ndim else ())
    coordinates = torch.rand(coordinate_shape) * (image_shape[-1] - 1)

    sample_fn(image=image, coordinates=coordinates)

    # a batch size of n_samples here is the bug: it costs n_samples x image
    # in the backward pass
    assert grid_sample_batch_sizes == [1]


@pytest.mark.parametrize(
    "sample_fn, image_shape, coordinate_ndim",
    [
        (sample_image_1d, (16,), 0),
        (sample_image_2d, (16, 16), 2),
        (sample_image_3d, (16, 16, 16), 3),
    ],
)
def test_gradients_flow_to_image_and_coordinates(
    sample_fn, image_shape, coordinate_ndim
):
    """Sampling is differentiable with respect to both image and coordinates."""
    image = torch.rand(image_shape, dtype=torch.float64, requires_grad=True)
    coordinate_shape = (7,) + ((coordinate_ndim,) if coordinate_ndim else ())
    # stay off exact voxel centres: the interpolant is only piecewise linear,
    # so its derivative is undefined on the grid lines
    coordinates = (
        torch.rand(coordinate_shape, dtype=torch.float64) * (image_shape[-1] - 3) + 1.1
    )
    coordinates.requires_grad_(True)

    torch.autograd.gradcheck(
        lambda i, c: sample_fn(image=i, coordinates=c),
        (image, coordinates),
        eps=1e-6,
        atol=1e-8,
    )
