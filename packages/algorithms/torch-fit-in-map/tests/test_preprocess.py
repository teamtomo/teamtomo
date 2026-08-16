"""Tests for volume preprocessing utilities."""

import torch


def test_crop_or_pad_larger():
    """crop_or_pad_to_shape should center-crop a larger volume."""
    from torch_fit_in_map import crop_or_pad_to_shape

    vol = torch.ones(30, 30, 30)
    out = crop_or_pad_to_shape(vol, (20, 20, 20))
    assert out.shape == (20, 20, 20)


def test_crop_or_pad_smaller():
    """crop_or_pad_to_shape should zero-pad a smaller volume."""
    from torch_fit_in_map import crop_or_pad_to_shape

    vol = torch.ones(10, 10, 10)
    out = crop_or_pad_to_shape(vol, (20, 20, 20))
    assert out.shape == (20, 20, 20)
    # Corners (padding region) should be zero; centre of original data should be one
    assert out[0, 0, 0].item() == 0.0
    assert out[10, 10, 10].item() == 1.0  # data placed at [5:15,5:15,5:15]


def test_crop_or_pad_non_cubic():
    """crop_or_pad_to_shape should handle non-cubic targets."""
    from torch_fit_in_map import crop_or_pad_to_shape

    vol = torch.ones(40, 20, 10)
    out = crop_or_pad_to_shape(vol, (30, 25, 15))
    assert out.shape == (30, 25, 15)
