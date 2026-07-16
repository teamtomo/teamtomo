import numpy as np
import pytest
import torch
from torch_tilt_series import TiltSeries

from torch_reconstruct_tomogram import (
    load_tilt_series_images,
    normalize_on_central_crop,
)


def test_load_tilt_series_images_uses_path_and_indices(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    raw = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    mrcfile.write(tmp_path / "stack.mrc", raw, overwrite=True)

    tilt_series = TiltSeries(
        tilt_angles=torch.tensor([0.0, 0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((2, 2)),
        image_path=tmp_path / "stack.mrc",
        image_indices=torch.tensor([3, 1]),
    )
    images = load_tilt_series_images(tilt_series)
    assert images.shape == (2, 8, 8)
    assert images.dtype == torch.float32
    assert torch.allclose(images[0], torch.as_tensor(raw[3]))
    assert torch.allclose(images[1], torch.as_tensor(raw[1]))


def test_load_tilt_series_images_without_indices_returns_full_stack(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    raw = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    mrcfile.write(tmp_path / "stack.mrc", raw, overwrite=True)

    tilt_series = TiltSeries(
        tilt_angles=torch.zeros(3),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((3, 2)),
        image_path=tmp_path / "stack.mrc",
    )
    images = load_tilt_series_images(tilt_series)
    assert torch.allclose(images, torch.as_tensor(raw))


def test_load_tilt_series_images_requires_image_path():
    tilt_series = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
    )
    with pytest.raises(ValueError):
        load_tilt_series_images(tilt_series)


def test_normalize_on_central_crop():
    images = torch.rand(2, 32, 32) * 10 + 5
    normalized = normalize_on_central_crop(images)
    crop = normalized[:, 12:20, 12:20]
    assert torch.allclose(crop.mean(dim=(-2, -1)), torch.zeros(2), atol=1e-5)
    assert torch.allclose(
        crop.std(dim=(-2, -1), correction=0), torch.ones(2), atol=1e-5
    )
