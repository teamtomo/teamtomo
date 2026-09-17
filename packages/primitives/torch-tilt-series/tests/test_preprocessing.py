import torch

from torch_tilt_series.preprocessing import preprocess_tilt_series_images


def test_preprocess_tilt_series_images_shape_preserved():
    torch.manual_seed(0)
    images = torch.rand(3, 64, 64) * 10 + 5
    result = preprocess_tilt_series_images(images, low=0.05, bandpass_padding=16)
    assert result.shape == images.shape


def test_preprocess_tilt_series_images_bandpass_padding_is_clamped_to_image_size():
    torch.manual_seed(0)
    images = torch.rand(2, 32, 32) * 10 + 5
    # bandpass_padding (128, the default) exceeds the image size -> would crash
    # torch.nn.functional.pad's reflect mode unless silently clamped
    result = preprocess_tilt_series_images(images, low=0.05)
    assert result.shape == images.shape


def test_preprocess_tilt_series_images_normalizes_by_default():
    torch.manual_seed(0)
    images = torch.rand(2, 64, 64) * 10 + 5
    result = preprocess_tilt_series_images(images, low=0.05, bandpass_padding=16)
    h, w = result.shape[-2:]
    crop = result[..., int(0.375 * h) : int(0.625 * h), int(0.375 * w) : int(0.625 * w)]
    assert torch.allclose(crop.mean(dim=(-2, -1)), torch.zeros(2), atol=1e-4)
    assert torch.allclose(
        crop.std(dim=(-2, -1), correction=0), torch.ones(2), atol=1e-4
    )


def test_preprocess_tilt_series_images_normalize_false_skips_normalization():
    torch.manual_seed(0)
    images = torch.rand(2, 64, 64) * 10 + 5
    result = preprocess_tilt_series_images(
        images, low=0.0, bandpass_padding=16, normalize=False
    )
    assert not torch.allclose(result.std(dim=(-2, -1)), torch.ones(2), atol=1e-2)


def test_preprocess_tilt_series_images_high_pass_removes_dc():
    torch.manual_seed(0)
    images = torch.rand(1, 64, 64) + 100.0  # large constant offset
    result = preprocess_tilt_series_images(
        images,
        low=0.05,
        bandpass_padding=16,
        subtract_background=False,
        normalize=False,
    )
    # a high-pass filter must remove the large constant (DC) offset
    assert result.abs().mean() < images.abs().mean()


def test_preprocess_tilt_series_images_removes_linear_gradient():
    h, w = 64, 64
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    gradient = (0.5 * y + 0.25 * x)[None]
    torch.manual_seed(0)
    images = gradient + torch.randn(1, h, w) * 0.05

    result = preprocess_tilt_series_images(
        images, low=0.0, bandpass_padding=16, subtract_background=True, normalize=False
    )
    result_no_bg = preprocess_tilt_series_images(
        images, low=0.0, bandpass_padding=16, subtract_background=False, normalize=False
    )
    # subtracting the background plane before filtering must leave less
    # residual low-order structure than skipping it
    assert result.std() < result_no_bg.std()
