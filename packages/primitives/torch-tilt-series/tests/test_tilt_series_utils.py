import torch

from torch_tilt_series.utils import normalize_on_central_crop, subtract_plane


def test_normalize_on_central_crop():
    images = torch.rand(2, 32, 32) * 10 + 5
    normalized = normalize_on_central_crop(images)
    crop = normalized[:, 12:20, 12:20]
    assert torch.allclose(crop.mean(dim=(-2, -1)), torch.zeros(2), atol=1e-5)
    assert torch.allclose(
        crop.std(dim=(-2, -1), correction=0), torch.ones(2), atol=1e-5
    )


def test_subtract_plane_removes_known_gradient():
    h, w = 16, 16
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    plane = 0.5 * y - 0.25 * x + 3.0
    torch.manual_seed(0)
    noise = torch.randn(h, w) * 0.01
    image = plane + noise

    corrected = subtract_plane(image)
    assert corrected.shape == image.shape
    assert torch.allclose(corrected, noise, atol=1e-3)


def test_subtract_plane_is_batched_over_leading_dims():
    torch.manual_seed(0)
    images = torch.rand(3, 5, 20, 24)
    corrected = subtract_plane(images)
    assert corrected.shape == images.shape

    # batched result must match applying subtract_plane to each image alone
    for i in range(3):
        for j in range(5):
            expected = subtract_plane(images[i, j])
            assert torch.allclose(corrected[i, j], expected, atol=1e-4)


def test_subtract_plane_output_has_near_zero_slope():
    torch.manual_seed(0)
    h, w = 32, 40
    gradient = 2.0 * torch.linspace(0, 1, h)[:, None] + 1.0
    image = gradient.expand(h, w) + torch.randn(h, w)

    corrected = subtract_plane(image)
    # the fitted plane must have removed essentially all of the linear trend
    row_means = corrected.mean(dim=1)
    slope = row_means[-1] - row_means[0]
    assert slope.abs() < 0.5
