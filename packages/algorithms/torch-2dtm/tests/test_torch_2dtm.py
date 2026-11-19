import pytest
import torch
import torch.nn.functional as F
import torch_2dtm

from scipy.stats import special_ortho_group

from torch_2dtm.utils import normalize_template_projection


def test_template_match_dft_2d():
    # Create random test data
    # 1. Create a random image and compute its FFT
    image_size = (128, 128)
    image = torch.randn(*image_size, dtype=torch.float32)
    image_dft = torch.fft.rfftn(image, dim=(0, 1))  # Shape: (128, 65)

    # 2. Create a random 3D template and compute its FFT
    template_size = (64, 64, 64)
    template = torch.randn(*template_size, dtype=torch.float32)
    template_dft = torch.fft.rfftn(template, dim=(0, 1, 2))  # Shape: (64, 64, 33)

    # 3. Create a batch of random rotation matrices with shape (b, 3, 3)
    num_orientations = 10
    rotation_matrices = torch.tensor(
        special_ortho_group.rvs(size=num_orientations, dim=3), dtype=torch.float32
    )

    # 4. Create an arbitrary stack of Fourier space filters (identity filter here)
    # These filters operate on rffts of the 2D projection images
    # Filter shape: (..., h, w // 2 + 1)
    filters_shape = (5, 4, 3, template_size[0], template_size[1] // 2 + 1)
    filters = torch.ones(filters_shape, dtype=torch.complex64)

    # Perform template matching
    cross_correlation = torch_2dtm.match_template_dft_2d(
        image_dft=image_dft,
        template_dft=template_dft,
        rotation_matrices=rotation_matrices,
        filters=filters,
    )

    # correct output shape is (..., num_orientations, h, w)
    assert cross_correlation.shape == (5, 4, 3, num_orientations, *image_size)


def _off_centre_blob_volume(n: int) -> torch.Tensor:
    """Cubic volume with an anisotropic Gaussian blob away from the centre."""
    z, y, x = torch.meshgrid(*[torch.arange(n, dtype=torch.float64)] * 3, indexing="ij")
    c = n // 2
    blob = torch.exp(
        -(
            (z - (c + 2)) ** 2 / (2 * 2.0**2)
            + (y - (c - 5)) ** 2 / (2 * 2.0**2)
            + (x - (c + 4)) ** 2 / (2 * 3.0**2)
        )
    )
    return blob


def _circular_cross_correlation(image: torch.Tensor, kernel: torch.Tensor):
    """Direct real-space cc[k] = sum_x image[(x + k) % N] * kernel[x]."""
    h, w = kernel.shape
    padded = torch.cat([image, image[: h - 1]], dim=0)
    padded = torch.cat([padded, padded[:, : w - 1]], dim=1)
    windows = padded.unfold(0, h, 1).unfold(1, w, 1)  # (H, W, h, w)
    return torch.einsum("ijhw,hw->ij", windows, kernel)


@pytest.mark.parametrize("image_shape", [(128, 128), (128, 129), (129, 129)])
@pytest.mark.parametrize("template_size", [32, 33])
def test_match_template_dft_2d_odd_sizes(image_shape, template_size):
    h_im, w_im = image_shape
    n = template_size

    # template prepared with the documented convention
    volume = _off_centre_blob_volume(n)
    template_dft = torch.fft.fftshift(
        torch.fft.rfftn(torch.fft.ifftshift(volume.float(), dim=(-3, -2, -1))),
        dim=(-3, -2),
    )

    # identity rotation projects along z, i.e. sums over the first (z) axis
    projection = volume.sum(dim=0)  # (n, n)

    # plant the (contrast-inverted) projection at a known offset on top of noise
    offset = (37, 58)
    generator = torch.Generator().manual_seed(0)
    image = 0.05 * torch.randn(image_shape, dtype=torch.float64, generator=generator)
    image[offset[0] : offset[0] + n, offset[1] : offset[1] + n] -= projection
    image_dft = torch.fft.rfftn(image.float())

    filters = torch.ones(n, n // 2 + 1, dtype=torch.complex64)
    cross_correlation = torch_2dtm.match_template_dft_2d(
        image_dft=image_dft,
        template_dft=template_dft,
        rotation_matrices=torch.eye(3).unsqueeze(0),
        filters=filters,
        image_shape=image_shape,
    )

    assert cross_correlation.shape == (1, *image_shape)

    # cc = image * conj(template) with the template's array origin at the top-left
    # corner of the padded template box, so the peak is at the planted corner
    peak = torch.unravel_index(cross_correlation[0].argmax(), image_shape)
    assert tuple(int(i) for i in peak) == offset

    # compare against a direct real-space correlation with the same template
    # processing (zero mean, contrast flip, edge-mean / variance normalization,
    # zero mean in the padded box)
    template = -(projection - projection.mean())
    template = normalize_template_projection(template, (n, n), (h_im, w_im))
    template = F.pad(template, (0, w_im - n, 0, h_im - n))
    template = template - template.mean()
    expected = _circular_cross_correlation(image, template)

    max_abs = expected.abs().max()
    torch.testing.assert_close(
        cross_correlation[0].double() / max_abs,
        expected / max_abs,
        atol=1e-4,
        rtol=0,
    )


@pytest.mark.parametrize("image_shape", [(128, 130), (128, 127), (127, 128)])
def test_match_template_dft_2d_incompatible_image_shape(image_shape):
    image_dft = torch.zeros(128, 65, dtype=torch.complex64)
    template_dft = torch.zeros(32, 32, 17, dtype=torch.complex64)
    filters = torch.ones(32, 17, dtype=torch.complex64)
    with pytest.raises(ValueError, match="image_shape"):
        torch_2dtm.match_template_dft_2d(
            image_dft=image_dft,
            template_dft=template_dft,
            rotation_matrices=torch.eye(3).unsqueeze(0),
            filters=filters,
            image_shape=image_shape,
        )
