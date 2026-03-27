import torch
import torch_2dtm

from scipy.stats import special_ortho_group


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

    # 4. Create an arbitrary stack of Fourier space filters (identity filter in this example)
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
