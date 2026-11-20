import pytest
import torch

from torch_fourier_slice import extract_central_slices_rfft_3d, transform_slice_2d


@pytest.mark.parametrize("volume_rfft_shape", [(16, 16, 10), (16, 16, 8)])
def test_extract_central_slices_rfft_3d_rejects_non_cubic_volume(volume_rfft_shape):
    """The real width of a non-cubic rfft volume is ambiguous: raise."""
    volume_rfft = torch.zeros(volume_rfft_shape, dtype=torch.complex64)
    with pytest.raises(ValueError, match="square/cubic"):
        extract_central_slices_rfft_3d(
            volume_rfft=volume_rfft, rotation_matrices=torch.eye(3)
        )


@pytest.mark.parametrize("rfft_shape", [(16, 10), (17, 8)])
def test_transform_slice_2d_rejects_non_square_slice(rfft_shape):
    dfts = torch.zeros(rfft_shape, dtype=torch.complex64)
    with pytest.raises(ValueError, match="square/cubic"):
        transform_slice_2d(
            projection_image_dfts=dfts,
            rfft_shape=rfft_shape,
            stack_shape=(),
            transform_matrix=torch.eye(2),
        )
