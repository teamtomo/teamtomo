import torch
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="tiler")
from torch_segment_fiducials_2d import predict_fiducial_mask


def test_predict_fiducial_mask():
    # smoke test, single image
    image = torch.rand((512, 512))
    fiducial_mask = predict_fiducial_mask(
        image, pixel_spacing=8, probability_threshold=0.5
    )
    assert fiducial_mask.shape == (512, 512)

    # smoke test, stack of images
    image = torch.rand((1, 2, 3, 512, 512))
    fiducial_mask = predict_fiducial_mask(
        image, pixel_spacing=8, probability_threshold=0.5
    )
    assert fiducial_mask.shape == (1, 2, 3, 512, 512)
