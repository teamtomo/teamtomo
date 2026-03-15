import pytest
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning, module="tiler")
from torch_segment_fiducials_2d.model import ResidualUNet18, get_latest_checkpoint


@pytest.mark.slow
def test_download_and_load_latest_checkpoint():
    checkpoint_file = get_latest_checkpoint()
    model = ResidualUNet18.load_from_checkpoint(checkpoint_file, map_location="cpu")
    assert isinstance(model, ResidualUNet18)
