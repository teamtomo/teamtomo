import pytest
import warnings

from torch_segment_fiducials_2d.dataset import download_training_data


@pytest.mark.slow
def test_download_training_data(tmp_path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        download_training_data(tmp_path)
    assert (tmp_path / "images").exists()
    assert (tmp_path / "masks").exists()
