import pytest
import torch


@pytest.fixture(autouse=True)
def set_default_device():
    with torch.device("cpu"):
        yield
