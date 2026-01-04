import pytest
import torch


@pytest.fixture(autouse=True)
def set_default_device():
    torch.set_default_device("cpu")
