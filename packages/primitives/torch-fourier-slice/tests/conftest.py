import torch
from pytest import fixture


@fixture
def cube() -> torch.Tensor:
    volume = torch.zeros((32, 32, 32))
    volume[8:24, 8:24, 8:24] = 1
    volume[16, 16, 16] = 32
    return volume
