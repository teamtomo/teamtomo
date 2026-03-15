import torch
import pytest
from torch_tiltxcorr.utils import (
    calculate_cross_correlation,
    get_shift_from_correlation_image,
)


def test_shift_detection():
    a = torch.zeros((24, 24))
    a[13, 13] = 1
    b = torch.zeros((24, 24))
    b[12, 12] = 1

    ccc = calculate_cross_correlation(a, a)
    shift = get_shift_from_correlation_image(ccc)
    assert shift.tolist() == [0.0, 0.0]

    ccc = calculate_cross_correlation(a, b)
    dy, dx = get_shift_from_correlation_image(ccc)
    assert dy == pytest.approx(1.0, 1e-3)
    assert dx == pytest.approx(1.0, 1e-3)

    b = torch.zeros((24, 24))
    b[12, 12] = 2.5
    b[11, 12] = 2.5

    ccc = calculate_cross_correlation(a, b)
    dy, dx = get_shift_from_correlation_image(ccc)
    assert dy == pytest.approx(1.5, 1e-2)
    assert dx == pytest.approx(1.0, 1e-2)
