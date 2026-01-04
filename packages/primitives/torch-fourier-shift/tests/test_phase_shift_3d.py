import torch

from torch_fourier_shift import fourier_shift_image_3d
from torch_fourier_shift.phase_shift_grids import phase_shift_grid_3d


def test_get_phase_shifts_2d_full_fft():
    shifts = torch.zeros(size=(1, 3))
    phase_shifts = phase_shift_grid_3d(shifts, image_shape=(2, 2, 2), rfft=False)
    assert torch.allclose(phase_shifts, torch.ones(size=(2, 2, 2), dtype=torch.complex64))

    shifts = torch.tensor([[1, 2, 3]])
    phase_shifts = phase_shift_grid_3d(shifts, image_shape=(2, 2, 2), rfft=False)
    expected = torch.tensor([[[1. + 0.0000e+00j, -1. - 2.3850e-08j],
                              [1. + 1.7485e-07j, -1. - 6.7553e-07j]],
                             [[-1. - 8.7423e-08j, 1. + 3.4969e-07j],
                              [-1. - 2.3850e-08j, 1. + 4.7700e-08j]]])
    assert torch.allclose(phase_shifts, expected)


def test_get_phase_shifts_3d_rfft():
    shifts = torch.zeros(size=(1, 3))
    phase_shifts = phase_shift_grid_3d(shifts, image_shape=(2, 2, 2), rfft=True)
    assert phase_shifts.shape == (1, 2, 2, 2)
    expected = torch.ones(size=(2, 2, 2), dtype=torch.complex64)
    assert torch.allclose(phase_shifts, expected)

    shifts = torch.tensor([[1, 2, 3]])
    phase_shifts = phase_shift_grid_3d(shifts, image_shape=(2, 2, 2), rfft=True)
    expected = torch.tensor([[[1. + 0.0000e+00j, -1. + 2.3850e-08j],
                              [1. + 1.7485e-07j, -1. - 1.5100e-07j]],
                             [[-1. - 8.7423e-08j, 1. + 3.0199e-07j],
                              [-1. - 2.3850e-08j, 1. + 0.0000e+00j]]])
    assert torch.allclose(phase_shifts, expected)


def test_fourier_shift_image_3d():
    image = torch.zeros((4, 4, 4))
    image[2, 2, 2] = 1

    # +1px in each dimension
    shifts = torch.ones((1, 3))
    shifted = fourier_shift_image_3d(image, shifts)
    expected = torch.zeros((4, 4, 4))
    expected[3, 3, 3] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)

    # +1px in z
    shifts = torch.zeros((1, 3))
    shifts[0, 0] = 1
    shifted = fourier_shift_image_3d(image, shifts)
    expected = torch.zeros((4, 4, 4))
    expected[3, 2, 2] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)

    # +1px in y
    shifts = torch.zeros((1, 3))
    shifts[0, 1] = 1
    shifted = fourier_shift_image_3d(image, shifts)
    expected = torch.zeros((4, 4, 4))
    expected[2, 3, 2] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)

    # +1px in x
    shifts = torch.zeros((1, 3))
    shifts[0, 2] = 1
    shifted = fourier_shift_image_3d(image, shifts)
    expected = torch.zeros((4, 4, 4))
    expected[2, 2, 3] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)
