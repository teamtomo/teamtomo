import torch

from torch_fourier_shift import fourier_shift_image_2d
from torch_fourier_shift.phase_shift_grids import phase_shift_grid_2d


def test_get_phase_shifts_2d_full_fft():
    shifts = torch.zeros(size=(1, 2))
    phase_shifts = phase_shift_grid_2d(shifts, image_shape=(2, 2), rfft=False)
    assert torch.allclose(phase_shifts, torch.ones(size=(2, 2), dtype=torch.complex64))

    shifts = torch.tensor([[1, 2]])
    phase_shifts = phase_shift_grid_2d(shifts, image_shape=(2, 2), rfft=False)
    expected = torch.tensor(
        [[[1 + 0.0000e00j, 1 + 1.7485e-07j], [-1 - 8.7423e-08j, -1 - 2.3850e-08j]]]
    )
    assert torch.allclose(phase_shifts, expected)


def test_get_phase_shifts_2d_rfft():
    shifts = torch.zeros(size=(1, 2))
    phase_shifts = phase_shift_grid_2d(shifts, image_shape=(2, 2), rfft=True)
    assert phase_shifts.shape == (1, 2, 2)
    expected = torch.ones(size=(2, 2), dtype=torch.complex64)
    assert torch.allclose(phase_shifts, expected)

    shifts = torch.tensor([[1, 2]])
    phase_shifts = phase_shift_grid_2d(shifts, image_shape=(2, 2), rfft=False)
    expected = torch.tensor(
        [[[1 + 0.0000e00j, 1 + 1.7485e-07j], [-1 - 8.7423e-08j, -1 - 2.3850e-08j]]]
    )
    assert torch.allclose(phase_shifts, expected)


def test_fourier_shift_image_2d():
    image = torch.zeros((4, 4))
    image[2, 2] = 1

    # +1px in each dimension
    shifts = torch.ones((1, 2))
    shifted = fourier_shift_image_2d(image, shifts)
    expected = torch.zeros((4, 4))
    expected[3, 3] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)

    # +1px in y
    shifts = torch.zeros((1, 2))
    shifts[0, 0] = 1
    shifted = fourier_shift_image_2d(image, shifts)
    expected = torch.zeros((4, 4))
    expected[3, 2] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)

    # +1px in x
    shifts = torch.zeros((1, 2))
    shifts[0, 1] = 1
    shifted = fourier_shift_image_2d(image, shifts)
    expected = torch.zeros((4, 4))
    expected[2, 3] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)


def test_fourier_shift_preserves_dimensions_2d():
    """Test that fourier_shift_image_1d preserves input dimensions."""
    image_2d_odd = torch.randn((127, 127))
    shifts = torch.tensor((5.0, 5.0))
    shifted_2d_odd = fourier_shift_image_2d(image_2d_odd, shifts)
    assert shifted_2d_odd.shape == image_2d_odd.shape, (
        f"2D odd: Expected {image_2d_odd.shape}, got {shifted_2d_odd.shape}"
    )
