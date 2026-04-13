import torch

from torch_fourier_shift.fourier_shift_image import fourier_shift_image_1d
from torch_fourier_shift.phase_shift_grids import phase_shift_grid_1d


def test_get_phase_shifts_1d_full_fft():
    shifts = torch.zeros(size=(1, 2))
    phase_shifts = phase_shift_grid_1d(shifts, image_shape=(2,), rfft=False)
    assert torch.allclose(phase_shifts, torch.ones(size=(2,), dtype=torch.complex64))

    shifts = torch.tensor([1, 2])
    phase_shifts = phase_shift_grid_1d(shifts, image_shape=(2,), rfft=False)
    expected = torch.tensor(
        [[1.0 - 0.0000e00j, -1.0 - 8.7423e-08j], [1.0 - 0.0000e00j, 1.0 + 1.7485e-07j]]
    )
    assert torch.allclose(phase_shifts, expected)


def test_get_phase_shifts_1d_rfft():
    shifts = torch.zeros(size=(1, 2))
    phase_shifts = phase_shift_grid_1d(shifts, image_shape=(2,), rfft=True)
    assert phase_shifts.shape == (1, 2, 2)
    expected = torch.ones(size=(1, 2, 2), dtype=torch.complex64)
    assert torch.allclose(phase_shifts, expected)

    shifts = torch.tensor([[1, 2]])
    phase_shifts = phase_shift_grid_1d(shifts, image_shape=(2,), rfft=False)
    expected = torch.tensor(
        [
            [
                [1.0 - 0.0000e00j, -1.0 - 8.7423e-08j],
                [1.0 - 0.0000e00j, 1.0 + 1.7485e-07j],
            ]
        ]
    )
    assert torch.allclose(phase_shifts, expected)


def test_fourier_shift_image_1d():
    image = torch.zeros((4,))
    image[2] = 1

    # +1px
    shifts = torch.ones((1,))
    shifted = fourier_shift_image_1d(image, shifts)
    expected = torch.zeros((4,))
    expected[3] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)

    # -1px
    shifts = -1 * torch.ones((1,))
    shifted = fourier_shift_image_1d(image, shifts)
    expected = torch.zeros((4,))
    expected[1] = 1
    assert torch.allclose(shifted, expected, atol=1e-5)


def test_fourier_shift_preserves_dimensions_1d():
    """Test that fourier_shift_image_1d preserves input dimensions."""
    image_1d_odd = torch.randn(127)
    shifts = torch.tensor(5.0)
    shifted_1d_odd = fourier_shift_image_1d(image_1d_odd, shifts)
    assert shifted_1d_odd.shape == image_1d_odd.shape, (
        f"1D odd: Expected {image_1d_odd.shape}, got {shifted_1d_odd.shape}"
    )
