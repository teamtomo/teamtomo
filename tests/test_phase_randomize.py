import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from torch_fourier_filter.phase_randomize import phase_randomize


def test_phase_randomize():
    # Test parameters
    image_shape_2d = (64, 64)
    image_shape_3d = (32, 64, 64)
    cuton = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate random 2D and 3D DFTs
    dft_2d = torch.fft.fft2(torch.rand(image_shape_2d, device=device))
    dft_3d = torch.fft.fftn(torch.rand(image_shape_3d, device=device))

    # Test 2D phase randomization preserves magnitudes
    randomized_2d = phase_randomize(dft_2d, image_shape_2d, cuton=cuton, device=device)
    assert torch.allclose(
        torch.abs(dft_2d), torch.abs(randomized_2d)
    ), "Magnitude spectrum not preserved for 2D input"

    # Test 3D phase randomization preserves magnitudes
    randomized_3d = phase_randomize(dft_3d, image_shape_3d, cuton=cuton, device=device)
    assert torch.allclose(
        torch.abs(dft_3d), torch.abs(randomized_3d)
    ), "Magnitude spectrum not preserved for 3D input"

    # Test rfft preserves magnitudes
    dft_2d_rfft = torch.fft.rfft2(torch.rand(image_shape_2d, device=device))
    randomized_2d_rfft = phase_randomize(
        dft_2d_rfft, image_shape_2d, rfft=True, cuton=cuton, device=device
    )
    assert torch.allclose(
        torch.abs(dft_2d_rfft), torch.abs(randomized_2d_rfft)
    ), "Magnitude spectrum not preserved for rfft input"

    # Ensure phases are randomized only above cuton
    freq_grid_2d = fftfreq_grid(
        image_shape_2d, rfft=False, fftshift=False, norm=True, device=device
    )
    freq_mask_2d = freq_grid_2d > cuton
    original_phases_2d = torch.angle(dft_2d)
    randomized_phases_2d = torch.angle(randomized_2d)
    assert torch.allclose(
        original_phases_2d[~freq_mask_2d], randomized_phases_2d[~freq_mask_2d]
    ), "Phases not preserved below cuton for 2D input"
    assert not torch.allclose(
        original_phases_2d[freq_mask_2d], randomized_phases_2d[freq_mask_2d]
    ), "Phases not randomized above cuton for 2D input"

    freq_grid_3d = fftfreq_grid(
        image_shape_3d, rfft=False, fftshift=False, norm=True, device=device
    )
    freq_mask_3d = freq_grid_3d > cuton
    original_phases_3d = torch.angle(dft_3d)
    randomized_phases_3d = torch.angle(randomized_3d)
    assert torch.allclose(
        original_phases_3d[~freq_mask_3d], randomized_phases_3d[~freq_mask_3d]
    ), "Phases not preserved below cuton for 3D input"
    assert not torch.allclose(
        original_phases_3d[freq_mask_3d], randomized_phases_3d[freq_mask_3d]
    ), "Phases not randomized above cuton for 3D input"
