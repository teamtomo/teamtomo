import torch
from _utils import create_test_image, create_test_volume

from torch_find_peaks.refine_peaks import refine_peaks_2d, refine_peaks_3d


def test_refine_peaks_2d_basic():
    """Test basic functionality of 2D Gaussian fitting."""
    # Create test peaks with parameters [amplitude, y, x, sigma_y, sigma_x]
    peaks = torch.tensor([[4, 49, 52, 5, 5], [1, 31, 27, 2, 2]], dtype=torch.float32)
    data = create_test_image(size=100, peaks=peaks, noise_level=0.05)

    # Fit Gaussians to the peaks
    # peak_coords in [y, x] order
    fitted_params = refine_peaks_2d(
        data,
        peak_coords=torch.tensor([[45, 49],[30, 30]], dtype=torch.float32),
        boxsize=20,
        max_iterations=500,
        learning_rate=0.05,
        tolerance=1e-8,
    )
    
    # Check that we found the correct number of peaks
    assert len(fitted_params) == len(peaks)

    assert torch.allclose(fitted_params[0], peaks[0], atol=1e-1)
    assert torch.allclose(fitted_params[1], peaks[1], atol=1e-1)

def test_refine_peaks_3d_basic():
    """Test basic functionality of 3D Gaussian fitting."""
    # Create test peaks with parameters [amplitude, z, y, x, sigma_z, sigma_y, sigma_x]
    peaks = torch.tensor([[4, 47, 50, 49, 5, 5, 5], [1, 32, 30, 28, 2, 2, 2]], dtype=torch.float32)
    data = create_test_volume(size=100, peaks=peaks, noise_level=0.05)

    # Fit Gaussians to the peaks
    # peak_coords in [z, y, x] order
    fitted_params = refine_peaks_3d(
        data,
        peak_coords=torch.tensor([[45, 49, 53], [28, 32, 30]], dtype=torch.float32),
        boxsize=20,
        max_iterations=500,
        learning_rate=0.05,
        tolerance=1e-8,
    )

    # Check that we found the correct number of peaks
    assert len(fitted_params) == len(peaks)

    assert torch.allclose(fitted_params[0], peaks[0], atol=1e-1)
    assert torch.allclose(fitted_params[1], peaks[1], atol=1e-1)
