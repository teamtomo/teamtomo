import torch
from _utils import create_test_image, create_test_volume

from torch_find_peaks.find_peaks import find_peaks_2d, find_peaks_3d


def test_peak_picking_2d():
    # Format of peaks: [amplitude, y, x, sigma_y, sigma_x]
    peaks = torch.tensor([[1, 50, 50, 5, 5], [1, 30, 30, 2, 2]], dtype=torch.float32)
    data = create_test_image(size=100, peaks=peaks, noise_level=0.05)

    # Small distance and low threshold should pick extra peaks
    peak_detections, _ = find_peaks_2d(data, min_distance=1, threshold_abs=0.3)
    assert peak_detections.shape[0] > 2
    # Appropriate values should find exactly the two peaks
    peak_detections, heights = find_peaks_2d(data, min_distance=5, threshold_abs=0.5)
    assert peak_detections.shape[0] == 2
    # Assertions check coordinates in [y, x] order
    assert torch.allclose(peak_detections[0], torch.tensor([30, 30]), atol=1.5)
    assert torch.allclose(peak_detections[1], torch.tensor([50, 50]), atol=1.5)
    assert torch.allclose(heights[0], torch.tensor(1.0), atol=0.2)
    assert torch.allclose(heights[1], torch.tensor(1.0), atol=0.2)

def test_peak_picking_3d():
    # Format of peaks: [amplitude, z, y, x, sigma_z, sigma_y, sigma_x]
    peaks = torch.tensor([[1, 50, 50, 50, 2, 2, 2], [1, 30, 30, 30, 2, 2, 2]], dtype=torch.float32)
    data = create_test_volume(size=100, peaks=peaks, noise_level=0.05)

    # Small distance and low threshold should pick extra peaks
    peak_detections, _ = find_peaks_3d(data, min_distance=1, threshold_abs=0.1)
    assert peak_detections.shape[0] > 2
    # Appropriate values should find exactly the two peaks
    peak_detections, heights = find_peaks_3d(data, min_distance=5, threshold_abs=0.5)
    
    assert peak_detections.shape[0] == 2
    # Assertions check coordinates in [z, y, x] order
    assert torch.allclose(peak_detections[0], torch.tensor([30, 30, 30]), atol=1.5)
    assert torch.allclose(peak_detections[1], torch.tensor([50, 50, 50]), atol=1.5)
    assert torch.allclose(heights[0], torch.tensor(1.0), atol=0.2)
    assert torch.allclose(heights[1], torch.tensor(1.0), atol=0.2)
