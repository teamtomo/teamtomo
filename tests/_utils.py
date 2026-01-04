import torch
from torch_grid_utils import coordinate_grid

from torch_find_peaks.gaussians import Gaussian2D, Gaussian3D


def create_test_image(size:int=100, peaks: torch.tensor = torch.tensor([]), noise_level=0.1):
    """
    Create a test image with known Gaussian peaks.
    
    Args:
        size: Size of the square image
        peaks: (n,5) tensor of peak parameters (amplitude, y, x, sigma_y, sigma_x)
        noise_level: Level of noise to add
        
    Returns:
        - Image tensor with Gaussian peaks
    """
    # Create a blank image
    image = torch.randn((size, size)) * noise_level

    gaussian_model = Gaussian2D(
        amplitude=peaks[:, 0],
        center_y=peaks[:, 1],
        center_x=peaks[:, 2],
        sigma_y=peaks[:, 3],
        sigma_x=peaks[:, 4],
    )

    grid = coordinate_grid((size,size))

    # Add Gaussian peaks to the image
    image += gaussian_model(grid).sum(dim=0)

    return image

def create_test_volume(size:int=100, peaks: torch.tensor = torch.tensor([]), noise_level=0.1):
    """
    Create a test volume with known Gaussian peaks.
    
    Args:
        size: Size of the cube volume
        peaks: (n,7) tensor of peak parameters (amplitude, z, y, x, sigma_z, sigma_y, sigma_x)
        noise_level: Level of noise to add
        
    Returns:
        - Volume tensor with Gaussian peaks
    """
    # Create a blank volume
    image = torch.randn((size, size, size)) * noise_level

    gaussian_model = Gaussian3D(
        amplitude=peaks[:, 0],
        center_z=peaks[:, 1],
        center_y=peaks[:, 2],
        center_x=peaks[:, 3],
        sigma_z=peaks[:, 4],
        sigma_y=peaks[:, 5],
        sigma_x=peaks[:, 6],
    )

    grid = coordinate_grid((size,size,size))

    # Add Gaussian peaks to the volume
    image += gaussian_model(grid).sum(dim=0)

    return image
