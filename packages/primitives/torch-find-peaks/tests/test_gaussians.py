import torch
import pytest
from torch_find_peaks.gaussians import Gaussian2D, Gaussian3D

from torch_grid_utils.fftfreq_grid import dft_center
from torch_grid_utils.coordinate_grid import coordinate_grid

def test_gaussian_2d_basic():
    gaus = Gaussian2D()

    center = dft_center(
        image_shape=(20, 20),
        rfft=False,
        fftshift=True
    )
    grid = coordinate_grid(
        image_shape=(20, 20),
        center=center,
    )

    forward = gaus(grid)
    assert forward.shape == (20, 20)
    assert forward[tuple(center)] == torch.max(forward)
    assert forward[0,0] == torch.min(forward)


@pytest.mark.parametrize(
    "shape",
    [
        (20),
        (5,5),
        (3,2,1,2)       
    ]
)
def test_gaussian_2d_batching(shape):

    gaus = Gaussian2D(
        amplitude=torch.ones(shape),
        center_y=torch.zeros(shape),
        center_x=torch.zeros(shape),
        sigma_y=torch.ones(shape),
        sigma_x=torch.ones(shape),
    )

    center = dft_center(
        image_shape=(20, 20),
        rfft=False,
        fftshift=True
    )
    grid = coordinate_grid(
        image_shape=(20, 20),
        center=center,
    )

    forward = gaus(grid)

    if isinstance(shape, int):
        shape = (shape,)
    assert forward.shape == shape + (20, 20)


def test_gaussian_3d_basic():

    gaus = Gaussian3D()

    center = dft_center(
        image_shape=(10, 10, 10),
        rfft=False,
        fftshift=True
    )
    grid = coordinate_grid(
        image_shape=(10, 10, 10),
        center=center,
    )

    forward = gaus(grid)
    assert forward.shape == (10, 10, 10)
    assert forward[tuple(center)] == torch.max(forward)
    assert forward[0, 0, 0] == torch.min(forward)


@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (4, 4),
        (2, 3, 2, 3)
    ]
)
def test_gaussian_3d_batching(shape):

    gaus = Gaussian3D(
        amplitude=torch.ones(shape),
        center_z=torch.zeros(shape),
        center_y=torch.zeros(shape),
        center_x=torch.zeros(shape),
        sigma_z=torch.ones(shape),
        sigma_y=torch.ones(shape),
        sigma_x=torch.ones(shape),
    )

    center = dft_center(
        image_shape=(10, 10, 10),
        rfft=False,
        fftshift=True
    )
    grid = coordinate_grid(
        image_shape=(10, 10, 10),
        center=center,
    )

    forward = gaus(grid)
    
    if isinstance(shape, int):
        shape = (shape,)
    assert forward.shape == shape + (10, 10, 10)

