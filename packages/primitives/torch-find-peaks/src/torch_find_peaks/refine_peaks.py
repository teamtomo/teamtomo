from typing import Union, Any, Literal

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_grid_utils import coordinate_grid
from torch_grid_utils.fftfreq_grid import dft_center
from torch_subpixel_crop import subpixel_crop_2d, subpixel_crop_3d

from .gaussians import Gaussian2D, Gaussian3D


def _refine_peaks_2d_torch(
    image: torch.Tensor,
    peak_data: torch.Tensor,
    boxsize: int,
    max_iterations: int,
    learning_rate: float,
    tolerance: float,
) -> torch.Tensor:
    """
    Internal function to refine the positions of peaks in a 2D tensor.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 5) containing the fitted parameters for each peak.
        Each row contains [amplitude, y, x, sigma_x, sigma_y].
    """
    
    # Crop regions around peaks
    boxes = subpixel_crop_2d(image, peak_data[...,1:3], boxsize).detach()
    # Prepare coordinates
    center = dft_center((boxsize, boxsize), rfft=False, fftshift=True)
    grid = coordinate_grid((boxsize, boxsize), center=center, device=image.device)

    # Initialize model
    model = Gaussian2D(amplitude=peak_data[..., 0],
                       center_x=torch.zeros_like(peak_data[..., 0]),
                       center_y=torch.zeros_like(peak_data[..., 0]),
                       sigma_x=peak_data[..., 3],
                       sigma_y=peak_data[..., 4]).to(image.device)

    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Fit the Gaussians
    for _ in range(max_iterations):
        optimizer.zero_grad()

        # Calculate predicted values
        output = model(grid)
        # Calculate loss
        loss = criterion(output, boxes)
        # Check convergence
        if loss.item() < tolerance:
            break

        # Backpropagate and update
        loss.backward(retain_graph=False)  # Ensure no graph retention
        optimizer.step()

        # Ensure positive values for amplitude and sigma
        with torch.no_grad():
            model.amplitude.data.clamp_(min=0)
            model.sigma_x.data.clamp_(min=0.001)
            model.sigma_y.data.clamp_(min=0.001)

    # Combine the (...,1) model parameters to a (...,5) tensor
    # and add the peak coordinates - keeping yx order
    fitted_params = torch.stack([
        model.amplitude,
        model.center_y + peak_data[..., 1],  # y coordinate first
        model.center_x + peak_data[..., 2],  # x coordinate second
        model.sigma_x,
        model.sigma_y
    ], dim=-1)

    return fitted_params


def refine_peaks_2d(
    image: Any,
    peak_coords: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    boxsize: int,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    amplitude: Union[torch.Tensor, float] = 1.,
    sigma_x: Union[torch.Tensor, float] = 1.,
    sigma_y: Union[torch.Tensor, float] = 1.,
    return_as: Literal["torch", "numpy", "dataframe"] = "torch",
) -> torch.Tensor:
    """
    Refine the positions of peaks in a 2D image by fitting 2D Gaussian functions.

    Parameters
    ----------
    image : Any
        A 2D tensor-like object (e.g., torch.Tensor, numpy.ndarray)
        containing the image data.
    peak_coords : torch.Tensor, np.ndarray, or pd.DataFrame
        A tensor-like object of shape (n, 2) containing the initial peak coordinates (y, x).
    boxsize : int
        Size of the region to crop around each peak (must be even).
    max_iterations : int, optional
        Maximum number of optimization iterations. Default is 1000.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.01.
    tolerance : float, optional
        Convergence tolerance for the optimization. Default is 1e-6.
    amplitude : Union[torch.Tensor, float], optional
        Initial amplitude of the Gaussian. Default is 1.0.
    sigma_x : Union[torch.Tensor, float], optional
        Initial standard deviation in the x direction. Default is 1.0.
    sigma_y : Union[torch.Tensor, float], optional
        Initial standard deviation in the y direction. Default is 1.0.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 5) containing the fitted parameters for each peak.
        Each row contains [amplitude, y, x, sigma_x, sigma_y].
    """
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    if isinstance(peak_coords, pd.DataFrame):
        amplitude = torch.as_tensor(peak_coords["height"].to_numpy())
        peak_coords = torch.as_tensor(peak_coords[["y","x"]].to_numpy())
    if not isinstance(peak_coords, torch.Tensor):
        peak_coords = torch.as_tensor(peak_coords)

    num_peaks = peak_coords.shape[0]
    if not isinstance(amplitude, torch.Tensor):
        amplitude = torch.tensor([amplitude] * num_peaks, device=image.device)
    if not isinstance(sigma_x, torch.Tensor):
        sigma_x = torch.tensor([sigma_x] * num_peaks, device=image.device)
    if not isinstance(sigma_y, torch.Tensor):
        sigma_y = torch.tensor([sigma_y] * num_peaks, device=image.device)

    initial_peak_data = torch.stack([
        amplitude,
        peak_coords[:, 0],  # y
        peak_coords[:, 1],  # x
        sigma_x,
        sigma_y,
    ], dim=-1)

    refined_peak_data = _refine_peaks_2d_torch(
        image=image,
        peak_data=initial_peak_data,
        boxsize=boxsize,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
    )

    if return_as=="torch":
        return refined_peak_data
    elif return_as=="numpy":
        return refined_peak_data.detach().cpu().numpy()
    elif return_as=="dataframe":
        return pd.DataFrame(refined_peak_data.detach().cpu().numpy(), columns=["amplitude", "y", "x", "sigma_x", "sigma_y"])
    else:
        raise ValueError(f"Invalid return_as value: {return_as}")

def _refine_peaks_3d_torch(
    volume: torch.Tensor,
    peak_data: torch.Tensor,
    boxsize: int,
    max_iterations: int,
    learning_rate: float,
    tolerance: float,
) -> torch.Tensor:
    """
    Internal function to refine the positions of peaks in a 3D tensor.

    Parameters
    ----------
    volume : torch.Tensor
        A 3D tensor containing the volume data.
    peak_data : torch.Tensor
        A tensor of shape (n, 7) containing the initial peak parameters.
        Each row contains [amplitude, z, y, x, sigma_x, sigma_y, sigma_z].
    boxsize : int
        Size of the region to crop around each peak (must be even).
    max_iterations : int
        Maximum number of optimization iterations.
    learning_rate : float
        Learning rate for the optimizer.
    tolerance : float
        Convergence tolerance for the optimization.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 7) containing the refined parameters for each peak.
        Each row contains [amplitude, z, y, x, sigma_x, sigma_y, sigma_z].
    """
    # Ensure boxsize is even
    if boxsize % 2 != 0:
        raise ValueError("boxsize must be even")

    # Crop regions around peaks
    boxes = subpixel_crop_3d(volume, peak_data[:, 1:4], boxsize).detach()

    # Prepare coordinates
    center = dft_center((boxsize, boxsize, boxsize), rfft=False, fftshift=True)
    grid = coordinate_grid((boxsize, boxsize, boxsize), center=center, device=volume.device)

    # Initialize model
    model = Gaussian3D(
        amplitude=peak_data[:, 0],
        center_x=torch.zeros_like(peak_data[:, 0]),
        center_y=torch.zeros_like(peak_data[:, 0]),
        center_z=torch.zeros_like(peak_data[:, 0]),
        sigma_x=peak_data[:, 4],
        sigma_y=peak_data[:, 5],
        sigma_z=peak_data[:, 6],
    ).to(volume.device)

    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Fit the Gaussians
    for _ in range(max_iterations):
        optimizer.zero_grad()

        # Calculate predicted values
        output = model(grid)
        # Calculate loss
        loss = criterion(output, boxes)
        # Check convergence
        if loss.item() < tolerance:
            break

        # Backpropagate and update
        loss.backward(retain_graph=False)  # Ensure no graph retention
        optimizer.step()

        # Ensure positive values for amplitude and sigma
        with torch.no_grad():
            model.amplitude.data.clamp_(min=0)
            model.sigma_x.data.clamp_(min=0.001)
            model.sigma_y.data.clamp_(min=0.001)
            model.sigma_z.data.clamp_(min=0.001)

    # Combine the (...,1) model parameters to a (...,7) tensor
    # and add the peak coordinates in zyx order
    fitted_params = torch.stack([
        model.amplitude,
        model.center_z + peak_data[:, 1],  # z coordinate first
        model.center_y + peak_data[:, 2],  # y coordinate second
        model.center_x + peak_data[:, 3],  # x coordinate third
        model.sigma_x,
        model.sigma_y,
        model.sigma_z
    ], dim=-1)

    return fitted_params, boxes, output


def refine_peaks_3d(
    volume: Any,
    peak_coords: Union[torch.Tensor, np.ndarray, pd.DataFrame],
    boxsize: int,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    amplitude: Union[torch.Tensor, float] = 1.,
    sigma_x: Union[torch.Tensor, float] = 1.,
    sigma_y: Union[torch.Tensor, float] = 1.,
    sigma_z: Union[torch.Tensor, float] = 1.,
    return_as: Literal["torch", "numpy", "dataframe"] = "torch",
) -> torch.Tensor:
    """
    Refine the positions of peaks in a 3D volume by fitting 3D Gaussian functions.

    Parameters
    ----------
    volume : Any
        A 3D tensor-like object (e.g., torch.Tensor, numpy.ndarray)
        containing the volume data.
    peak_coords : torch.Tensor, np.ndarray, or pd.DataFrame
        A tensor-like object of shape (n, 3) containing the initial peak coordinates (z, y, x).
    boxsize : int
        Size of the region to crop around each peak (must be even).
    max_iterations : int, optional
        Maximum number of optimization iterations. Default is 1000.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.01.
    tolerance : float, optional
        Convergence tolerance for the optimization. Default is 1e-6.
    amplitude : Union[torch.Tensor, float], optional
        Initial amplitude of the Gaussian. Default is 1.0.
    sigma_x : Union[torch.Tensor, float], optional
        Initial standard deviation in the x direction. Default is 1.0.
    sigma_y : Union[torch.Tensor, float], optional
        Initial standard deviation in the y direction. Default is 1.0.
    sigma_z : Union[torch.Tensor, float], optional
        Initial standard deviation in the z direction. Default is 1.0.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 7) containing the fitted parameters for each peak.
        Each row contains [amplitude, z, y, x, sigma_x, sigma_y, sigma_z].
    """
    if not isinstance(volume, torch.Tensor):
        volume = torch.as_tensor(volume)
    if isinstance(peak_coords, pd.DataFrame):
        amplitude = torch.as_tensor(peak_coords["height"].to_numpy(),device=volume.device)
        peak_coords = torch.as_tensor(peak_coords[["z", "y", "x"]].to_numpy(),device=volume.device)
    if not isinstance(peak_coords, torch.Tensor):
        peak_coords = torch.as_tensor(peak_coords)

    num_peaks = peak_coords.shape[0]
    if not isinstance(amplitude, torch.Tensor):
        amplitude = torch.tensor([amplitude] * num_peaks, device=volume.device)
    if not isinstance(sigma_x, torch.Tensor):
        sigma_x = torch.tensor([sigma_x] * num_peaks, device=volume.device)
    if not isinstance(sigma_y, torch.Tensor):
        sigma_y = torch.tensor([sigma_y] * num_peaks, device=volume.device)
    if not isinstance(sigma_z, torch.Tensor):
        sigma_z = torch.tensor([sigma_z] * num_peaks, device=volume.device)

    initial_peak_data = torch.stack([
        amplitude,
        peak_coords[:, 0],  # z
        peak_coords[:, 1],  # y
        peak_coords[:, 2],  # x
        sigma_x,
        sigma_y,
        sigma_z,
    ], dim=-1)

    refined_peak_data, boxes, output = _refine_peaks_3d_torch(
        volume=volume,
        peak_data=initial_peak_data,
        boxsize=boxsize,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
    )

    if return_as == "torch":
        return refined_peak_data
    elif return_as == "numpy":
        return refined_peak_data.detach().cpu().numpy()
    elif return_as == "dataframe":
        return pd.DataFrame(refined_peak_data.detach().cpu().numpy(), columns=["amplitude", "z", "y", "x", "sigma_x", "sigma_y", "sigma_z"])
    elif return_as == "diagnostic":
        # Return the boxes and output for diagnostic purposes
        return {
            "refined_peaks": pd.DataFrame(refined_peak_data.detach().cpu().numpy(), columns=["amplitude", "z", "y", "x", "sigma_x", "sigma_y", "sigma_z"]),
            "boxes": boxes,
            "output": output
        }
    else:
        raise ValueError(f"Invalid return_as value: {return_as}")
