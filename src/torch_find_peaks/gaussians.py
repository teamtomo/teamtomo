from typing import Union

import einops
import torch
import torch.nn as nn


class WarpedGaussian2D(nn.Module):
    """
    A 2D warped Gaussian function.

    Parameters
    ----------
    amplitude : torch.tensor, optional
        Amplitude of the Gaussian. Default is torch.tensor([1.0]).
    center_y : torch.tensor, optional
        Y-coordinate of the center. Default is torch.tensor([0.0]).
    center_x : torch.tensor, optional
        X-coordinate of the center. Default is torch.tensor([0.0]).
    sigma_y : torch.tensor, optional
        Standard deviation along the y-axis. Default is torch.tensor([1.0]).
    sigma_x : torch.tensor, optional
        Standard deviation along the x-axis. Default is torch.tensor([1.0]).
    warp : torch.tensor, optional
        Warp factor for the Gaussian. Default is torch.tensor([1.0]).
    warp_angle : torch.tensor, optional
        Angle of the warp in radians. Default is torch.tensor([0.0]).

    Methods
    -------
    forward(grid)
        Compute the warped Gaussian values for a given 2D grid.
        Expects grid in yx order (grid[..., 0] is y, grid[..., 1] is x).
    """

    def __init__(self,
                 amplitude: torch.tensor = torch.tensor([1.0]),
                 center_y: torch.tensor = torch.tensor([0.0]),
                 center_x: torch.tensor = torch.tensor([0.0]),
                 sigma_y: torch.tensor = torch.tensor([1.0]),
                 sigma_x: torch.tensor = torch.tensor([1.0]),
                 warp: torch.tensor = torch.tensor([1.0]),
                 warp_angle: torch.tensor = torch.tensor([0.0])
    ):
        super(WarpedGaussian2D, self).__init__()
        # Ensure that the parameters are tensors
        if not isinstance(amplitude, torch.Tensor):
            amplitude = torch.tensor(amplitude)
        if not isinstance(center_y, torch.Tensor):
            center_y = torch.tensor(center_y)
        if not isinstance(center_x, torch.Tensor):
            center_x = torch.tensor(center_x)
        if not isinstance(sigma_y, torch.Tensor):
            sigma_y = torch.tensor(sigma_y)
        if not isinstance(sigma_x, torch.Tensor):
            sigma_x = torch.tensor(sigma_x)
        if not isinstance(warp, torch.Tensor):
            warp = torch.tensor(warp)
        if not isinstance(warp_angle, torch.Tensor):
            warp_angle = torch.tensor(warp_angle)
        # Check if all parameters are of the same shape
        assert amplitude.shape == center_y.shape == center_x.shape == sigma_y.shape == sigma_x.shape == warp.shape == warp_angle.shape, \
            "All parameters must have the same shape."

        self.amplitude = nn.Parameter(amplitude)
        self.center_y = nn.Parameter(center_y)
        self.center_x = nn.Parameter(center_x)
        self.sigma_y = nn.Parameter(sigma_y)
        self.sigma_x = nn.Parameter(sigma_x)
        self.warp = nn.Parameter(warp)
        self.warp_angle = nn.Parameter(warp_angle)

    def forward(self, grid):
        """
        Forward pass for 2D warped Gaussian list.
        
        Args:
            grid: Tensor of shape (h,w, 2) containing 2D coordinates in yx order.
            
        Returns
        -------
            Tensor of warped Gaussian values
        """
        amplitude = einops.rearrange(self.amplitude, '... -> 1 1 ...')
        center_x = einops.rearrange(self.center_x, '... -> 1 1 ...')
        center_y = einops.rearrange(self.center_y, '... -> 1 1 ...')
        sigma_x = einops.rearrange(self.sigma_x, '... -> 1 1 ...')
        sigma_y = einops.rearrange(self.sigma_y, '... -> 1 1 ...')
        warp = einops.rearrange(self.warp, '... -> 1 1 ...')
        warp_angle = einops.rearrange(self.warp_angle, '... -> 1 1 ...')

        grid_x = einops.rearrange(grid[..., 1], 'h w -> h w 1')
        grid_y = einops.rearrange(grid[..., 0], 'h w -> h w 1')

        u = (grid_x - center_x) * torch.cos(warp_angle) - (grid_y - center_y) * torch.sin(warp_angle)
        v = (grid_x - center_x) * torch.sin(warp_angle) + (grid_y - center_y) * torch.cos(warp_angle)

        warped_gaussian = amplitude * torch.exp(
            -((u - warp * v ** 2) ** 2 / (2 * sigma_x ** 2) +
              v ** 2 / (2 * sigma_y ** 2))
        )

        return einops.rearrange(warped_gaussian, 'h w ... -> ... h w')


class Gaussian2D(nn.Module):
    """
    A 2D Gaussian function.

    Parameters
    ----------
    amplitude : torch.tensor, optional
        Amplitude of the Gaussian. Default is torch.tensor([1.0]).
    center_y : torch.tensor, optional
        Y-coordinate of the center. Default is torch.tensor([0.0]).
    center_x : torch.tensor, optional
        X-coordinate of the center. Default is torch.tensor([0.0]).
    sigma_y : torch.tensor, optional
        Standard deviation along the y-axis. Default is torch.tensor([1.0]).
    sigma_x : torch.tensor, optional
        Standard deviation along the x-axis. Default is torch.tensor([1.0]).

    Methods
    -------
    forward(grid)
        Compute the Gaussian values for a given 2D grid.
        Expects grid in yx order (grid[..., 0] is y, grid[..., 1] is x).
    """

    def __init__(self,
                 amplitude: Union[torch.Tensor | float] = 1.0,
                 center_y: Union[torch.Tensor | float] = 0.0,
                 center_x: Union[torch.Tensor | float] = 0.0,
                 sigma_y: Union[torch.Tensor | float] = 1.0,
                 sigma_x: Union[torch.Tensor | float] = 1.0
    ):
        super(Gaussian2D, self).__init__()
        # Ensure that the parameters are tensors
        if not isinstance(amplitude, torch.Tensor):
            amplitude = torch.tensor(amplitude)
        if not isinstance(center_y, torch.Tensor):
            center_y = torch.tensor(center_y)
        if not isinstance(center_x, torch.Tensor):
            center_x = torch.tensor(center_x)
        if not isinstance(sigma_y, torch.Tensor):
            sigma_y = torch.tensor(sigma_y)
        if not isinstance(sigma_x, torch.Tensor):
            sigma_x = torch.tensor(sigma_x)
        # Check if all parameters are of the same shape
        assert amplitude.shape == center_y.shape == center_x.shape == sigma_y.shape == sigma_x.shape, \
            "All parameters must have the same shape."

        self.amplitude = nn.Parameter(amplitude)
        self.center_y = nn.Parameter(center_y)
        self.center_x = nn.Parameter(center_x)
        self.sigma_y = nn.Parameter(sigma_y)
        self.sigma_x = nn.Parameter(sigma_x)

    def forward(self, grid):
        """
        Forward pass for 2D Gaussian list.
        
        Args:
            grid: Tensor of shape (h,w, 2) containing 2D coordinates in yx order.
            
        Returns
        -------
            Tensor of Gaussian values
        """
        # Add batch dimension
        grid_x = einops.rearrange(grid[..., 1], 'h w -> h w' + ' 1'*self.amplitude.dim())
        grid_y = einops.rearrange(grid[..., 0], 'h w -> h w' + ' 1'*self.amplitude.dim())
        
        amplitude = einops.rearrange(self.amplitude, '... -> 1 1 ...')
        center_x = einops.rearrange(self.center_x, '... -> 1 1 ...')
        center_y = einops.rearrange(self.center_y, '... -> 1 1 ...')
        sigma_x = einops.rearrange(self.sigma_x, '... -> 1 1 ...')
        sigma_y = einops.rearrange(self.sigma_y, '... -> 1 1 ...')

        gaussian = amplitude * torch.exp(
            -((grid_x - center_x) ** 2 / (2 * sigma_x ** 2) +
              (grid_y - center_y) ** 2 / (2 * sigma_y ** 2))
        )

        return einops.rearrange(gaussian, 'h w ... -> ... h w')


class Gaussian3D(nn.Module):
    """
    A 3D Gaussian function.

    Parameters
    ----------
    amplitude : torch.tensor, optional
        Amplitude of the Gaussian. Default is torch.tensor([1.0]).
    center_z : torch.tensor, optional
        Z-coordinate of the center. Default is torch.tensor([0.0]).
    center_y : torch.tensor, optional
        Y-coordinate of the center. Default is torch.tensor([0.0]).
    center_x : torch.tensor, optional
        X-coordinate of the center. Default is torch.tensor([0.0]).
    sigma_z : torch.tensor, optional
        Standard deviation along the z-axis. Default is torch.tensor([1.0]).
    sigma_y : torch.tensor, optional
        Standard deviation along the y-axis. Default is torch.tensor([1.0]).
    sigma_x : torch.tensor, optional
        Standard deviation along the x-axis. Default is torch.tensor([1.0]).

    Methods
    -------
    forward(grid)
        Compute the Gaussian values for a given 3D grid.
        Expects grid in zyx order (grid[..., 0] is z, grid[..., 1] is y, grid[..., 2] is x).
    """

    def __init__(self,
                 amplitude: Union[torch.Tensor | float] = 1.0,
                 center_z: Union[torch.Tensor | float] = 0.0,
                 center_y: Union[torch.Tensor | float] = 0.0,
                 center_x: Union[torch.Tensor | float] = 0.0,
                 sigma_z: Union[torch.Tensor | float] = 1.0,
                 sigma_y: Union[torch.Tensor | float] = 1.0,
                 sigma_x: Union[torch.Tensor | float] = 1.0
    ):
        super(Gaussian3D, self).__init__()
        # Ensure that the parameters are tensors
        if not isinstance(amplitude, torch.Tensor):
            amplitude = torch.tensor(amplitude)
        if not isinstance(center_z, torch.Tensor):
            center_z = torch.tensor(center_z)
        if not isinstance(center_y, torch.Tensor):
            center_y = torch.tensor(center_y)
        if not isinstance(center_x, torch.Tensor):
            center_x = torch.tensor(center_x)
        if not isinstance(sigma_z, torch.Tensor):
            sigma_z = torch.tensor(sigma_z)
        if not isinstance(sigma_y, torch.Tensor):
            sigma_y = torch.tensor(sigma_y)
        if not isinstance(sigma_x, torch.Tensor):
            sigma_x = torch.tensor(sigma_x)
        # Check if all parameters are of the same shape
        assert amplitude.shape == center_z.shape == center_y.shape == center_x.shape == sigma_z.shape == sigma_y.shape == sigma_x.shape, \
            "All parameters must have the same shape."

        self.amplitude = nn.Parameter(amplitude)
        self.center_z = nn.Parameter(center_z)
        self.center_y = nn.Parameter(center_y)
        self.center_x = nn.Parameter(center_x)
        self.sigma_z = nn.Parameter(sigma_z)
        self.sigma_y = nn.Parameter(sigma_y)
        self.sigma_x = nn.Parameter(sigma_x)

    def forward(self, grid):
        """
        Forward pass for 3D Gaussian list.
        
        Args:
            grid: Tensor of shape (d, h, w, 3) containing 3D coordinates in zyx order 
                 (grid[..., 0] is z, grid[..., 1] is y, grid[..., 2] is x).
            
        Returns
        -------
            Tensor of Gaussian values
        """
         # Add batch dimension
        grid_x = einops.rearrange(grid[..., 2], 'd h w -> d h w' + ' 1'*self.amplitude.dim())
        grid_y = einops.rearrange(grid[..., 1], 'd h w -> d h w' + ' 1'*self.amplitude.dim())
        grid_z = einops.rearrange(grid[..., 0], 'd h w -> d h w' + ' 1'*self.amplitude.dim())

        amplitude = einops.rearrange(self.amplitude, '... -> 1 1 1 ...')
        center_x = einops.rearrange(self.center_x, '... -> 1 1 1 ...')
        center_y = einops.rearrange(self.center_y, '... -> 1 1 1 ...')
        center_z = einops.rearrange(self.center_z, '... -> 1 1 1 ...')
        sigma_x = einops.rearrange(self.sigma_x, '... -> 1 1 1 ...')
        sigma_y = einops.rearrange(self.sigma_y, '... -> 1 1 1 ...')
        sigma_z = einops.rearrange(self.sigma_z, '... -> 1 1 1 ...')


        gaussian = amplitude * torch.exp(
            -((grid_x - center_x) ** 2 / (2 * sigma_x ** 2) +
              (grid_y - center_y) ** 2 / (2 * sigma_y ** 2) +
              (grid_z - center_z) ** 2 / (2 * sigma_z ** 2))
        )

        return einops.rearrange(gaussian, 'd h w ... -> ... d h w')
