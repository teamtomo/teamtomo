"""Functions to construct and convert between Cartesian and polar coordinate grids."""

import torch

from torch_grid_utils.coordinate_grid import coordinate_grid


def polar_grid(
    image_shape: tuple[int, int],
    center: torch.Tensor | tuple[float, float] | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a 2D polar coordinate grid from image dimensions.

    For input `image_shape` of `(h, w)`, this function produces polar coordinates
    `(rho, theta)` with shapes `(h, w)`.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Shape of the 2D image `(h, w)` for which polar coordinates should be
        returned.
    center : torch.Tensor | tuple[float, float] | None
        Center point relative to which coordinates will be calculated.
        If `None`, defaults to the image center `(h//2, w//2)`.
    device : torch.device | None
        PyTorch device on which to put the coordinate grid.

    Returns
    -------
    rho : torch.Tensor
        Radial distance from the center. Shape `(h, w)`.
    theta : torch.Tensor
        Polar angle in radians, ranging from -π to π. Shape `(h, w)`.
    """
    if len(image_shape) != 2:
        raise ValueError(
            f"polar_grid currently only supports 2D images, got shape {image_shape}"
        )

    # Create Cartesian coordinate grid
    if center is None:
        center = (image_shape[0] / 2, image_shape[1] / 2)

    cartesian_grid = coordinate_grid(
        image_shape=image_shape,
        center=center,
        norm=False,
        device=device,
    )

    # Convert to polar coordinates
    rho, theta = cartesian_to_polar(cartesian_grid)

    return rho, theta


def cartesian_to_polar(
    cartesian_grid: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a Cartesian coordinate grid to polar coordinates.

    For a 2D Cartesian grid with shape `(..., h, w, 2)`, this function produces
    radial and angular coordinates `(rho, theta)` with shapes `(..., h, w)`.

    Parameters
    ----------
    cartesian_grid : torch.Tensor
        Cartesian coordinate grid with shape `(..., 2)` where the last dimension
        contains `[y, x]` coordinates (matching dimension order `(h, w)`).
    eps : float
        Small constant added to the radial distance to avoid division by zero
        and numerical issues at the origin.

    Returns
    -------
    rho : torch.Tensor
        Radial distance from the origin. Same shape as input without the last
        dimension: `(..., )`.
    theta : torch.Tensor
        Polar angle in radians, ranging from -π to π. Same shape as `rho`.
    """
    if cartesian_grid.shape[-1] != 2:
        raise ValueError(
            f"cartesian_grid must have shape (..., 2), got {cartesian_grid.shape}"
        )

    y = cartesian_grid[..., 0]
    x = cartesian_grid[..., 1]

    rho = torch.sqrt(x**2 + y**2 + eps)
    theta = torch.atan2(y, x)

    return rho, theta


def polar_to_cartesian(
    rho: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Convert polar coordinates to a Cartesian coordinate grid.

    Parameters
    ----------
    rho : torch.Tensor
        Radial distance from the origin. Shape `(..., )`.
    theta : torch.Tensor
        Polar angle in radians. Must have the same shape as `rho`.

    Returns
    -------
    cartesian_grid : torch.Tensor
        Cartesian coordinate grid with shape `(..., 2)` where the last dimension
        contains `[y, x]` coordinates (matching dimension order `(h, w)`).
    """
    if rho.shape != theta.shape:
        raise ValueError(
            f"rho and theta must have the same shape, got {rho.shape} and {theta.shape}"
        )

    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)

    return torch.stack([y, x], dim=-1)


def normalize_polar_grid(
    rho: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the radial distance in a polar coordinate grid.

    Normalizes the radial distance `rho` to the range [0, 1] by dividing by the
    maximum radial distance. The angular coordinate `theta` is returned unchanged.

    Parameters
    ----------
    rho : torch.Tensor
        Radial distance from the origin. Shape `(..., )`.
    theta : torch.Tensor
        Polar angle in radians. Must have the same shape as `rho`.

    Returns
    -------
    rho_norm : torch.Tensor
        Normalized radial distance in the range [0, 1]. Same shape as input `rho`.
    theta : torch.Tensor
        Polar angle in radians, unchanged from input. Same shape as input `theta`.
    """
    if rho.shape != theta.shape:
        raise ValueError(
            f"rho and theta must have the same shape, got {rho.shape} and {theta.shape}"
        )

    rho_norm = rho / rho.max()

    return rho_norm, theta


def fftfreq_grid_polar(
    fft_freq_grid: torch.Tensor,
    eps: float = 1e-12,
    normalize_rho: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a Cartesian frequency grid to polar coordinates.

    This is a convenience function specifically for frequency grids that normalizes
    the radial frequency by default.

    Parameters
    ----------
    fft_freq_grid : torch.Tensor
        Cartesian frequency grid with shape `(..., h, w, 2)` where the last
        dimension contains `[ky, kx]` frequencies (matching dimension order `(h, w)`).
    eps : float
        Small constant added to the radial distance to avoid division by zero.
    normalize_rho : bool
        Whether to normalize the radial frequency to the range [0, 1] by dividing
        by the maximum radial frequency.

    Returns
    -------
    rho_norm : torch.Tensor
        Normalized radial frequency (0..1) if `normalize_rho=True`, otherwise
        unnormalized radial frequency. Shape `(..., h, w)`.
    theta : torch.Tensor
        Polar angle in radians (-π..π). Shape `(..., h, w)`.
    """
    rho, theta = cartesian_to_polar(fft_freq_grid, eps=eps)

    if normalize_rho:
        rho, theta = normalize_polar_grid(rho, theta)

    return rho, theta
