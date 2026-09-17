import torch


def separable_sinc2_correction(
    shape: tuple[int, ...], device: torch.device | None = None
) -> torch.Tensor:
    """sinc^2 correction for linear interpolation, applied per-axis.

    Parameters
    ----------
    shape: tuple[int, ...]
        Shape of the real-space array to correct.
    device: torch.device | None
        PyTorch device on which the returned correction will be stored.

    Returns
    -------
    correction: torch.Tensor
        `shape` array of sinc^2 correction factors, broadcastable against
        an array of `shape` for elementwise division.
    """
    ndim = len(shape)
    correction = torch.ones((), device=device)

    for axis, size in enumerate(shape):
        freq = torch.fft.fftshift(torch.fft.fftfreq(size, device=device))
        view_shape = [1] * ndim
        view_shape[axis] = size
        correction = correction * torch.sinc(freq).view(view_shape)  # memory efficient

    return correction**2


def boundary_shell_average(
    volume: torch.Tensor, n: int, n_spatial_dims: int
) -> torch.Tensor:
    """Average of `n` voxels/pixels from rectangular boundary, handling batch/channels.

    Parameters
    ----------
    volume: torch.Tensor
        `(..., d, d)` or `(..., d, d, d)` volume, optionally with leading batch/channel
        dimensions.
    n: int
        Number of voxels/pixels from the boundary to include in the average.
    n_spatial_dims: int
        Number of spatial dimensions in the volume.

    Returns
    -------
    torch.Tensor
        `(...)` average value of `n` voxels/pixels from the boundary,
        one value per leading batch/channel element (0-dim if `volume` is `(d, d)`
        or `(d, d, d)`).
    """
    spatial_shape = volume.shape[-n_spatial_dims:]
    d = spatial_shape[0]

    assert n >= 1, "n must be >= 1"
    assert len(set(spatial_shape)) == 1, "all spatial dimensions must be equal."
    assert n < d // 2, "n must be less than half the spatial size."

    spatial_dims = tuple(range(-n_spatial_dims, 0))
    interior_slice = (Ellipsis,) + (slice(n, -n),) * n_spatial_dims
    interior = volume[interior_slice]

    total_sum = volume.sum(dim=spatial_dims)
    total_elements = 1
    for s in spatial_shape:
        total_elements *= s

    interior_sum = interior.sum(dim=spatial_dims)
    interior_elements = 1
    for s in interior.shape[-n_spatial_dims:]:
        interior_elements *= s

    shell_sum = total_sum - interior_sum
    shell_elements = total_elements - interior_elements

    return shell_sum / shell_elements


def compute_cube_face_averages(volume: torch.Tensor, n: int = 1) -> torch.Tensor:
    """Get the average value of all voxels within n-voxels of the cube faces.

    Parameters
    ----------
    volume: torch.Tensor
        `(..., d, d, d)` volume, optionally with leading batch/channel dims.
    n: int
        Number of voxels from the cube faces to include in the average.

    Returns
    -------
    torch.Tensor
        `(...)` average value of all voxels within n-voxels of the cube faces,
        one value per leading batch/channel element (0-dim if `volume` is `(d, d, d)`).
    """
    return boundary_shell_average(volume, n=n, n_spatial_dims=3)


def compute_square_edge_averages(image: torch.Tensor, n: int = 1) -> torch.Tensor:
    """Get the average value of all pixels within n-pixels of the square edges.

    Parameters
    ----------
    image: torch.Tensor
        `(..., d, d)` image, optionally with leading batch/channel dims.
    n: int
        Number of pixels from the square edges to include in the average.

    Returns
    -------
    torch.Tensor
        `(...)` average value of all pixels within n-pixels of the square edges,
        one value per leading batch/channel element (0-dim if `image` is `(d, d)`).
    """
    return boundary_shell_average(image, n=n, n_spatial_dims=2)
