import einops
import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid, fftshift_2d, rfft_shape


def _central_slice_fftfreq_grid(
    volume_shape: tuple[int, int, int],
    rfft: bool,
    fftshift: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    # generate 2d grid of DFT sample frequencies, shape (h, w, 2)
    h, w = volume_shape[-2:]
    grid = fftfreq_grid(image_shape=(h, w), rfft=rfft, device=device)  # (h, w, 2)

    # get grid of same shape with all zeros, append as third coordinate
    if rfft is True:
        zeros = torch.zeros(size=rfft_shape((h, w)), dtype=grid.dtype, device=device)
    else:
        zeros = torch.zeros(size=(h, w), dtype=grid.dtype, device=device)
    central_slice_grid, _ = einops.pack([zeros, grid], pattern="h w *")  # (h, w, 3)

    # fftshift if requested
    if fftshift is True:
        central_slice_grid = einops.rearrange(
            central_slice_grid, "h w freq -> freq h w"
        )
        central_slice_grid = fftshift_2d(central_slice_grid, rfft=rfft)
        central_slice_grid = einops.rearrange(
            central_slice_grid, "freq h w -> h w freq"
        )
    return central_slice_grid
