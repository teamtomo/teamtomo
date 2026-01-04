import einops
import torch
from torch_grid_utils.fftfreq_grid import fftshift_1d, rfft_shape


def _central_line_fftfreq_grid(
    image_shape: tuple[int, int],
    rfft: bool,
    fftshift: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    # generate 1d grid of DFT sample frequencies, shape (w, 1)
    (w,) = image_shape[-1:]
    grid = (
        torch.fft.rfftfreq(w, device=device)
        if rfft
        else torch.fft.fftfreq(w, device=device)
    )

    # get grid of same shape with all zeros, append as third coordinate
    if rfft is True:
        zeros = torch.zeros(size=rfft_shape((w,)), dtype=grid.dtype, device=device)
    else:
        zeros = torch.zeros(size=(w,), dtype=grid.dtype, device=device)
    central_slice_grid, _ = einops.pack([zeros, grid], pattern="w *")  # (w, 2)

    # fftshift if requested
    if fftshift is True:
        central_slice_grid = einops.rearrange(central_slice_grid, "w freq -> freq w")
        central_slice_grid = fftshift_1d(central_slice_grid, rfft=rfft)
        central_slice_grid = einops.rearrange(central_slice_grid, "freq w -> w freq")
    return central_slice_grid
