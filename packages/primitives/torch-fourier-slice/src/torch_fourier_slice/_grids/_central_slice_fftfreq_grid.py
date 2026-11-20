import torch
from torch_grid_utils import fftfreq_grid


def _central_slice_fftfreq_grid(
    volume_shape: tuple[int, int, int],
    rfft: bool,
    fftshift: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    # 2d grid of DFT sample frequencies, shape (h, w, 2)
    h, w = volume_shape[-2:]
    grid = fftfreq_grid(image_shape=(h, w), rfft=rfft, fftshift=fftshift, device=device)

    # prepend zeros as the d coordinate of the central slice, shape (h, w, 3)
    return torch.cat([torch.zeros_like(grid[..., :1]), grid], dim=-1)
