import torch
from torch_grid_utils import fftfreq_grid


def _central_line_fftfreq_grid(
    image_shape: tuple[int, int],
    rfft: bool,
    fftshift: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    # 1d grid of DFT sample frequencies along w, shape (w, 1)
    (w,) = image_shape[-1:]
    grid = fftfreq_grid(image_shape=(w,), rfft=rfft, fftshift=fftshift, device=device)

    # prepend zeros as the h coordinate of the central line, shape (w, 2)
    return torch.cat([torch.zeros_like(grid), grid], dim=-1)
