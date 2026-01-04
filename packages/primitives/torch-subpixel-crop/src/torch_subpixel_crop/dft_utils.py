from typing import Tuple, Sequence

import torch


def rfft_shape(input_shape: Sequence[int]) -> Tuple[int]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)


def dft_center(
    image_shape: Tuple[int, ...],
    rfft: bool,
    fftshifted: bool,
    device: torch.device | None = None,
) -> torch.LongTensor:
    """Return the position of the DFT center for a given input shape."""
    fft_center = torch.zeros(size=(len(image_shape),), device=device)
    image_shape = torch.as_tensor(image_shape).float()
    if rfft is True:
        image_shape = torch.tensor(rfft_shape(image_shape))
    if fftshifted is True:
        fft_center = torch.divide(image_shape, 2, rounding_mode='floor')
    if rfft is True:
        fft_center[-1] = 0
    return fft_center.long()
