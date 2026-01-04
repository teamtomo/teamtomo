from typing import Sequence

import torch


def rfft_shape(input_shape: Sequence[int]) -> tuple[int, ...]:
    """Get the output shape of an rfft on an input with input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)

def fftshift_1d(input: torch.Tensor, rfft: bool = True) -> torch.Tensor:
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-1, ))
    else:
        output = input
    return output

def fftshift_2d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-2, -1))
    else:
        output = torch.fft.fftshift(input, dim=(-2,))
    return output

def ifftshift_1d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-1, ))
    else:
        output = input
    return output

def ifftshift_2d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-2, -1))
    else:
        output = torch.fft.ifftshift(input, dim=(-2,))
    return output


def fftshift_3d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    if rfft is False:
        output = torch.fft.fftshift(input, dim=(-3, -2, -1))
    else:
        output = torch.fft.fftshift(input, dim=(-3, -2,))
    return output


def ifftshift_3d(input: torch.Tensor, rfft: bool) -> torch.Tensor:
    if rfft is False:
        output = torch.fft.ifftshift(input, dim=(-3, -2, -1))
    else:
        output = torch.fft.ifftshift(input, dim=(-3, -2,))
    return output
