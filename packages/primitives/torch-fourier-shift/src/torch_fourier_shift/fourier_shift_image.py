import torch

from torch_fourier_shift.fourier_shift_dft import (
    fourier_shift_dft_1d,
    fourier_shift_dft_2d,
    fourier_shift_dft_3d,
)


def fourier_shift_image_1d(image: torch.Tensor, shifts: torch.Tensor):
    """Translate one or more 1D images by phase shifting their Fourier transforms.

    Parameters
    ----------
    image: torch.Tensor
        `(..., w)` image(s).
    shifts: torch.Tensor
        `(..., )` array of 1D shifts in `w`.

    Returns
    -------
    shifted_images: torch.Tensor
        `(..., w)` array of shifted images.
    """
    w = image.shape[-1]
    image = torch.fft.rfftn(image, dim=(-1, ))
    image = fourier_shift_dft_1d(
        image,
        image_shape=(w, ),
        shifts=shifts,
        rfft=True,
        fftshifted=False
    )
    image = torch.fft.irfftn(image, dim=(-1, ))
    return torch.real(image)


def fourier_shift_image_2d(image: torch.Tensor, shifts: torch.Tensor):
    """Translate one or more 2D images by phase shifting their Fourier transforms.

    Parameters
    ----------
    image: torch.Tensor
        `(..., h, w)` image(s).
    shifts: torch.Tensor
        `(..., 2)` array of 2D shifts in `h` and `w`.

    Returns
    -------
    shifted_images: torch.Tensor
        `(..., h, w)` array of shifted images.
    """
    h, w = image.shape[-2:]
    image = torch.fft.rfftn(image, dim=(-2, -1))
    image = fourier_shift_dft_2d(
        image,
        image_shape=(h, w),
        shifts=shifts,
        rfft=True,
        fftshifted=False
    )
    image = torch.fft.irfftn(image, dim=(-2, -1))
    return torch.real(image)


def fourier_shift_image_3d(image: torch.Tensor, shifts: torch.Tensor):
    """Translate one or more 3D images by phase shifting their Fourier transforms.

    Parameters
    ----------
    image: torch.Tensor
        `(..., d, h, w)` image(s).
    shifts: torch.Tensor
        `(..., 3)` array of 3D shifts in `d`, `h` and `w`.

    Returns
    -------
    shifted_image: torch.Tensor
        `(..., d, h, w)` array of shifted images.
    """
    d, h, w = image.shape[-3:]
    image = torch.fft.rfftn(image, dim=(-3, -2, -1))
    image = fourier_shift_dft_3d(
        image,
        image_shape=(d, h, w),
        shifts=shifts,
        rfft=True,
        fftshifted=False
    )
    image = torch.fft.irfftn(image, dim=(-3, -2, -1))
    return torch.real(image)
