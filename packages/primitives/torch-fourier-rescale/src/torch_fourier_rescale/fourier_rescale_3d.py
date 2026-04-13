import numpy as np
import torch

from .utils import (
    calculate_new_spacing,
    calculate_target_shape_from_spacing,
    fourier_rescale_dimension,
    normalize_spacing,
)


def fourier_rescale_3d(
    image: torch.Tensor,
    source_spacing: float | tuple[float, float, float] | None = None,
    target_spacing: float | tuple[float, float, float] | None = None,
    target_shape: tuple[int, int, int] | None = None,
    preserve_mean: bool = True,
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    """Rescale 3D image(s) from `source_spacing` to `target_spacing` or to `target_shape`.

    Rescaling is performed in Fourier space by either cropping or padding the
    discrete Fourier transform (DFT).

    Parameters
    ----------
    image: torch.Tensor
        `(..., d, h, w)` array of image data
    source_spacing: float | tuple[float, float, float] | None
        Pixel spacing in the input image. Required if target_spacing is specified.
        If None when target_shape is specified, assumes uniform spacing of 1.0.
    target_spacing: float | tuple[float, float, float] | None
        Pixel spacing in the output image. Mutually exclusive with target_shape.
    target_shape: tuple[int, int, int] | None
        Target spatial dimensions (depth, height, width) for the output image.
        Mutually exclusive with target_spacing.
    preserve_mean: bool = True
        Ensure that the mean (DC component) of the array is preserved after rescaling.

    Returns
    -------
    rescaled_image, (new_spacing_d, new_spacing_h, new_spacing_w)
    """
    # Case 1: Spacing-based rescaling
    if target_spacing is not None:
        if source_spacing is None:
            raise ValueError(
                "source_spacing is required when target_spacing is specified"
            )
        if target_shape is not None:
            raise ValueError("Cannot specify both target_spacing and target_shape")

        # Normalize to tuples
        source_spacing = normalize_spacing(source_spacing, 3)
        target_spacing = normalize_spacing(target_spacing, 3)

        # Early return if no change needed
        if np.allclose(source_spacing, target_spacing, atol=1e-8):
            return image, source_spacing

        # Calculate target_shape from spacing ratio
        source_shape = image.shape[-3:]
        target_shape = calculate_target_shape_from_spacing(
            source_shape, source_spacing, target_spacing
        )

    # Case 2: Shape-based rescaling
    elif target_shape is not None:
        # Set default source_spacing if not provided
        source_spacing = 1.0 if source_spacing is None else source_spacing
        source_spacing = normalize_spacing(source_spacing, 3)

    # Neither specified
    else:
        raise ValueError("Either target_spacing or target_shape must be specified")

    # place image center at array indices [0, 0, 0] and compute centered rfft3
    image = torch.fft.fftshift(image, dim=(-3, -2, -1))
    dft = torch.fft.rfftn(image, dim=(-3, -2, -1))
    dft = torch.fft.fftshift(dft, dim=(-3, -2))

    # Calculate target shape if using spacing
    if target_shape is None:
        # Calculate target shape from spacing
        source_shape = image.shape[-3:]
        target_shape = tuple(
            int(np.round(src_sh * (src_sp / tgt_sp)))
            for src_sh, src_sp, tgt_sp in zip(
                source_shape, source_spacing, target_spacing
            )
        )

    # Fourier pad/crop
    dft = fourier_rescale_rfft_3d(
        dft=dft,
        image_shape=image.shape[-3:],
        target_shape=target_shape,
    )
    new_shape = target_shape

    # transform back to real space and recenter
    dft = torch.fft.ifftshift(dft, dim=(-3, -2))
    if preserve_mean:
        # we changed the number of elements in the FT so set norm='forward' to deactivate
        # default fft normalization by 1/n and normalise by the correct factor
        rescaled_image = torch.fft.irfftn(
            dft, dim=(-3, -2, -1), s=new_shape, norm="forward"
        )
        rescaled_image = rescaled_image * (1 / np.prod(image.shape[-3:]))
    else:
        rescaled_image = torch.fft.irfftn(dft, dim=(-3, -2, -1), s=new_shape)
    rescaled_image = torch.fft.ifftshift(rescaled_image, dim=(-3, -2, -1))

    # Calculate new spacing after rescaling
    new_spacing = calculate_new_spacing(source_spacing, image.shape[-3:], new_shape)

    return rescaled_image, new_spacing


def fourier_rescale_rfft_3d(
    dft: torch.Tensor,
    image_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Rescale a 3D rfft by padding or cropping to achieve target shape.

    Parameters
    ----------
    dft : torch.Tensor
        The result of fftshift(rfft(volume, dim=(-3, -2, -1)), dim=(-3, -2)).
    image_shape : tuple[int, int, int]
        Original image shape (d, h, w)
    target_shape : tuple[int, int, int]
        Target image shape (d, h, w)

    Returns
    -------
    torch.Tensor
        The rescaled DFT
    """
    d, h, w = image_shape
    target_d, target_h, target_w = target_shape

    # Handle depth dimension (regular FFT dimension)
    if target_d != d:
        dft = fourier_rescale_dimension(
            dft, dim=-3, source_dim_length=d, target_dim_length=target_d, is_rfft=False
        )

    # Handle height dimension (regular FFT dimension)
    if target_h != h:
        dft = fourier_rescale_dimension(
            dft, dim=-2, source_dim_length=h, target_dim_length=target_h, is_rfft=False
        )

    # Handle width dimension (rfft dimension)
    if target_w != w:
        dft = fourier_rescale_dimension(
            dft, dim=-1, source_dim_length=w, target_dim_length=target_w, is_rfft=True
        )

    return dft
