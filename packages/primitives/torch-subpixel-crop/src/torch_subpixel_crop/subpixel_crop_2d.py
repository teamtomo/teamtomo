import einops
import torch
from torch.nn import functional as F
from torch_fourier_shift import fourier_shift_dft_2d, fourier_shift_image_2d
from torch_grid_utils import coordinate_grid
from typing import Optional

from torch_subpixel_crop.dft_utils import dft_center
from torch_subpixel_crop.grid_sample_utils import array_to_grid_sample


def subpixel_crop_2d(
        image: torch.Tensor,
        positions: torch.Tensor,
        sidelength: int,
        mask: Optional[torch.Tensor] = None,
        return_rfft: bool = False,
        decenter: bool = False,
):
    """Extract square patches from 2D images with subpixel precision.

    Parameters
    ----------
    image : torch.Tensor
        `(b, h, w)` or `(h, w)` array of 2D images.
    positions : torch.Tensor
        `(..., b, 2)` or `(..., 2)` array of coordinates for patch centers.
    sidelength : int
        Sidelength of square patches extracted from `images`.
    mask : torch.Tensor
        Optional mask to apply in real space before FFT, shape (size, size)
         or broadcastable to (..., b, size, size)
    return_rfft : bool, default False
        If `True`, return the rft of the patches. It can save an FFT
         operation because the subpixel shift already requires an FFT. The returned
         rffts have their origin at (0, 0)
    decenter : bool, default False
        In case the patches are returned as rft, optionally also apply an
         additional shift of sidelength / 2 so that the real space image has its origin
         at 0. NOTE: this does not change the origin in Fourier space, the rffts
         still have their origin at (0, 0)

    Returns
    -------
    patches : torch.Tensor
        `(..., b, sidelength, sidelength)` or `(..., sidelength, sidelength)`
        array of patches.
    """
    # Handle unbatched input
    if image.ndim == 2:
        input_images_are_batched = False
        image = einops.rearrange(image, 'h w -> 1 h w')
        positions = einops.rearrange(positions, '... yx -> ... 1 yx')
    else:
        input_images_are_batched = True

    # Flatten batch dimensions
    positions, ps = einops.pack([positions], pattern='* batch yx')

    # Process ALL images at once (no loop!)
    patches = _extract_patches_batched(
        images=image,  # (batch, h, w)
        positions=positions,  # (..., batch, 2)
        output_image_sidelength=sidelength,
        mask=mask,
        return_rfft=return_rfft,
        decenter=decenter,
    )

    # Restore original shape
    [patches] = einops.unpack(patches, pattern='* batch h w', packed_shapes=ps)

    if not input_images_are_batched:
        patches = einops.rearrange(patches, '... 1 h w -> ... h w')

    return patches


def _extract_patches_batched(
        images: torch.Tensor,  # (batch, h, w)
        positions: torch.Tensor,  # (n, batch, 2)
        output_image_sidelength: int,
        mask: Optional[torch.Tensor] = None,
        return_rfft: bool = False,
        decenter: bool = False,
) -> torch.Tensor:  # (n, batch, ph, pw)
    batch, h, w = images.shape
    n_pos, batch_check, _ = positions.shape
    if batch != batch_check:
        raise ValueError('Mismatch in batch size for images and positions.')

    # Find integer positions and shifts
    integer_positions = torch.round(positions)
    shifts = integer_positions - positions  # (n_pos, batch, 2)

    # Generate coordinate grid
    ph = pw = output_image_sidelength
    center = dft_center((ph, pw), rfft=False, fftshifted=True,
                        device=images.device)
    grid = coordinate_grid(
        image_shape=(ph, pw),
        center=center,
        device=images.device
    )  # (ph, pw, 2)

    # Broadcast grid for all positions and batches
    grid = einops.rearrange(grid, 'ph pw yx -> 1 1 ph pw yx')
    integer_positions = einops.rearrange(
        integer_positions, 'n_pos batch yx -> n_pos batch 1 1 yx'
    )
    grid = grid + integer_positions  # (n_pos, batch, ph, pw, 2)

    # flip batch and n_pos, so that n_pos indexes depth
    grid_flat = (
        einops.rearrange(grid, 'n_pos batch ph pw yx -> batch n_pos ph pw yx')
    )
    # add z=0 to trick the 3D case of grid_sample for our 2d application
    # we end up with (batch n_pos ph pw zyx)
    grid_flat = F.pad(grid_flat, [1, 0], mode='constant', value=0)

    # add empty channel and depth dimension to images, depth is used to sample
    # multiple positions simultaneously
    images_rep = einops.rearrange(
        images, 'batch h w -> batch 1 1 h w'
    )

    # Extract all patches at once
    patches = F.grid_sample(
        input=images_rep,
        grid=array_to_grid_sample(grid_flat, array_shape=(0, h, w)),
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )  # (batch, 1, n_pos, ph, pw)

    # rearrange to correct order
    patches = einops.rearrange(
        patches, 'batch 1 n_pos ph pw -> n_pos batch ph pw',
    )

    if mask is not None:
        patches = patches * mask

    if return_rfft:
        # fft the patches
        patches_dft = torch.fft.rfftn(patches, dim=(-2, -1))

        if decenter:
            shifts = shifts + pw / 2

        # apply the subpixel shift
        patches_dft = fourier_shift_dft_2d(
            dft=patches_dft,
            image_shape=(ph, pw),
            shifts=shifts,
            rfft=True,
            fftshifted=False,
        )

        return patches_dft

    else:
        patches = fourier_shift_image_2d(image=patches, shifts=shifts)
        return patches