import torch
import einops
import torch.nn.functional as F
from typing import Optional
from torch_fourier_shift import fourier_shift_image_3d, fourier_shift_dft_3d
from torch_grid_utils import coordinate_grid

from torch_subpixel_crop.dft_utils import dft_center
from torch_subpixel_crop.grid_sample_utils import array_to_grid_sample


def subpixel_crop_3d(
        image: torch.Tensor,  # (d, h, w)
        positions: torch.Tensor,  # (b, 3) zyx
        sidelength: int,
        mask: Optional[torch.Tensor] = None,
        return_rfft: bool = False,
        decenter: bool = False,
) -> torch.Tensor:
    """Extract cubic patches from a 3D image with subpixel precision.

    Patches are extracted at the nearest integer coordinates then phase shifted
    such that the requested position is at the center of the patch.

    The center of an image is defined to be the position of the DC component of an
    fftshifted discrete Fourier transform.

    Parameters
    ----------
    image: torch.Tensor
        `(d, h, w)` array containing the volume.
    positions: torch.Tensor
        `(..., 3)` array of coordinates for patch centers.
    sidelength: int
        Sidelength of cubic patches extracted from `image`.
    mask : torch.Tensor
        Optional mask to apply in real space before FFT, shape (size, size)
         or broadcastable to (..., b, size, size)
    return_rfft : bool, default False
        If `True`, return the rft of the patches. It can save an FFT
         operation because the subpixel shift already requires an FFT.The returned
         rffts have their origin at (0, 0)
    decenter : bool, default False
        In case the patches are returned as rft, optionally also apply an
         additional shift of sidelength / 2 so that the real space image has its origin
         at 0. NOTE: this does not change the origin in Fourier space, the rffts
         still have their origin at (0, 0)

    Returns
    -------
    patches: torch.Tensor
        `(..., sidelength, sidelength, sidelength)` array of cropped regions from `volume`
        with their centers at `positions`.
    """
    d, h, w = image.shape
    positions, ps = einops.pack([positions], pattern='* zyx')
    b, _ = positions.shape

    # find integer positions and shifts to be applied
    integer_positions = torch.round(positions)
    shifts = integer_positions - positions

    # generate coordinate grids for sampling around each integer position
    pd, ph, pw = (sidelength, sidelength, sidelength)
    center = dft_center((pd, ph, pw), rfft=False, fftshifted=True, device=image.device)
    grid = coordinate_grid(
        image_shape=(pd, ph, pw),
        center=center,
        device=image.device
    )  # (d, h, w, 2)
    broadcastable_positions = einops.rearrange(integer_positions, 'b zyx -> b 1 1 1 zyx')
    grid = grid + broadcastable_positions  # (b, d, h, w, 3)

    # extract patches, grid sample handles boundaries
    patches = F.grid_sample(
        input=einops.repeat(image, 'd h w -> b 1 d h w', b=b),
        grid=array_to_grid_sample(grid, array_shape=(d, h, w)),
        mode='nearest',
        padding_mode='zeros',
        align_corners=True
    )
    patches = einops.rearrange(patches, 'b 1 d h w -> b d h w')

    if mask is not None:
        patches = patches * mask

    # phase shift to center images
    if return_rfft:
        patches = torch.fft.rfftn(patches, dim=(-3, -2, -1))

        if decenter:
            shifts = shifts + pw / 2

        # apply the subpixel shift
        patches = fourier_shift_dft_3d(
            dft=patches,
            image_shape=(pd, ph, pw),
            shifts=shifts,
            rfft=True,
            fftshifted=False,
        )

    else:
        patches = fourier_shift_image_3d(image=patches, shifts=shifts)

    # unpack
    [patches] = einops.unpack(patches, pattern='* t h w', packed_shapes=ps)
    return patches
