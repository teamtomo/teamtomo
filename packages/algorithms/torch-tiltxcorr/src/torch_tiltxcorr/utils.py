import einops
import torch
from torch_affine_utils.transforms_2d import R, S, T
from torch_affine_utils import homogenise_coordinates
from torch_transform_image import affine_transform_image_2d
from torch_grid_utils import rectangle
from torch_fourier_rescale import fourier_rescale_2d


def apply_stretch_perpendicular_to_tilt_axis(
    image: torch.Tensor,
    tilt_axis_angle: float,
    scale_factor: float,
) -> torch.Tensor:
    # grab image dimensions and calculate center
    h, w = image.shape[-2:]
    device = image.device
    center = torch.tensor((h // 2, w // 2), device=device, dtype=image.dtype)

    # construct transforms
    T0 = T(-1 * center, device=device)  # origin from array [0, 0] to center of image
    R0 = R(
        -1 * tilt_axis_angle, yx=True, device=device
    )  # rotate coords so tilt axis is aligned along image Y
    S0 = S(
        (1, 1 / scale_factor), device=device
    )  # scale coords in X (perpendicular to tilt axis)
    R1 = R(
        tilt_axis_angle, yx=True, device=device
    )  # rotate coords to align current Y with tilt axis in image
    T1 = T(center, device=device)  # origin back to [0, 0]

    M = T1 @ R1 @ S0 @ R0 @ T0

    # apply transform
    image = affine_transform_image_2d(
        image, matrices=M, interpolation="bicubic", yx_matrices=True
    )
    return image


def taper_image_edges(image: torch.Tensor) -> torch.Tensor:
    # calculate the size of the edge taper
    h, w = image.shape[-2:]
    # 0.1 is the fraction of padding, IMOD advises 10%
    taper_width = min(h, w) * 0.1
    taper_mask_shape = (int(h - taper_width), int(w - taper_width))

    # create edge taper mask
    edge_taper_mask = rectangle(
        dimensions=taper_mask_shape,
        image_shape=(h, w),
        smoothing_radius=taper_width / 2,
        device=image.device,
    )
    image = image * edge_taper_mask
    return image


def calculate_cross_correlation(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Calculate the 2D cross correlation between images of the same size.

    The position of the maximum relative to the center of the image gives a shift.
    This is the shift that when applied to `b` best aligns it to `a`.
    """
    h, w = a.shape[-2:]
    fta = torch.fft.rfftn(a, dim=(-2, -1))
    ftb = torch.fft.rfftn(b, dim=(-2, -1))
    result = fta * torch.conj(ftb)
    # AreTomo using some like this (filtered FFT-based approach):
    # result = result / torch.sqrt(result.abs() + .0001)
    # result = result * b_envelope(300, a.shape, 10)
    result = torch.fft.irfftn(result, dim=(-2, -1), s=(h, w))
    result = torch.fft.ifftshift(result, dim=(-2, -1))
    result /= h * w  # normalize the result
    return result


def get_shift_from_correlation_image(
    correlation_image: torch.Tensor,
) -> torch.Tensor:
    """
    Extract shift from 2D correlation image

    The shift should be applied to img2 to align with img1.
    Uses Fourier upsampling for sub-voxel accuracy: extracts a region around the
    integer peak, upsamples it using bandwidth-limited Fourier rescaling, and finds
    the peak position in the upsampled image.

    Parameters
    ----------
    correlation_image : torch.Tensor
        2D correlation image

    Returns
    -------
    torch.Tensor
        2D shift vector [z, y, x]
    """
    # fix these values for now to ensure stable behaviour
    patch_size: int = 16
    upsample_size: int = 1024

    dtype, device = correlation_image.dtype, correlation_image.device
    shape = torch.tensor(correlation_image.shape, device=device, dtype=dtype)
    center = torch.div(shape, 2, rounding_mode="floor")

    # Find integer peak location
    flat_idx = torch.argmax(correlation_image)
    peak_coords = torch.tensor(
        torch.unravel_index(flat_idx, correlation_image.shape),
        device=device,
        dtype=dtype,
    )

    half_patch = patch_size // 2

    # Check if we can extract a full patch around the peak
    if torch.any(peak_coords < half_patch) or torch.any(
        peak_coords >= shape - half_patch
    ):
        return peak_coords - center

    # Extract patch around peak
    py, px = peak_coords.int().tolist()
    patch = correlation_image[
        py - half_patch : py + half_patch,
        px - half_patch : px + half_patch,
    ]

    # Upsample using Fourier rescaling
    upsample_factor = upsample_size / patch_size
    upsampled, _ = fourier_rescale_2d(
        image=patch,
        source_spacing=upsample_factor,
        target_spacing=1.0,
    )

    # Find peak in upsampled volume
    up_flat_idx = torch.argmax(upsampled)
    up_peak_coords = torch.tensor(
        torch.unravel_index(up_flat_idx, upsampled.shape),
        device=device,
        dtype=dtype,
    )

    # Convert upsampled peak position back to original coordinates
    up_center = upsample_size / 2
    offset = (up_peak_coords - up_center) / upsample_factor
    subpixel_peak = peak_coords + offset

    return subpixel_peak - center


def transform_shifts_from_stretched_images(
    shift: torch.Tensor,  # (b, 2) or (2,)
    tilt_axis_angle: float,
    scale_factor: torch.Tensor | float,  # (b,) or scalar
) -> torch.Tensor:
    device = shift.device

    # to homogenous coordinate vectors: (b, 2) -> (b, 3, 1)
    shift_yxw = homogenise_coordinates(shift)  # (b, 3)
    shift_yxw = einops.rearrange(shift_yxw, "b yxw -> b yxw 1")  # (b, 3, 1)

    # Compose transforms - rotation matrices are the same for all in batch
    R0 = R(-1 * tilt_axis_angle, yx=True, device=device)  # (3, 3)
    R1 = R(tilt_axis_angle, yx=True, device=device)  # (3, 3)

    # Create scale matrices
    # S expects (y_scale, x_scale), and we scale x by 1/scale_factor
    scale_pairs = einops.rearrange(
        [
            torch.ones_like(scale_factor, device=device),  # y scale
            1.0 / scale_factor,  # x scale
        ],
        "yx ... -> ... yx",
    )  # (..., 2)
    S0 = S(scale_pairs)

    # Compose: M = R1 @ S0 @ R0
    M = R1 @ S0 @ R0  # (b, 3, 3)

    # Apply batched matrix multiplication
    transformed_shift = M @ shift_yxw  # (b, 3, 3) @ (b, 3, 1) = (b, 3, 1)
    transformed_shift = transformed_shift[:, :2, 0]  # (b, 2)
    return transformed_shift


def normalise_in_mask_area(image, mask):
    n = torch.sum(mask)
    mean = torch.sum(image * mask) / n
    std = (torch.sum(image**2 * mask) / n - mean**2) ** 0.5
    image = (image - mean) / std
    return image
