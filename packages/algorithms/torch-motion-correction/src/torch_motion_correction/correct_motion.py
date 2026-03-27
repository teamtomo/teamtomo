"""Correct movie motion using a deformation field."""

import einops
import torch
import torch.nn.functional as F
from torch_cubic_spline_grids import CubicBSplineGrid3d, CubicCatmullRomGrid3d
from torch_fourier_shift import fourier_shift_dft_2d
from torch_grid_utils import coordinate_grid
from torch_image_interpolation import sample_image_2d
from torch_image_interpolation.grid_sample_utils import array_to_grid_sample

from torch_motion_correction.deformation_field_utils import (
    evaluate_deformation_field,
    evaluate_deformation_field_at_t,
)


def correct_motion(
    image: torch.Tensor,  # (t, h, w)
    deformation_grid: torch.Tensor,  # (yx, t, h, w)
    pixel_spacing: float,
    grad: bool = False,
    grid_type: str = "catmull_rom",
    device: torch.device = None,
) -> torch.Tensor:
    """Correct movie motion using a deformation field.

    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    deformation_grid: torch.Tensor
        (yx, t, h, w) deformation grid
    pixel_spacing: float
        Pixel spacing in Angstroms
    grad: bool
        Whether to enable gradients. Default is False.
    grid_type: str
        Type of grid to use ('catmull_rom' or 'bspline'). Default is 'catmull_rom'.
    device: torch.device, optional
        Device for computation. Default is None, which uses the device of the
        input image.

    Returns
    -------
    corrected_frames: torch.Tensor
        (t, h, w) corrected images
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        deformation_grid = deformation_grid.to(device)

    t, _, _ = image.shape
    _, _, gh, gw = deformation_grid.shape
    normalized_t = torch.linspace(0, 1, steps=t, device=image.device)

    # Use conditional gradient context to save memory
    gradient_context = torch.enable_grad() if grad else torch.no_grad()

    with gradient_context:
        # correct motion in each frame
        corrected_frames = [
            _correct_frame(
                frame=frame,
                frame_deformation_grid=evaluate_deformation_field_at_t(
                    deformation_field=deformation_grid,
                    t=frame_t,
                    grid_shape=(10 * gh, 10 * gw),
                    grid_type=grid_type,
                ),
                pixel_spacing=pixel_spacing,
            )
            for frame, frame_t in zip(image, normalized_t)
        ]
    corrected_frames = torch.stack(corrected_frames, dim=0).detach()
    return corrected_frames  # (t, h, w)


def _correct_frame(
    frame: torch.Tensor,
    pixel_spacing: float,
    frame_deformation_grid: torch.Tensor,  # (yx, h, w)
) -> torch.Tensor:
    """Correct a single frame using a deformation grid.

    Parameters
    ----------
    frame: torch.Tensor
        (h, w) frame to correct
    pixel_spacing: float
        Pixel spacing in Angstroms
    frame_deformation_grid: torch.Tensor
        (yx, h, w) deformation grid

    Returns
    -------
    corrected_frame: torch.Tensor
        (h, w) corrected frame
    """
    # grab frame and deformation grid dimensions
    h, w = frame.shape

    # prepare a grid of pixel positions
    pixel_grid = coordinate_grid(
        image_shape=(h, w),
        device=frame.device,
    )  # (h, w, 2) yx coords

    pixel_shifts = get_pixel_shifts(
        frame=frame,
        pixel_spacing=pixel_spacing,
        frame_deformation_grid=frame_deformation_grid,
        pixel_grid=pixel_grid,
    )  # (h, w, yx)

    # todo: make sure semantics around deformation field interpolants
    # (i.e. spatiotemporally resolved shifts) are crystal clear
    deformed_pixel_coords = pixel_grid + pixel_shifts

    # sample original image data
    corrected_frame = sample_image_2d(
        image=frame,
        coordinates=deformed_pixel_coords,
        interpolation="bicubic",
    )

    return corrected_frame


def get_pixel_shifts(
    frame: torch.Tensor,
    pixel_spacing: float,
    frame_deformation_grid: torch.Tensor,
    pixel_grid: torch.Tensor,
) -> torch.Tensor:
    """
    Get pixel shifts from a deformation grid.

    Parameters
    ----------
    frame: torch.Tensor
        (h, w) frame to correct
    pixel_spacing: float
        Pixel spacing in Angstroms
    frame_deformation_grid: torch.Tensor
        (yx, h, w) deformation grid
    pixel_grid: torch.Tensor
        (h, w, 2) pixel grid

    Returns
    -------
    pixel_shifts: torch.Tensor
        (h, w, yx) pixel shifts
    """
    # grab frame and deformation grid dimensions
    h, w = frame.shape
    _, gh, gw = frame_deformation_grid.shape

    # interpolate oversampled per frame deformation grid at each pixel position
    image_dim_lengths = torch.as_tensor(
        [h - 1, w - 1], device=frame.device, dtype=torch.float32
    )
    deformation_grid_dim_lengths = torch.as_tensor(
        [gh - 1, gw - 1], device=frame.device, dtype=torch.float32
    )
    normalized_pixel_grid = pixel_grid / image_dim_lengths
    deformation_grid_interpolants = normalized_pixel_grid * deformation_grid_dim_lengths
    deformation_grid_interpolants = array_to_grid_sample(
        deformation_grid_interpolants, array_shape=(gh, gw)
    )  # (gh, gw, xy)
    shifts_angstroms = F.grid_sample(
        input=einops.rearrange(frame_deformation_grid, "yx h w -> 1 yx h w"),
        grid=einops.rearrange(deformation_grid_interpolants, "h w xy -> 1 h w xy"),
        mode="bicubic",
        padding_mode="reflection",
        align_corners=True,
    )  # (b, yx, h, w)

    pixel_shifts = shifts_angstroms / pixel_spacing
    # find pixel positions to sample image data at, accounting for deformations
    pixel_shifts = einops.rearrange(pixel_shifts, "1 yx h w -> h w yx")

    return pixel_shifts


def correct_motion_two_grids(
    image: torch.Tensor,  # (t, h, w)
    new_deformation_grid: CubicCatmullRomGrid3d
    | CubicBSplineGrid3d,  # CubicCatmullRomGrid3d - optimizable with gradients
    base_deformation_grid: CubicCatmullRomGrid3d
    | CubicBSplineGrid3d,  # CubicCatmullRomGrid3d - frozen base grid
    pixel_spacing: float,
    grad: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """Correct movie motion using two deformation grids.

    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    new_deformation_grid: CubicCatmullRomGrid3d | CubicBSplineGrid3d
        Optimizable deformation grid with gradients
    base_deformation_grid: CubicCatmullRomGrid3d | CubicBSplineGrid3d
        Frozen base deformation grid
    pixel_spacing: float
        Pixel spacing in Angstroms
    grad: bool
        Whether to enable gradients. Default is True.
    device: torch.device, optional
        Device for computation. Default is None, which uses the device of the
        input image.

    Returns
    -------
    corrected_frames: torch.Tensor
        (t, h, w) corrected images
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)

    t, _, _ = image.shape

    # Get grid resolution from new_deformation_grid
    _, _, gh, gw = new_deformation_grid.data.shape

    normalized_t = torch.linspace(0, 1, steps=t, device=device)

    # Use conditional gradient context
    gradient_context = torch.enable_grad() if grad else torch.no_grad()

    with gradient_context:
        # Correct motion in each frame by evaluating both grids
        corrected_frames = []

        for _, (frame, frame_t) in enumerate(zip(image, normalized_t)):
            corrected_frame = _correct_frame_two_grids(
                frame=frame,
                new_grid=new_deformation_grid,
                base_grid=base_deformation_grid,
                frame_t=frame_t,
                grid_shape=(10 * gh, 10 * gw),
                pixel_spacing=pixel_spacing,
            )
            corrected_frames.append(corrected_frame)

    corrected_frames = torch.stack(corrected_frames, dim=0)

    return corrected_frames  # (t, h, w)


def _correct_frame_two_grids(
    frame: torch.Tensor,
    new_grid: CubicCatmullRomGrid3d
    | CubicBSplineGrid3d,  # CubicCatmullRomGrid3d with gradients
    base_grid: CubicCatmullRomGrid3d
    | CubicBSplineGrid3d,  # CubicCatmullRomGrid3d frozen
    frame_t: float,
    grid_shape: tuple[int, int],
    pixel_spacing: float,
) -> torch.Tensor:
    """Correct a single frame using two deformation grids.

    Parameters
    ----------
    frame: torch.Tensor
        (h, w) frame to correct
    new_grid: CubicCatmullRomGrid3d | CubicBSplineGrid3d
        Optimizable deformation grid with gradients
    base_grid: CubicCatmullRomGrid3d | CubicBSplineGrid3d
        Frozen base deformation grid
    frame_t: float
        Timepoint to evaluate at [0, 1]
    grid_shape: tuple[int, int]
        (h, w) shape of the grid to evaluate at
    pixel_spacing: float
        Pixel spacing in Angstroms

    Returns
    -------
    corrected_frame: torch.Tensor
        (h, w) corrected frame
    """
    h, w = grid_shape

    # Create normalized coordinate grid for this timepoint
    y = torch.linspace(0, 1, steps=h, device=frame.device)
    x = torch.linspace(0, 1, steps=w, device=frame.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yx_grid = einops.rearrange([yy, xx], "yx h w -> (h w) yx")
    tyx_grid = F.pad(yx_grid, (1, 0), value=frame_t)  # (h*w, 3)

    # Evaluate both grids at these coordinates
    # new_grid is called directly (preserves gradients!)
    new_shifts = new_grid(tyx_grid)  # (h*w, 2) with gradients

    # base_grid is also called directly but we detach (no gradients needed)
    base_shifts = base_grid(tyx_grid).detach()  # (h*w, 2) no gradients

    # Combine shifts (addition preserves gradients from new_shifts)
    combined_shifts = new_shifts + base_shifts  # (h*w, 2)

    # Reshape back to spatial grid
    combined_shifts = einops.rearrange(combined_shifts, "(h w) yx -> yx h w", h=h, w=w)

    # Now apply the combined shifts to the frame
    corrected_frame = _correct_frame(
        frame=frame,
        frame_deformation_grid=combined_shifts,
        pixel_spacing=pixel_spacing,
    )

    return corrected_frame


def correct_motion_slow(
    image: torch.Tensor,
    deformation_grid: torch.Tensor,
    grad: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """Correct movie motion using a deformation field.

    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    deformation_grid: torch.Tensor
        (yx, t, h, w) deformation grid
    grad: bool
        Whether to enable gradients. Default is False.
    device: torch.device, optional
        Device for computation. Default is None, which uses the device of the
        input image.

    Returns
    -------
    corrected_frames: torch.Tensor
        (t, h, w) corrected images
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        deformation_grid = deformation_grid.to(device)

    t, _, _ = image.shape
    normalized_t = torch.linspace(0, 1, steps=t, device=image.device)

    # Use conditional gradient context to save memory
    gradient_context = torch.enable_grad() if grad else torch.no_grad()

    with gradient_context:
        # correct motion in each frame
        corrected_frames = [
            _correct_frame_slow(
                frame=frame,
                deformation_grid=deformation_grid,
                t=frame_t,
            )
            for frame, frame_t in zip(image, normalized_t)
        ]
    corrected_frames = torch.stack(corrected_frames, dim=0).detach()
    return corrected_frames  # (t, h, w)


def _correct_frame_slow(
    frame: torch.Tensor,
    deformation_grid: torch.Tensor,
    t: float,  # [0, 1]
) -> torch.Tensor:
    """Correct a single frame using a deformation grid.

    Parameters
    ----------
    frame: torch.Tensor
        (h, w) frame to correct
    deformation_grid: torch.Tensor
        (yx, h, w) deformation grid
    t: float
        Timepoint to evaluate at [0, 1]

    Returns
    -------
    corrected_frame: torch.Tensor
        (h, w) corrected frame
    """
    # grab frame dimensions
    h, w = frame.shape

    # prepare a grid of pixel positions
    pixel_grid = coordinate_grid(
        image_shape=(h, w),
        device=frame.device,
    )  # (h, w, 2) yx coords

    dim_lengths = torch.as_tensor(
        [h - 1, w - 1], device=frame.device, dtype=torch.float32
    )
    normalized_pixel_grid = pixel_grid / dim_lengths

    # add normalized time coordinate to every pixel coordinate
    # (h, w, 2) -> (h, w, 3)
    # yx -> tyx
    tyx = F.pad(normalized_pixel_grid, pad=(1, 0), value=t)

    # evaluate interpolated shifts at every pixel
    shifts_px = evaluate_deformation_field(
        deformation_field=deformation_grid,
        tyx=tyx,
    )

    # find pixel positions to sample image data at, accounting for deformations
    deformed_pixel_coords = pixel_grid + shifts_px

    # sample original image data
    corrected_frame = sample_image_2d(
        image=frame,
        coordinates=deformed_pixel_coords,
        interpolation="bicubic",
    )

    return corrected_frame


def correct_motion_fast(
    image: torch.Tensor,
    deformation_grid: torch.Tensor,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Fast motion correction for single patch deformation fields using FFT.

    This function checks that the deformation field represents a single patch
    (final two dimensions are both 1) and uses FFT-based correction with
    fourier_shift_dft_2d for efficiency.

    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    deformation_grid: torch.Tensor
        Deformation field (2, t, 1, 1) for single patch correction
    device: torch.device, optional
        Device for computation

    Returns
    -------
    corrected_frames: torch.Tensor
        (t, h, w) corrected images
    """
    if device is None:
        device = image.device
    else:
        image = image.to(device)
        deformation_grid = deformation_grid.to(device)

    # Check that deformation field has single patch dimensions
    if deformation_grid.shape[-2:] != (1, 1):
        raise ValueError(
            f"Expected single patch deformation field with shape (2, t, 1, 1), "
            f"but got shape {deformation_grid.shape}. "
            f"Final two dimensions must be (1, 1) for single patch correction."
        )

    t, h, w = image.shape

    # Extract shifts from deformation field (2, t, 1, 1) -> (t, 2)
    shifts = einops.rearrange(deformation_grid, "c t 1 1 -> t c")
    shifts *= -1  # flip for phase shift

    print(f"Single patch correction: applying shifts to {t} frames")
    print(
        f"Shift range: y=[{shifts[:, 0].min():.2f}, {shifts[:, 0].max():.2f}], "
        f"x=[{shifts[:, 1].min():.2f}, {shifts[:, 1].max():.2f}] pixels"
    )

    # Convert image to frequency domain for efficient shifting

    image_fft = torch.fft.rfftn(image, dim=(-2, -1))  # (t, h, w_freq)

    # Apply shifts using fourier_shift_dft_2d

    shifted_fft = fourier_shift_dft_2d(
        dft=image_fft,
        image_shape=(h, w),
        shifts=shifts,  # (t, 2) shifts
        rfft=True,
        fftshifted=False,
    )

    corrected_frames = torch.fft.irfftn(shifted_fft, s=(h, w))

    return corrected_frames
