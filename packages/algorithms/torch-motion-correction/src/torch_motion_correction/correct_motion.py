"""Correct movie motion using a deformation field."""

import einops
import torch
import torch.nn.functional as F
from torch_fourier_shift import fourier_shift_dft_2d
from torch_grid_utils import coordinate_grid
from torch_image_interpolation import sample_image_2d
from torch_image_interpolation.grid_sample_utils import array_to_grid_sample

from torch_motion_correction.deformation_field import DeformationField


def correct_motion(
    image: torch.Tensor,  # (t, h, w)
    deformation_field: DeformationField,
    pixel_spacing: float,
    grad: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """Correct movie motion using a deformation field.

    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    deformation_field: DeformationField
        Spatiotemporal deformation field with shape (2, nt, nh, nw) in Angstroms.
        The ``grid_type`` attribute of the
        :class:`~torch_motion_correction.types.DeformationField` controls the
        interpolation method.
    pixel_spacing: float
        Pixel spacing in Angstroms
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

    image = image.to(device)
    deformation_field = deformation_field.to(device)

    t, _, _ = image.shape
    _, _, gh, gw = deformation_field.data.shape
    normalized_t = torch.linspace(0, 1, steps=t, device=image.device)

    # Use conditional gradient context to save memory
    gradient_context = torch.enable_grad() if grad else torch.no_grad()

    with gradient_context:
        # correct motion in each frame
        corrected_frames = [
            _correct_frame(
                frame=frame,
                frame_deformation_grid=deformation_field.evaluate_at_t(
                    t=frame_t,
                    grid_shape=(10 * gh, 10 * gw),
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

    # TODO: make sure semantics around deformation field interpolants
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
    """Get pixel shifts from a deformation grid.

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
    new_deformation_field: DeformationField,  # optimizable
    base_deformation_field: DeformationField,  # frozen ref
    pixel_spacing: float,
    grad: bool = True,
    device: torch.device = None,
) -> torch.Tensor:
    """Correct movie motion using two deformation fields.

    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    new_deformation_field: DeformationField
        Optimizable deformation field with gradients. Created via
        :meth:`DeformationField.initialize_optimization_grids`.
    base_deformation_field: DeformationField
        Frozen base deformation field.
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

    image = image.to(device)
    new_deformation_field = new_deformation_field.to(device)
    base_deformation_field = base_deformation_field.to(device)

    t, _, _ = image.shape

    # Derive oversampled grid resolution from the deformation field
    _, _nt, nh, nw = new_deformation_field.data.shape
    gh, gw = nh, nw

    normalized_t = torch.linspace(0, 1, steps=t, device=device)

    gradient_context = torch.enable_grad() if grad else torch.no_grad()

    with gradient_context:
        corrected_frames = [
            _correct_frame_two_grids(
                frame=frame,
                new_field=new_deformation_field,
                base_field=base_deformation_field,
                frame_t=frame_t,
                grid_shape=(10 * gh, 10 * gw),
                pixel_spacing=pixel_spacing,
            )
            for frame, frame_t in zip(image, normalized_t)
        ]

    corrected_frames = torch.stack(corrected_frames, dim=0)

    return corrected_frames  # (t, h, w)


def _correct_frame_two_grids(
    frame: torch.Tensor,
    new_field: DeformationField,  # optimizable, gradients preserved
    base_field: DeformationField,  # frozen reference
    frame_t: float,
    grid_shape: tuple[int, int],
    pixel_spacing: float,
) -> torch.Tensor:
    """Correct a single frame using two deformation fields.

    Parameters
    ----------
    frame: torch.Tensor
        (h, w) frame to correct
    new_field: DeformationField
        Optimizable deformation field with gradients
    base_field: DeformationField
        Frozen base deformation field
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

    y = torch.linspace(0, 1, steps=h, device=frame.device)
    x = torch.linspace(0, 1, steps=w, device=frame.device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yx_grid = einops.rearrange([yy, xx], "yx h w -> (h w) yx")
    tyx_grid = F.pad(yx_grid, (1, 0), value=frame_t)  # (h*w, 3)

    new_shifts = new_field(tyx_grid)  # (h*w, 2) with gradients
    base_shifts = base_field(tyx_grid).detach()  # (h*w, 2) no gradients

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
    deformation_field: DeformationField,
    grad: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """Correct movie motion using a deformation field.

    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    deformation_field: DeformationField
        Spatiotemporal deformation field with shape (2, nt, nh, nw) in Angstroms.
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

    image = image.to(device)
    deformation_field = deformation_field.to(device)

    t, _, _ = image.shape
    normalized_t = torch.linspace(0, 1, steps=t, device=image.device)

    # Use conditional gradient context to save memory
    gradient_context = torch.enable_grad() if grad else torch.no_grad()

    with gradient_context:
        # correct motion in each frame
        corrected_frames = [
            _correct_frame_slow(
                frame=frame,
                deformation_grid=deformation_field.data,
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
    shifts_px = DeformationField(data=deformation_grid)(tyx)

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
    deformation_field: DeformationField,
    device: torch.device = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Fast motion correction for single-patch deformation fields using FFT.

    Requires a single-patch deformation field (final two spatial dimensions are
    both 1) and applies FFT-based whole-frame shifts via ``fourier_shift_dft_2d``.

    Parameters
    ----------
    image: torch.Tensor
        (t, h, w) array of images to motion correct
    deformation_field: DeformationField
        Deformation field with shape (2, t, 1, 1) in Angstroms.
    device: torch.device, optional
        Device for computation
    verbose: bool
        Whether to print progress information. Default is False.

    Returns
    -------
    corrected_frames: torch.Tensor
        (t, h, w) corrected images
    """
    if device is None:
        device = image.device

    image = image.to(device)
    deformation_field = deformation_field.to(device)

    # Check that deformation field has single patch dimensions
    if deformation_field.data.shape[-2:] != (1, 1):
        raise ValueError(
            f"Expected single patch deformation field with shape (2, t, 1, 1), "
            f"but got shape {deformation_field.data.shape}. "
            f"Final two dimensions must be (1, 1) for single patch correction."
        )

    t, h, w = image.shape

    # Extract shifts from deformation field (2, t, 1, 1) -> (t, 2)
    shifts = einops.rearrange(deformation_field.data, "c t 1 1 -> t c")
    shifts *= -1  # flip for phase shift

    if verbose:
        print(f"Single patch correction: applying shifts to {t} frames")
        print(
            f"Shift range: y=[{shifts[:, 0].min():.2f}, {shifts[:, 0].max():.2f}], "
            f"x=[{shifts[:, 1].min():.2f}, {shifts[:, 1].max():.2f}] pixels"
        )

    image_fft = torch.fft.rfftn(image, dim=(-2, -1))  # (t, h, w_freq)

    shifted_fft = fourier_shift_dft_2d(
        dft=image_fft,
        image_shape=(h, w),
        shifts=shifts,  # (t, 2) shifts
        rfft=True,
        fftshifted=False,
    )

    corrected_frames = torch.fft.irfftn(shifted_fft, s=(h, w))

    return corrected_frames
