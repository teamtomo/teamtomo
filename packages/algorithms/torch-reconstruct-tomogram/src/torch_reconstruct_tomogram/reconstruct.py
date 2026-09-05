"""(sub-)tomogram reconstruction in pytorch."""

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch_fourier_rescale import fourier_rescale_rfft_2d
from torch_fourier_slice import insert_central_slices_rfft_3d_multichannel
from torch_grid_utils import fftfreq_grid
from torch_tilt_series import (
    TiltSeries,
    load_tilt_series_images,
    preprocess_tilt_series_images,
)

from torch_reconstruct_tomogram.projection import _extract_particle_tilt_series

_PAD_FACTOR = 2.0


def _writable(data):
    if isinstance(data, np.ndarray) and not data.flags.writeable:
        data = data.copy()
    return data


def _reconstruct_subvolume(
    tilt_series: TiltSeries,
    images: torch.Tensor,
    points_zyx: torch.Tensor,
    sidelength: int,
    output_pixel_spacing: float | None = None,
) -> torch.Tensor:
    """Reconstruct subvolume(s), given already-loaded images."""
    device = images.device
    input_pixel_spacing = tilt_series.pixel_spacing  # raises if unset
    if output_pixel_spacing is None:
        output_pixel_spacing = input_pixel_spacing

    points_zyx = torch.as_tensor(_writable(points_zyx), device=device).float()
    points_zyx, ps = einops.pack([points_zyx], "* zyx")

    # tomogram -> detector rotation: projection_matrices is sample -> detector
    # only, so compose tomo2sample's rotation in first. Every patch's
    # Fourier-insertion rotation must be expressed relative to the
    # tomogram frame.
    rotation_matrices = (
        tilt_series.projection_matrices[:, :3, :3] @ tilt_series.tomo2sample[:3, :3]
    )
    rotation_matrices = torch.linalg.pinv(rotation_matrices)

    sidelength_padded_output = int(_PAD_FACTOR * sidelength)
    sidelength_padded_native = max(
        1,
        round(sidelength_padded_output * output_pixel_spacing / input_pixel_spacing),
    )

    particle_tilt_series_rfft = _extract_particle_tilt_series(
        tilt_series,
        images,
        points_zyx,
        sidelength=sidelength_padded_native,
        return_rfft=True,
    )

    particle_tilt_series_rfft = torch.fft.fftshift(particle_tilt_series_rfft, dim=(-2,))

    particle_tilt_series_rfft = fourier_rescale_rfft_2d(
        dft=particle_tilt_series_rfft,
        image_shape=(sidelength_padded_native, sidelength_padded_native),
        target_shape=(sidelength_padded_output, sidelength_padded_output),
    )

    particle_tilt_series_rfft = einops.rearrange(
        particle_tilt_series_rfft,
        "n_positions n_tilts h w_rfft -> n_tilts n_positions h w_rfft",
    )

    patches_rfft, weights = insert_central_slices_rfft_3d_multichannel(
        image_rfft=particle_tilt_series_rfft,
        volume_shape=(sidelength_padded_output,) * 3,
        rotation_matrices=rotation_matrices,
        zyx_matrices=True,
        fftfreq_max=0.5,
    )

    valid_weights = weights > 1e-3
    patches_rfft[:, valid_weights] /= weights[valid_weights]

    patches_rfft = torch.fft.ifftshift(patches_rfft, dim=(-3, -2))

    patches = torch.fft.irfftn(
        patches_rfft,
        s=(sidelength_padded_output,) * 3,
        dim=(-3, -2, -1),
    )

    patches = torch.fft.ifftshift(patches, dim=(-3, -2, -1))

    grid = fftfreq_grid(
        image_shape=(sidelength_padded_output,) * 3,
        rfft=False,
        fftshift=True,
        norm=True,
        device=device,
    )
    patches = patches / torch.sinc(grid) ** 2

    p = (sidelength_padded_output - sidelength) // 2
    patches = F.pad(patches, [-p] * 6)

    [patches] = einops.unpack(patches, ps, "* d h w")

    return patches


def reconstruct_subvolume(
    tilt_series: TiltSeries,
    points_zyx: torch.Tensor,
    sidelength: int,
    output_pixel_spacing: float | None = None,
    preprocess: bool = True,
) -> torch.Tensor:
    """Reconstruct 3D patch(es) at location(s) in the sample.

    Rank-polymorphic: input (..., 3) -> output (..., d, h, w)

    - tilt_series supplies the projection geometry,
    - points_zyx are zyx coordinates in Angstroms, relative to the tomogram center
    - sidelength is the output subvolume size in voxels
    - output_pixel_spacing is the voxel size of the output in Angstroms
      (defaults to `tilt_series.pixel_spacing`); the per-tilt 2D crops are
      Fourier-rescaled to this pixel size before 3D reconstruction, so local
      (subvolume) and global (tomogram) reconstructions can each target an
      arbitrary output pixel size independent of the raw data's
    - preprocess, if True (default), applies
      `torch_tilt_series.preprocess_tilt_series_images` (plane subtraction, a
      DC-excluding bandpass with no low-pass, i.e. up to Nyquist, and
      central-crop normalization) to the loaded images before reconstruction
    """
    images = load_tilt_series_images(tilt_series)
    if preprocess:
        images = preprocess_tilt_series_images(images)
    return _reconstruct_subvolume(
        tilt_series,
        images,
        points_zyx,
        sidelength,
        output_pixel_spacing=output_pixel_spacing,
    )


def _cosine_taper_window(core_length: int, margin: int, device) -> torch.Tensor:
    """1D cosine-taper window, flat in the middle, tapered at the edges.

    1.0 over the central `core_length` samples, cosine-tapered from 0 up to 1
    (and back down to 0) over `margin` samples on each side. Total length is
    core_length + 2 * margin.
    """
    if margin == 0:
        return torch.ones(core_length, device=device)
    ramp = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, margin, device=device)))
    core = torch.ones(core_length, device=device)
    return torch.cat([ramp, core, ramp.flip(0)])


def reconstruct_tomogram(
    tilt_series: TiltSeries,
    volume_shape: tuple[int, int, int],
    sidelength: int,
    batch_size: int | None = None,
    output_pixel_spacing: float | None = None,
    preprocess: bool = True,
    blend_margin: int | None = None,
) -> torch.Tensor:
    """Reconstruct the full tomogram by tiling reconstructed patches in 3D."""
    images = load_tilt_series_images(tilt_series)
    if preprocess:
        images = preprocess_tilt_series_images(images)

    pixel_spacing = tilt_series.pixel_spacing  # raises if unset
    if output_pixel_spacing is None:
        output_pixel_spacing = pixel_spacing

    if blend_margin is None:
        blend_margin = sidelength // 4
    patch_sidelength = sidelength + 2 * blend_margin
    half = patch_sidelength // 2

    d, h, w = volume_shape
    r = sidelength // 2
    device = images.device

    z_centers = torch.arange(start=r, end=d + r, step=sidelength, device=device)
    y_centers = torch.arange(start=r, end=h + r, step=sidelength, device=device)
    x_centers = torch.arange(start=r, end=w + r, step=sidelength, device=device)
    # absolute 0-indexed voxel coordinates of each patch center
    centers_voxel = torch.stack(
        torch.meshgrid(z_centers, y_centers, x_centers, indexing="ij"), dim=-1
    )

    volume_center = torch.tensor([d, h, w], device=device) // 2
    centers_zyx_ang = (centers_voxel - volume_center) * output_pixel_spacing

    window_1d = _cosine_taper_window(sidelength, blend_margin, device="cpu")
    window_3d = (
        window_1d[:, None, None] * window_1d[None, :, None] * window_1d[None, None, :]
    )

    tomogram_sum = torch.zeros(volume_shape, dtype=torch.float32)
    weight_sum = torch.zeros(volume_shape, dtype=torch.float32)

    centers_flat, _ = einops.pack([centers_voxel], "* zyx")
    centers_ang_flat, _ = einops.pack([centers_zyx_ang], "* zyx")
    chunk_size = batch_size or len(centers_flat)

    for start in range(0, len(centers_flat), chunk_size):
        chunk_centers = centers_flat[start : start + chunk_size]
        chunk_centers_ang = centers_ang_flat[start : start + chunk_size]

        patches_batch = _reconstruct_subvolume(
            tilt_series,
            images,
            chunk_centers_ang,
            patch_sidelength,
            output_pixel_spacing=output_pixel_spacing,
        ).cpu()

        for j in range(len(patches_batch)):
            cz, cy, cx = chunk_centers[j].tolist()
            z0, y0, x0 = cz - half, cy - half, cx - half
            z1, y1, x1 = (
                z0 + patch_sidelength,
                y0 + patch_sidelength,
                x0 + patch_sidelength,
            )

            # clip the patch's placement to the volume bounds
            cz0, cy0, cx0 = max(z0, 0), max(y0, 0), max(x0, 0)
            cz1, cy1, cx1 = min(z1, d), min(y1, h), min(x1, w)
            if cz0 >= cz1 or cy0 >= cy1 or cx0 >= cx1:
                continue

            src = (
                slice(cz0 - z0, cz1 - z0),
                slice(cy0 - y0, cy1 - y0),
                slice(cx0 - x0, cx1 - x0),
            )
            dst = (slice(cz0, cz1), slice(cy0, cy1), slice(cx0, cx1))
            weight_block = window_3d[src]
            tomogram_sum[dst] += patches_batch[j][src] * weight_block
            weight_sum[dst] += weight_block

        del patches_batch
        if device.type != "cpu":
            torch.cuda.empty_cache()

    tomogram = tomogram_sum / weight_sum.clamp_min(1e-6)

    return tomogram.to(device)
