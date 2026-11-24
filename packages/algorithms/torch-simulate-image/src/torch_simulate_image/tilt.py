"""Tilt a specimen volume the way a microscope stage does."""

from typing import Literal

import torch
from torch_affine_utils.transforms_3d import Ry, Rz, T
from torch_grid_utils import dft_center
from torch_transform_image import affine_transform_image_3d

__all__ = ["tilt_volume"]


def tilt_volume(
    volume: torch.Tensor,
    tilt_deg: float | int,
    detector_rotation_deg: float | int = 0.0,
    fill_value: float = 0.0,
    interpolation: Literal["trilinear", "nearest"] = "trilinear",
) -> torch.Tensor:
    """Tilt a specimen volume about the microscope tilt axis.

    Applies ``Rz(detector_rotation) @ Ry(tilt)`` as a single resample about the
    volume centre. The tilt axis is y, as in ``torch_tilt_series.TiltSeries``,
    so a reconstruction of the simulated tilt series shares a frame with the
    input volume.

    Parameters
    ----------
    volume : torch.Tensor
        Specimen volume, shape `(d, h, w)` in `(z, y, x)` order. A leading
        dimension is treated as channels by the underlying resampler and comes
        back last, `(d, h, w, c)`; prefer one volume per call.
    tilt_deg : float | int
        Stage tilt about the y axis, in degrees.
    detector_rotation_deg : float | int
        In-plane rotation from the microscope frame to the detector frame, in
        degrees, about the beam (z) axis. This is the tilt-axis angle recorded
        in an mdoc, measured from +y. Defaults to 0, i.e. detector rows aligned
        with the microscope.
    fill_value : float
        Value given to samples pulled from outside the volume. Tilting a finite
        box brings such samples into view at high tilt; for a solvated specimen
        set this to the bulk solvent potential so the padding is ice rather than
        vacuum.
    interpolation : {"trilinear", "nearest"}
        Sampling mode.

    Returns
    -------
    torch.Tensor
        Tilted volume, shape `(d, h, w)`.

    Examples
    --------
    A tilt series of an ice-embedded specimen, detector 5 degrees off the
    microscope tilt axis:

    >>> for tilt in range(-60, 63, 3):
    ...     tilted = tilt_volume(
    ...         volume,
    ...         tilt_deg=tilt,
    ...         detector_rotation_deg=-5.0,
    ...         fill_value=3.6,
    ...     )
    """
    device = volume.device
    d, h, w = volume.shape[-3:]
    center = dft_center(image_shape=(d, h, w), device=device, fftshift=True, rfft=False)

    # Rz(detector) @ Ry(tilt): the stage tilt, then into the detector frame.
    # One 4x4, so one interpolation.
    rotation = Rz(float(detector_rotation_deg), zyx=True, device=device) @ Ry(
        float(tilt_deg), zyx=True, device=device
    )
    matrix = torch.inverse(
        T(center, device=device) @ rotation @ T(-center, device=device)
    )

    if fill_value != 0.0:
        # out-of-bounds samples come back as zero, so offset the volume, rotate,
        # and restore -- leaving the padding at `fill_value`
        rotated = affine_transform_image_3d(
            image=volume - fill_value,
            matrices=matrix,
            interpolation=interpolation,
            zyx_matrices=True,
        )
        return rotated + fill_value

    return affine_transform_image_3d(
        image=volume,
        matrices=matrix,
        interpolation=interpolation,
        zyx_matrices=True,
    )
