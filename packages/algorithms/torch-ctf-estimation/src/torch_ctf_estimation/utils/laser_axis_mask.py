"""Utility to build a 2D rFFT strip mask along the laser axis (or axes)."""

from __future__ import annotations

import math

import torch
from torch_grid_utils import image_shape_from_dft_shape


def build_laser_axis_mask(
    image_shape: tuple[int, int],
    laser_xy_angle_deg: float,
    dual_laser: bool,
    mask_width: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a 2D rFFT-layout mask that zeros strips along the laser axis.

    A strip through the FFT origin at ``laser_xy_angle_deg`` is zeroed out.
    When ``dual_laser`` is True a second orthogonal strip at
    ``laser_xy_angle_deg + 90°`` is also zeroed.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Real-space patch shape ``(H, W)``. The mask will have shape
        ``(H, W // 2 + 1)`` matching the rFFT output.
    laser_xy_angle_deg : float
        Angle of the first laser axis in degrees, measured from the +ky axis
        (row direction) towards +kx: 0 puts the axis along ky, 90 along kx. This
        is the convention of the LPP model in torch-ctf
        (:func:`torch_ctf.ctf_lpp.make_laser_coords`), so the mask covers the
        same strip the modelled laser does.
    dual_laser : bool
        If True, also mask the orthogonal axis at ``laser_xy_angle_deg + 90°``.
    mask_width : float
        Width of each strip in fftfreq units (range [0, 0.5]).  A value of 0.1
        masks ±0.05 around the axis, corresponding to 10 % of the image
        dimension.
    device : torch.device | None, optional
        Device for the output tensor. Defaults to CPU.

    Returns
    -------
    torch.Tensor
        Float mask of shape ``(H, W // 2 + 1)``.  Pixels inside a masked strip
        are 0.0; all others are 1.0.
    """
    h, w = image_shape
    half_width = mask_width / 2.0

    ky = torch.fft.fftfreq(h, device=device).reshape(-1, 1)  # pylint: disable=not-callable
    kx = torch.fft.rfftfreq(w, device=device).reshape(1, -1)  # pylint: disable=not-callable

    def _distance_to_axis(angle_deg: float) -> torch.Tensor:
        # The axis at angle θ from +ky towards +kx runs along
        # (ky, kx) = (cos θ, sin θ); its normal is (-sin θ, cos θ).
        theta = math.radians(angle_deg)
        return torch.abs(kx * math.cos(theta) - ky * math.sin(theta))

    mask = _distance_to_axis(laser_xy_angle_deg) >= half_width
    if dual_laser:
        mask = mask & (_distance_to_axis(laser_xy_angle_deg + 90.0) >= half_width)

    return mask.float()


def apply_laser_axis_mask(
    patch_ps: torch.Tensor,
    mean_ps: torch.Tensor,
    laser_xy_angle_deg: float,
    dual_laser: bool,
    mask_width: float,
    image_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero FFT strips along the laser axis on patch and mean power spectra.

    Parameters
    ----------
    patch_ps : torch.Tensor
        Patch power spectra, last two dims are rFFT ``(ph, pw_rfft)``.
    mean_ps : torch.Tensor
        Mean power spectrum, shape ``(ph, pw_rfft)``.
    laser_xy_angle_deg : float
        Laser axis angle in degrees.
    dual_laser : bool
        If True, also mask the orthogonal axis.
    mask_width : float
        Strip width in fftfreq units.
    image_shape : tuple[int, int] or None, optional
        Real-space ``(h, w)`` of the patches. If None, square patches
        ``(ph, ph)`` are assumed; a ValueError is raised if that does not
        match the rfft width.

    Returns
    -------
    patch_ps : torch.Tensor
        Masked patch power spectra.
    mean_ps : torch.Tensor
        Masked mean power spectrum.
    axis_mask : torch.Tensor
        The applied mask, shape ``(ph, pw_rfft)``.
    """
    real_h, real_w = image_shape_from_dft_shape(
        patch_ps.shape[-2:], rfft=True, image_shape=image_shape, assume_square=True
    )
    axis_mask = build_laser_axis_mask(
        image_shape=(real_h, real_w),
        laser_xy_angle_deg=laser_xy_angle_deg,
        dual_laser=dual_laser,
        mask_width=mask_width,
        device=patch_ps.device,
    )
    return patch_ps * axis_mask, mean_ps * axis_mask, axis_mask
