"""Utility to build a 2D rFFT strip mask along the laser axis (or axes)."""

from __future__ import annotations

import math

import torch


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
        Angle of the first laser axis in degrees, measured from the x-axis
        (column direction) in the image plane.
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

    theta = math.radians(laser_xy_angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Distance from each frequency pixel to the line through origin at angle θ.
    # The perpendicular direction to (cos θ, sin θ) is (-sin θ, cos θ), so the
    # signed distance is: kx * (-sin θ) + ky * cos θ  (using kx for columns).
    dist1 = torch.abs(ky * cos_t - kx * sin_t)
    mask = dist1 >= half_width

    if dual_laser:
        theta2 = theta + math.pi / 2.0
        cos_t2 = math.cos(theta2)
        sin_t2 = math.sin(theta2)
        dist2 = torch.abs(ky * cos_t2 - kx * sin_t2)
        mask = mask & (dist2 >= half_width)

    return mask.float()


def apply_laser_axis_mask(
    patch_ps: torch.Tensor,
    mean_ps: torch.Tensor,
    laser_xy_angle_deg: float,
    dual_laser: bool,
    mask_width: float,
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

    Returns
    -------
    patch_ps : torch.Tensor
        Masked patch power spectra.
    mean_ps : torch.Tensor
        Masked mean power spectrum.
    axis_mask : torch.Tensor
        The applied mask, shape ``(ph, pw_rfft)``.
    """
    axis_mask = build_laser_axis_mask(
        image_shape=(patch_ps.shape[-2], (patch_ps.shape[-1] - 1) * 2),
        laser_xy_angle_deg=laser_xy_angle_deg,
        dual_laser=dual_laser,
        mask_width=mask_width,
        device=patch_ps.device,
    )
    return patch_ps * axis_mask, mean_ps * axis_mask, axis_mask
