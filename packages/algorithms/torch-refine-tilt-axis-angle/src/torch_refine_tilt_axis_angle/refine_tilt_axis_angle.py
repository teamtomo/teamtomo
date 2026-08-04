"""Refine an initial tilt axis angle."""

import torch


def refine_tilt_axis_angle(
    tilt_series: torch.Tensor,
    tilt_axis_angle: float = 90.0,
    coarse_angle_step: float = 0.5,
    min_fraction_of_nyquist: float = 0.08,
    max_fraction_of_nyquist: float = 0.95,
    refine: bool = True,
    refine_range: float = 3.0,
    refine_angle_step: float = 0.1,
) -> float:
    """Find the tilt-axis angle of a tilt series from its common line.

    All images of a translationally-aligned tilt series share Fourier
    information along a common line through the origin, perpendicular to
    the tilt axis. Coherently summing the images' Fourier transforms makes
    this line stand out as a ridge of high power, so the tilt-axis angle
    can be recovered by finding the ridge's orientation via a two-stage
    (coarse, then fine) angular grid search, evaluated as batched tensor
    operations across all candidate angles at once.

    Rectangular images are handled natively, via a normalized frequency
    shared by both axes and converted to per-axis pixel indices only at
    lookup time.

    Parameters
    ----------
    tilt_series : torch.Tensor
        Tensor containing the tilt series images with shape
        `(n_tilts, h, w)`. Should already be translationally aligned and,
        ideally, ramp- and bandpass-filtered. A pixel size of 10A is
        recommended.
    tilt_axis_angle : float, default=90.0
        Initial guess for the tilt axis angle in degrees. The search range
        is `[tilt_axis_angle - 90, tilt_axis_angle + 90]`. The default of
        90.0 searches the full `[0, 180]` range.
    coarse_angle_step : float, default=0.5
        Step size for the coarse grid search (degrees).
    min_fraction_of_nyquist : float, default=0.08
        Minimum frequency radius searched, as a fraction of Nyquist.
    max_fraction_of_nyquist : float, default=0.95
        Maximum frequency radius searched, as a fraction of Nyquist.
    refine : bool, default=True
        Whether to run a finer search around the coarse optimum.
    refine_range : float, default=3.0
        Range around the coarse optimum searched during refinement
        (degrees).
    refine_angle_step : float, default=0.1
        Step size for the fine grid search (degrees).

    Returns
    -------
    float
        The tilt axis angle in degrees, in
        `[tilt_axis_angle - 90, tilt_axis_angle + 90]`.
    """
    _, h, w = tilt_series.shape
    device = tilt_series.device

    # Hann taper: suppresses the FFT edge artifact from non-periodic boundaries.
    mask = torch.outer(
        torch.hann_window(h, periodic=False, device=device),
        torch.hann_window(w, periodic=False, device=device),
    )

    # coherent complex rfft sum across the stack
    windowed = (tilt_series - tilt_series.mean(dim=(-2, -1), keepdim=True)) * mask
    power_sum = torch.fft.rfft2(windowed).sum(dim=0).abs() ** 2

    # Shared normalized-frequency sample points, converted to per-axis pixel
    # indices only at lookup time. Sampled at the longer dimension's density.
    n_samples = max(h, w) // 2
    rhos = torch.linspace(
        min_fraction_of_nyquist * 0.5,
        max_fraction_of_nyquist * 0.5,
        n_samples,
        device=device,
    )

    # the tilt axis is perpendicular to the direction of the common line
    common_line_angle = tilt_axis_angle - 90.0

    # coarse grid search within +/-90 deg of the initial guess
    coarse_angles = torch.arange(
        common_line_angle - 90.0,
        common_line_angle + 90.0,
        coarse_angle_step,
        device=device,
    )
    image_shape = (h, w)
    power = _common_line_power(coarse_angles, rhos, power_sum, image_shape)
    best_angle = float(coarse_angles[torch.argmax(power)])

    if refine:
        # fine grid search around the coarse optimum
        fine_angles = torch.arange(
            best_angle - refine_range,
            best_angle + refine_range + refine_angle_step,
            refine_angle_step,
            device=device,
        )
        power = _common_line_power(fine_angles, rhos, power_sum, image_shape)
        best_angle = float(fine_angles[torch.argmax(power)])

    return best_angle + 90.0


def _common_line_power(
    angles_deg: torch.Tensor,
    rhos: torch.Tensor,
    power_sum: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    """Total power along the line through the origin at each angle.

    Looks up only the stored (non-negative-frequency) half of the rfft
    spectrum: for directions with `cos(theta) < 0`, the conjugate-symmetric
    point `F(-fy, -fx) = conj(F(fy, fx))` is looked up instead, which has
    the same magnitude.

    Parameters
    ----------
    angles_deg : torch.Tensor
        `(angle, )` candidate line angles in degrees.
    rhos : torch.Tensor
        `(rho, )` normalized frequency radii shared by both axes.
    power_sum : torch.Tensor
        `(h, w // 2 + 1)` power spectrum, non-fftshifted rfft2 layout.
    image_shape : tuple[int, int]
        `(h, w)` shape of the original (pre-rfft) images.

    Returns
    -------
    power : torch.Tensor
        `(angle, )` total power along the line at each candidate angle.
    """
    h, w = image_shape
    rad = torch.deg2rad(angles_deg)
    cos, sin = torch.cos(rad), torch.sin(rad)
    fy = sin[:, None] * rhos[None, :]  # (angle, rho)
    fx = cos[:, None] * rhos[None, :]  # (angle, rho)
    flip = (cos < 0)[:, None]
    fy, fx = torch.where(flip, -fy, fy), torch.where(flip, -fx, fx)

    rows = torch.round(fy * h).long() % h
    cols = torch.round(fx * w).long()
    valid = (cols >= 0) & (cols < power_sum.shape[-1])
    values = power_sum[rows, cols.clamp(0, power_sum.shape[-1] - 1)]
    return (values * valid).sum(dim=-1)
