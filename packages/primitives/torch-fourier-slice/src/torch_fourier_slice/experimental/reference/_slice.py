"""3D central slices in plain torch: 3D volume <-> 2D central slices.

The torch counterparts of ``ExtractSlices3D.apply`` / ``InsertSlices3D.apply`` in
``.._backend._slice_3d``: same arguments, same layouts, same results. Every step
mirrors the Mojo kernels in ``_mojo/_slice_3d.mojo`` (the sampling itself is in
``._rfft``), written as differentiable tensor ops, so autograd supplies the
gradients the Mojo backend computes with dedicated kernels.
"""

from __future__ import annotations

import math

import torch

from .._backend._common import (
    IDENTITY_MAGNIFICATION,
    Magnification,
    input_pixel_scale,
    radius_cutoff,
    sampling_scale,
)
from .._backend._validation import (
    interp_code,
    prep_rotations,
    prep_shifts_2d,
    prep_shifts_3d,
    rfft_volume_shape,
    validate_reconstruction,
)
from ._rfft import gather, scatter


def _fourier_coord(n: int, device: torch.device) -> torch.Tensor:
    """Rfft frequency of each row index (DC at 0): positive, then negative.

    As ``torch.fft.fftfreq`` (and ``_fourier_coord`` in ``_common.mojo``): for even
    ``n`` the Nyquist row ``n // 2`` is ``-n/2``.
    """
    i = torch.arange(n, device=device)
    return torch.where(i <= (n - 1) // 2, i, i - n).to(torch.float32)


def _slice_coordinates(
    proj_sidelength: int,
    radius: float,
    scale: float,
    ewald_curvature: float,
    device: torch.device,
    magnification: Magnification = IDENTITY_MAGNIFICATION,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A central slice's own frequencies and its unrotated sample coordinate.

    Returns ``(coord_y, coord_x, s, inside)``: the ``(h, w)`` rfft frequencies,
    the ``(h, w, 3)`` zyx sample coordinate before rotation (times ``scale``, with the
    Ewald z-offset) and the mask of pixels inside the radius cutoff.
    """
    coord_y = _fourier_coord(proj_sidelength, device)[:, None]
    coord_x = torch.arange(
        proj_sidelength // 2 + 1, device=device, dtype=torch.float32
    )[None, :]
    coord_y, coord_x = torch.broadcast_tensors(coord_y, coord_x)
    inside = coord_y * coord_y + coord_x * coord_x <= torch.tensor(
        radius * radius, dtype=torch.float32
    )
    # the magnification's Fourier-space matrix M, then the oversampling
    m = magnification
    sy = (m.mag_yy * coord_y + m.mag_yx * coord_x) * scale
    sx = (m.mag_xy * coord_y + m.mag_xx * coord_x) * scale
    sz = ewald_curvature * (sx * sx + sy * sy)
    return coord_y, coord_x, torch.stack([sz, sy, sx], dim=-1), inside


def _rotate(rotations: torch.Tensor, s: torch.Tensor, bv: int) -> torch.Tensor:
    """Sample coordinates ``k (bv, bp, h, w, 3)``: zyx matrices applied to ``s``."""
    rotations = rotations.expand(bv, -1, -1, -1)
    return torch.einsum("vpij,hwj->vphwi", rotations, s)


def _shift_phase(
    shifts_2d: torch.Tensor | None,
    shifts_3d: torch.Tensor | None,
    coord_y: torch.Tensor,
    coord_x: torch.Tensor,
    k: torch.Tensor,
    proj_sidelength: int,
    sidelength: int,
) -> torch.Tensor | None:
    """Combined phase of the 2D (image-plane) and 3D (volume-frame) shifts.

    The 2D ramp uses the slice's own frequencies, the 3D ramp the rotated sample
    coordinate ``k``, as in ``_slice_3d_shift_phase``. ``None`` without shifts.
    """
    phase = None
    if shifts_2d is not None:
        ramp = (
            coord_y * shifts_2d[..., 0, None, None]
            + coord_x * shifts_2d[..., 1, None, None]
        )
        phase = ramp * (-2.0 * math.pi / proj_sidelength)
    if shifts_3d is not None:
        ramp = torch.einsum("vphwi,vpi->vphw", k, shifts_3d.expand(k.shape[0], -1, -1))
        ramp = ramp * (-2.0 * math.pi / sidelength)
        phase = ramp if phase is None else phase + ramp
    return phase


def extract_slices_3d(
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    shifts_3d: torch.Tensor | None,
    output_shape: tuple[int, int] | None,
    fftfreq_max: float | None,
    interpolation: str,
    ewald_curvature: float = 0.0,
    magnification: Magnification = IDENTITY_MAGNIFICATION,
) -> torch.Tensor:
    """Extract complex ``(bv, bp, h, w)`` central slices from rfft volumes.

    Same arguments as ``ExtractSlices3D.apply``: zyx ``rotations``, yx
    ``shifts_2d`` and zyx ``shifts_3d``, in the kernels' ``(bv, bp, ...)`` layout.
    """
    device = reconstruction.device
    reconstruction, bv, sidelength, _ = validate_reconstruction(reconstruction)
    interp_code(interpolation)
    rotations, _, bp = prep_rotations(rotations, bv, device)
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, device)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, device)

    if output_shape is None:
        proj_sidelength = sidelength
    else:
        if len(output_shape) != 2 or output_shape[0] != output_shape[1]:
            raise ValueError(f"output_shape {output_shape} must be square")
        proj_sidelength = int(output_shape[0])
    radius = radius_cutoff(fftfreq_max, proj_sidelength)

    coord_y, coord_x, s, inside = _slice_coordinates(
        proj_sidelength,
        radius,
        sampling_scale(sidelength, proj_sidelength),
        float(ewald_curvature),
        device,
        magnification,
    )
    k = _rotate(rotations, s, bv)
    slices = gather(reconstruction.to(torch.complex64), k, interpolation)
    phase = _shift_phase(
        shifts_2d_t if has_shifts_2d else None,
        shifts_3d_t if has_shifts_3d else None,
        coord_y,
        coord_x,
        k,
        proj_sidelength,
        sidelength,
    )
    if phase is not None:
        slices = slices * torch.complex(torch.cos(phase), torch.sin(phase))
    # the DFT of the image at A r carries 1 / |det A|
    return slices * inside * magnification.mag_scale


def insert_slices_3d(
    projections: torch.Tensor,
    weights: torch.Tensor | None,
    rotations: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    shifts_3d: torch.Tensor | None,
    volume_shape: tuple[int, int, int],
    fftfreq_max: float | None,
    interpolation: str,
    ewald_curvature: float = 0.0,
    magnification: Magnification = IDENTITY_MAGNIFICATION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert complex central slices into Hermitian rfft volumes (+ weights).

    Same arguments as ``InsertSlices3D.apply``: ``projections`` is ``(bp, h, w)``
    or ``(bv, bp, h, w)``, ``volume_shape`` the real-space ``(d, d, d)``.
    Returns the ``(bv, d, d, d // 2 + 1)`` data volume and the matching weight
    volume (of unit weights if ``weights`` is ``None``). Each input pixel's value
    and weight are scaled so that every Hermitian pair of
    the input counts once (``input_pixel_scale``, ``_input_pixel_scale`` in
    ``_common.mojo``).
    """
    if projections.dim() == 3:
        projections = projections.unsqueeze(0)
        if weights is not None and weights.dim() == 3:
            weights = weights.unsqueeze(0)
    if projections.dim() != 4 or not projections.is_complex():
        raise ValueError("projections must be complex (bv, bp, h, w)")
    bv, bp, h, w = projections.shape
    if w != h // 2 + 1:
        raise ValueError("projections must be square rfft images (w == h // 2 + 1)")
    if weights is None:  # unit weights: the weight volume is the density
        weights = torch.ones(
            projections.shape, dtype=torch.float32, device=projections.device
        )
    if weights.shape != projections.shape:
        raise ValueError("weights must match projections shape")
    interp_code(interpolation)

    device = projections.device
    rotations, _, bp_rot = prep_rotations(rotations, bv, device)
    if bp_rot != bp:
        raise ValueError("rotations pose count must match slices")
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, device)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, device)
    _, sidelength, _ = rfft_volume_shape(volume_shape)
    # the cutoff is in slice pixels, as the extraction's
    radius = radius_cutoff(fftfreq_max, h)

    coord_y, coord_x, s, inside = _slice_coordinates(
        h,
        radius,
        sampling_scale(sidelength, h),
        float(ewald_curvature),
        device,
        magnification,
    )
    # each input pixel's Hermitian weight and the magnification's 1 / |det A|
    keep = inside * input_pixel_scale(h, device) * magnification.mag_scale
    k = _rotate(rotations, s, bv)

    values = projections.to(torch.complex64) * keep
    phase = _shift_phase(
        shifts_2d_t if has_shifts_2d else None,
        shifts_3d_t if has_shifts_3d else None,
        coord_y,
        coord_x,
        k,
        h,
        sidelength,
    )
    if phase is not None:
        # the conjugate phase: the adjoint of the forward shift
        values = values * torch.complex(torch.cos(phase), -torch.sin(phase))
    weights = weights.to(torch.float32) * keep
    return scatter(values, weights, k, sidelength, interpolation)
