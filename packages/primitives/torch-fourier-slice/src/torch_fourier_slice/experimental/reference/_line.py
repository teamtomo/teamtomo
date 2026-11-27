"""Central lines in plain torch: 3D volume or 2D image <-> 1D central lines.

The torch counterparts of ``ExtractLines3D`` / ``InsertLines3D`` in
``.._backend._line_3d`` and ``ExtractLines2D`` / ``InsertLines2D`` in
``.._backend._line_2d``: same arguments, same layouts, same results. Every step
mirrors ``_mojo/_line_3d.mojo`` and ``_mojo/_line_2d.mojo`` (the sampling itself
is in ``._rfft``), written as differentiable tensor ops, so autograd supplies the
gradients the Mojo backend computes with dedicated kernels.

A line is posed by a direction ``u`` (zyx in 3D, yx in 2D): line pixel ``x``
samples ``k = x * (n / L) * u`` in an array of side ``n``, for lines of length
``L`` (``sampling_scale``). Only the shift of the array the line is cut
from applies (3D for a volume, 2D for an image), as the phase ramp ``k . t``.
"""

from __future__ import annotations

import math

import torch

from .._backend._common import input_line_scale, sampling_scale
from .._backend._validation import (
    even_line_sidelength,
    interp_code,
    prep_directions_2d,
    prep_directions_3d,
    prep_shifts_2d,
    prep_shifts_3d,
    validate_image_2d,
    validate_image_shape,
    validate_reconstruction,
    validate_volume_shape,
)
from ._rfft import gather, scatter


def _output_length(output_length: int | None, sidelength: int) -> int:
    """Real-space length of the extracted lines (the array's side by default)."""
    if output_length is None:
        return sidelength
    line_sidelength = int(output_length)
    if line_sidelength % 2 != 0:
        raise ValueError(f"output_length {line_sidelength} must be even")
    return line_sidelength


def _line_samples(
    directions: torch.Tensor,
    bv: int,
    line_sidelength: int,
    fftfreq_max: float | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample coordinates ``k (bv, bp, w, ndim)`` of each line pixel.

    Also returns the ``(w,)`` mask of pixels inside the radius cutoff, which is
    ``fftfreq_max`` of the line length (Nyquist by default).
    """
    radius = (
        line_sidelength / 2.0
        if fftfreq_max is None
        else float(fftfreq_max) * line_sidelength
    )
    coord_x = torch.arange(
        line_sidelength // 2 + 1, device=directions.device, dtype=torch.float32
    )
    inside = coord_x * coord_x <= torch.tensor(radius * radius, dtype=torch.float32)
    sx = coord_x * float(scale)
    k = sx[:, None] * directions.expand(bv, -1, -1)[:, :, None, :]
    return k, inside


def _phase_factor(
    shifts: torch.Tensor, k: torch.Tensor, sidelength: int, conjugate: bool = False
) -> torch.Tensor:
    """``exp(-2 pi i k . t / n)`` for each sample (conjugated for an insertion)."""
    ramp = torch.einsum("vpwi,vpi->vpw", k, shifts.expand(k.shape[0], -1, -1))
    phase = ramp * (-2.0 * math.pi / sidelength)
    sin = -torch.sin(phase) if conjugate else torch.sin(phase)
    return torch.complex(torch.cos(phase), sin)


def _validate_lines(
    lines: torch.Tensor, weights: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Lines and weights as ``(bv, bp, w)``, as ``run_insert_lines_3d`` takes them."""
    if lines.dim() == 2:
        lines = lines.unsqueeze(0)
        if weights is not None and weights.dim() == 2:
            weights = weights.unsqueeze(0)
    if lines.dim() != 3 or not lines.is_complex():
        raise ValueError("lines must be complex (bv, bp, w)")
    if weights is not None and weights.shape != lines.shape:
        raise ValueError("weights must match lines shape")
    return lines, weights


def _insert(
    lines: torch.Tensor,
    weights: torch.Tensor | None,
    k: torch.Tensor,
    inside: torch.Tensor,
    shifts: torch.Tensor | None,
    line_sidelength: int,
    sidelength: int,
    interpolation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter lines into Hermitian rfft arrays of side ``sidelength``.

    Each sample's value and weight are scaled so that every Hermitian pair of the
    line counts once (``input_line_scale``), and shifts apply the conjugate phase.
    Without ``weights`` every sample weighs one, so the weights give the density.
    """
    keep = inside * input_line_scale(line_sidelength, lines.device)
    values = lines.to(torch.complex64) * keep
    if shifts is not None:
        values = values * _phase_factor(shifts, k, sidelength, conjugate=True)
    if weights is None:
        weights = torch.ones(lines.shape, device=lines.device)
    weights = weights.to(torch.float32) * keep
    return scatter(values, weights, k, sidelength, interpolation)


def extract_lines_3d(
    reconstruction: torch.Tensor,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None,
    output_length: int | None,
    fftfreq_max: float | None,
    interpolation: str,
) -> torch.Tensor:
    """Extract complex ``(bv, bp, w)`` central lines from rfft volumes.

    Same arguments as ``ExtractLines3D.apply``: zyx ``directions`` and
    ``shifts_3d`` in the kernels' ``(bv, bp, ...)`` layout.
    """
    device = reconstruction.device
    reconstruction, bv, sidelength, _ = validate_reconstruction(reconstruction)
    interp_code(interpolation)
    directions, _, bp = prep_directions_3d(directions, bv, device)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, device)
    line_sidelength = _output_length(output_length, sidelength)

    k, inside = _line_samples(
        directions,
        bv,
        line_sidelength,
        fftfreq_max,
        sampling_scale(sidelength, line_sidelength),
    )
    lines = gather(reconstruction.to(torch.complex64), k, interpolation)
    if has_shifts_3d:
        lines = lines * _phase_factor(shifts_3d_t, k, sidelength)
    return lines * inside


def insert_lines_3d(
    lines: torch.Tensor,
    weights: torch.Tensor | None,
    directions: torch.Tensor,
    shifts_3d: torch.Tensor | None,
    volume_shape: tuple[int, int, int],
    fftfreq_max: float | None,
    interpolation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert complex central lines into Hermitian rfft volumes (+ weights).

    Same arguments as ``InsertLines3D.apply``: ``lines`` is ``(bp, w)`` or
    ``(bv, bp, w)``, ``volume_shape`` the real-space ``(d, d, d)``. Returns the
    ``(bv, d, d, d // 2 + 1)`` data volume and the matching weight volume (of
    unit weights if ``weights`` is ``None``).
    """
    lines, weights = _validate_lines(lines, weights)
    bv, bp, w = lines.shape
    interp_code(interpolation)
    directions, _, bp_dir = prep_directions_3d(directions, bv, lines.device)
    if bp_dir != bp:
        raise ValueError("directions pose count must match lines")
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, lines.device)
    line_sidelength = even_line_sidelength(w)
    sidelength = validate_volume_shape(volume_shape)[0]

    k, inside = _line_samples(
        directions,
        bv,
        line_sidelength,
        fftfreq_max,
        sampling_scale(sidelength, line_sidelength),
    )
    return _insert(
        lines,
        weights,
        k,
        inside,
        shifts_3d_t if has_shifts_3d else None,
        line_sidelength,
        sidelength,
        interpolation,
    )


def extract_lines_2d(
    image_rfft: torch.Tensor,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    output_length: int | None,
    fftfreq_max: float | None,
    interpolation: str,
) -> torch.Tensor:
    """Extract complex ``(bv, bp, w)`` central lines from 2D rfft images.

    Same arguments as ``ExtractLines2D.apply``: yx ``directions`` and
    ``shifts_2d`` in the kernels' ``(bv, bp, ...)`` layout.
    """
    device = image_rfft.device
    image_rfft, bv, sidelength, _ = validate_image_2d(image_rfft)
    interp_code(interpolation)
    directions, _, bp = prep_directions_2d(directions, bv, device)
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, device)
    line_sidelength = _output_length(output_length, sidelength)

    k, inside = _line_samples(
        directions,
        bv,
        line_sidelength,
        fftfreq_max,
        sampling_scale(sidelength, line_sidelength),
    )
    lines = gather(image_rfft.to(torch.complex64), k, interpolation)
    if has_shifts_2d:
        lines = lines * _phase_factor(shifts_2d_t, k, sidelength)
    return lines * inside


def insert_lines_2d(
    lines: torch.Tensor,
    weights: torch.Tensor | None,
    directions: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    image_shape: tuple[int, int],
    fftfreq_max: float | None,
    interpolation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert complex central lines into Hermitian 2D rfft images (+ weights).

    Same arguments as ``InsertLines2D.apply``: ``lines`` is ``(bp, w)`` or
    ``(bv, bp, w)``, ``image_shape`` the real-space ``(h, h)``. Returns the
    ``(bv, h, h // 2 + 1)`` data image and the matching weight image (of unit
    weights if ``weights`` is ``None``).
    """
    lines, weights = _validate_lines(lines, weights)
    bv, bp, w = lines.shape
    interp_code(interpolation)
    directions, _, bp_dir = prep_directions_2d(directions, bv, lines.device)
    if bp_dir != bp:
        raise ValueError("directions pose count must match lines")
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, lines.device)
    line_sidelength = even_line_sidelength(w)
    sidelength = validate_image_shape(image_shape)[0]

    k, inside = _line_samples(
        directions,
        bv,
        line_sidelength,
        fftfreq_max,
        sampling_scale(sidelength, line_sidelength),
    )
    return _insert(
        lines,
        weights,
        k,
        inside,
        shifts_2d_t if has_shifts_2d else None,
        line_sidelength,
        sidelength,
        interpolation,
    )
