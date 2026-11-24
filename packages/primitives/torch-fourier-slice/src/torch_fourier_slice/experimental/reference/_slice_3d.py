"""3D central slices in plain torch: 3D volume <-> 2D central slices.

The torch counterparts of ``ExtractSlices3D.apply`` / ``InsertSlices3D.apply`` in
``.._backend._slice_3d``: same arguments, same layouts, same results. Every step
mirrors the Mojo kernels in ``_mojo/_slice_3d.mojo``, ``_gather.mojo`` and
``_scatter.mojo``, written as differentiable tensor ops, so autograd supplies the
gradients the Mojo backend computes with dedicated kernels.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from ..._dft_utils import NYQUIST_TOLERANCE
from .._backend._common import (
    hermitian_planes,
    input_pixel_scale,
    reconstruction_volume_shape,
)
from .._backend._validation import (
    interp_code,
    prep_rotations,
    prep_shifts_2d,
    prep_shifts_3d,
    validate_reconstruction,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def _cubic_kernel(s: torch.Tensor) -> torch.Tensor:
    """Catmull-Rom cubic convolution weight (a = -0.5), support ``|s| <= 2``."""
    s = s.abs()
    a = -0.5
    near = (a + 2.0) * s**3 - (a + 3.0) * s**2 + 1.0
    far = a * s**3 - 5.0 * a * s**2 + 8.0 * a * s - 4.0 * a
    return torch.where(s <= 1.0, near, torch.where(s <= 2.0, far, 0.0))


def _stencil(
    k: torch.Tensor, interpolation: str
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Integer voxels ``(z, y, x)`` around each sample ``k (..., 3)`` and their weights.

    Trilinear: the 8 surrounding voxels. Tricubic: the 4x4x4 block from
    ``floor(k) - 1`` to ``floor(k) + 2``. The weights are differentiable in ``k``.
    """
    k_floor = torch.floor(k)
    base = k_floor.long()
    frac = k - k_floor
    offsets: tuple[int, ...]
    if interpolation == "cubic":
        offsets = (-1, 0, 1, 2)
        weights = [_cubic_kernel(frac - o) for o in offsets]
    else:
        offsets = (0, 1)
        weights = [1.0 - frac, frac]
    for oz, wz in zip(offsets, weights, strict=True):
        for oy, wy in zip(offsets, weights, strict=True):
            for ox, wx in zip(offsets, weights, strict=True):
                yield (
                    base[..., 0] + oz,
                    base[..., 1] + oy,
                    base[..., 2] + ox,
                    wz[..., 0] * wy[..., 1] * wx[..., 2],
                )


def _wrap_frequency(i: torch.Tensor, sidelength: int) -> torch.Tensor:
    """Periodic image of DFT frequency ``i`` in the stored range ``[hi - n + 1, hi]``.

    The DFT of a sampled volume repeats every ``n``, so a frequency past the stored
    grid has the value of the stored one ``n`` away (``_wrap_frequency``).
    """
    lo = sidelength // 2 - sidelength + 1
    return torch.remainder(i - lo, sidelength) + lo


def _rfft_voxel(
    z: torch.Tensor, y: torch.Tensor, x: torch.Tensor, sidelength: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Where the DFT value at integer frequency ``(z, y, x)`` is stored.

    As ``_rfft_voxel`` in ``_common.mojo``: a negative ``x`` folds onto the stored
    half by Hermitian symmetry, then each frequency wraps periodically into the
    stored range, folding again if ``x`` wrapped negative. Every frequency maps to
    a stored voxel. Returns the array indices ``(z, y, x)`` and whether the stored
    value is the conjugate of the requested one.
    """
    conj = x < 0
    z, y, x = (torch.where(conj, -i, i) for i in (z, y, x))
    x = _wrap_frequency(x, sidelength)
    refold = x < 0
    z, y, x = (torch.where(refold, -i, i) for i in (z, y, x))
    conj = conj ^ refold
    z = _wrap_frequency(z, sidelength)
    y = _wrap_frequency(y, sidelength)
    z = torch.where(z < 0, z + sidelength, z)
    y = torch.where(y < 0, y + sidelength, y)
    return z, y, x, conj


def _in_nyquist_cube(k: torch.Tensor, sidelength: int) -> torch.Tensor:
    """Whether each sample ``k (..., 3)`` lies inside ``|k_i| <= n / 2``.

    Points outside it read as 0 (gather) or are dropped (scatter), as
    ``_in_nyquist_cube`` in ``_common.mojo``.
    """
    return (k.abs() <= sidelength / 2 + NYQUIST_TOLERANCE).all(dim=-1)


def _flat_index(
    z: torch.Tensor, y: torch.Tensor, x: torch.Tensor, sidelength: int
) -> torch.Tensor:
    """Index into a flattened ``(d, h, w)`` rfft volume."""
    return (z * sidelength + y) * (sidelength // 2 + 1) + x


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
    oversampling: float,
    ewald_curvature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A central slice's own frequencies and its unrotated sample coordinate.

    Returns ``(coord_y, coord_x, s, inside)``: the ``(h, w)`` rfft frequencies,
    the ``(h, w, 3)`` zyx sample coordinate before rotation (oversampled, with the
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
    sy = coord_y * oversampling
    sx = coord_x * oversampling
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


def _gather(volume: torch.Tensor, k: torch.Tensor, interpolation: str) -> torch.Tensor:
    """Interpolate the rfft volumes ``(bv, d, h, w)`` at ``k (bv, ...)``.

    The stencil reads the DFT periodically (``_rfft_voxel``); samples outside the
    Nyquist cube read as 0.
    """
    bv, sidelength = volume.shape[0], volume.shape[-2]
    flat = volume.reshape(bv, -1)
    batch = torch.arange(bv, device=volume.device).view(-1, *([1] * (k.dim() - 2)))
    out = torch.zeros(k.shape[:-1], dtype=volume.dtype, device=volume.device)
    for z, y, x, weight in _stencil(k, interpolation):
        z, y, x, conj = _rfft_voxel(z, y, x, sidelength)
        value = flat[batch, _flat_index(z, y, x, sidelength)]
        value = torch.where(conj, value.conj(), value)
        out = out + value * weight
    return out * _in_nyquist_cube(k, sidelength)


def _scatter(
    values: torch.Tensor,
    weights: torch.Tensor | None,
    k: torch.Tensor,
    sidelength: int,
    interpolation: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Splat ``values (bv, ...)`` (and real ``weights``) into rfft volumes at ``k``.

    The adjoint of ``_gather`` (samples outside the Nyquist cube are dropped),
    plus the reconstruction's Hermitian double-insert: a write to the kx = 0 or
    kx = n/2 plane also adds its conjugate at the mirrored ``(-z, -y)`` voxel,
    which for a self-mirror voxel is the voxel itself (``_accumulate_3d`` with
    ``friedel_double``).
    """
    bv = values.shape[0]
    in_cube = _in_nyquist_cube(k, sidelength)
    values = values * in_cube
    if weights is not None:
        weights = weights * in_cube
    n_voxels = sidelength * sidelength * (sidelength // 2 + 1)
    shape = (bv, sidelength, sidelength, sidelength // 2 + 1)
    offset = (torch.arange(bv, device=values.device) * n_voxels).view(
        -1, *([1] * (k.dim() - 2))
    )
    data = torch.zeros(bv * n_voxels, dtype=values.dtype, device=values.device)
    weight_volume = None
    if weights is not None:
        weight_volume = torch.zeros(
            bv * n_voxels, dtype=weights.dtype, device=weights.device
        )

    def add(index, value, weight):
        data.index_add_(0, (offset + index).reshape(-1), value.reshape(-1))
        if weight_volume is not None:
            weight_volume.index_add_(
                0, (offset + index).reshape(-1), weight.reshape(-1)
            )

    planes = torch.tensor(hermitian_planes(sidelength), device=values.device)
    for z, y, x, tap_weight in _stencil(k, interpolation):
        z, y, x, conj = _rfft_voxel(z, y, x, sidelength)
        value = torch.where(conj, values.conj(), values) * tap_weight
        weight = weights * tap_weight if weights is not None else None
        add(_flat_index(z, y, x, sidelength), value, weight)

        mirror_z = (sidelength - z) % sidelength
        mirror_y = (sidelength - y) % sidelength
        mirrored = torch.isin(x, planes)
        add(
            _flat_index(mirror_z, mirror_y, x, sidelength),
            value.conj() * mirrored,
            weight * mirrored if weight is not None else None,
        )

    data = data.reshape(shape)
    if weight_volume is not None:
        weight_volume = weight_volume.reshape(shape)
    return data, weight_volume


def extract_slices_3d(
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    shifts_3d: torch.Tensor | None,
    output_shape: tuple[int, int] | None,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
    ewald_curvature: float = 0.0,
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
        if output_shape[0] % 2 != 0:
            raise ValueError(f"output side length {output_shape[0]} must be even")
        proj_sidelength = int(output_shape[0])
    radius = (
        proj_sidelength / 2.0
        if fftfreq_max is None
        else float(fftfreq_max) * proj_sidelength
    )

    coord_y, coord_x, s, inside = _slice_coordinates(
        proj_sidelength, radius, float(oversampling), float(ewald_curvature), device
    )
    k = _rotate(rotations, s, bv)
    slices = _gather(reconstruction.to(torch.complex64), k, interpolation)
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
    return slices * inside


def insert_slices_3d(
    projections: torch.Tensor,
    weights: torch.Tensor | None,
    rotations: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    shifts_3d: torch.Tensor | None,
    oversampling: float,
    fftfreq_max: float | None,
    interpolation: str,
    ewald_curvature: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Insert complex central slices into Hermitian rfft volumes (+ weights).

    Same arguments as ``InsertSlices3D.apply``: ``projections`` is ``(bp, h, w)``
    or ``(bv, bp, h, w)``. Returns the ``(bv, d, h, w)`` data volume and, if
    ``weights`` were given, the matching weight volume (else ``None``). Each
    input pixel's value and weight are scaled so that every Hermitian pair of
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
    if h % 2 != 0 or h != (w - 1) * 2:
        raise ValueError("projections must be square with even box (h == 2*(w-1))")
    if weights is not None and weights.shape != projections.shape:
        raise ValueError("weights must match projections shape")
    interp_code(interpolation)

    device = projections.device
    rotations, _, bp_rot = prep_rotations(rotations, bv, device)
    if bp_rot != bp:
        raise ValueError("rotations pose count must match slices")
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, device)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, device)
    _, sidelength, _ = reconstruction_volume_shape(h, oversampling)
    radius = sidelength / 2.0 if fftfreq_max is None else float(fftfreq_max) * h

    coord_y, coord_x, s, inside = _slice_coordinates(
        h, radius, float(oversampling), float(ewald_curvature), device
    )
    keep = inside * input_pixel_scale(h, device)
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
    if weights is not None:
        weights = weights.to(torch.float32) * keep
    return _scatter(values, weights, k, sidelength, interpolation)
