"""Interpolating into and out of rfft arrays (DC at origin), in plain torch.

The sampling the Mojo kernels share between slices and lines
(``_gather.mojo``, ``_scatter.mojo``, ``_rfft_voxel`` / ``_rfft_pixel`` in
``_common.mojo``), for a 2D image ``(n, n // 2 + 1)`` or a 3D volume
``(n, n, n // 2 + 1)``, written as differentiable tensor ops. A sample coordinate
``k (..., ndim)`` is in voxels, ordered like the array's axes (zyx or yx).
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import torch

from ..._dft_utils import NYQUIST_TOLERANCE
from .._backend._common import hermitian_planes

if TYPE_CHECKING:
    from collections.abc import Iterator

Index = tuple[torch.Tensor, ...]


def _cubic_kernel(s: torch.Tensor) -> torch.Tensor:
    """Catmull-Rom cubic convolution weight (a = -0.5), support ``|s| <= 2``."""
    s = s.abs()
    a = -0.5
    near = (a + 2.0) * s**3 - (a + 3.0) * s**2 + 1.0
    far = a * s**3 - 5.0 * a * s**2 + 8.0 * a * s - 4.0 * a
    return torch.where(s <= 1.0, near, torch.where(s <= 2.0, far, 0.0))


def _stencil(
    k: torch.Tensor, interpolation: str
) -> Iterator[tuple[Index, torch.Tensor]]:
    """Integer frequencies around each sample ``k (..., ndim)`` and their weights.

    Linear: the ``2**ndim`` surrounding points. Cubic: the ``4**ndim`` block from
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
    taps = list(zip(offsets, weights, strict=True))
    for corner in itertools.product(taps, repeat=k.shape[-1]):
        index = tuple(base[..., axis] + o for axis, (o, _) in enumerate(corner))
        weight = math.prod(w[..., axis] for axis, (_, w) in enumerate(corner))
        yield index, weight


def _wrap_frequency(i: torch.Tensor, sidelength: int) -> torch.Tensor:
    """Periodic image of DFT frequency ``i`` in the stored range ``[hi - n + 1, hi]``.

    The DFT of a sampled array repeats every ``n``, so a frequency past the stored
    grid has the value of the stored one ``n`` away (``_wrap_frequency``).
    """
    lo = sidelength // 2 - sidelength + 1
    return torch.remainder(i - lo, sidelength) + lo


def _rfft_index(index: Index, sidelength: int) -> tuple[Index, torch.Tensor]:
    """Where the DFT value at integer frequency ``index`` is stored.

    As ``_rfft_voxel`` in ``_common.mojo``: a negative ``x`` (the last axis) folds
    onto the stored half by Hermitian symmetry, then each frequency wraps
    periodically into the stored range, folding again if ``x`` wrapped negative.
    Every frequency maps to a stored point. Returns the array indices and whether
    the stored value is the conjugate of the requested one.
    """
    conj = index[-1] < 0
    index = tuple(torch.where(conj, -i, i) for i in index)
    *rest, x = index
    x = _wrap_frequency(x, sidelength)
    refold = x < 0
    rest = [torch.where(refold, -i, i) for i in rest]
    x = torch.where(refold, -x, x)
    conj = conj ^ refold
    rest = [_wrap_frequency(i, sidelength) for i in rest]
    rest = [torch.where(i < 0, i + sidelength, i) for i in rest]
    return (*rest, x), conj


def _flat_index(index: Index, sidelength: int) -> torch.Tensor:
    """Index into a flattened ``(n, ..., n, n // 2 + 1)`` rfft array."""
    flat = index[0]
    for i in index[1:-1]:
        flat = flat * sidelength + i
    return flat * (sidelength // 2 + 1) + index[-1]


def _in_nyquist_box(k: torch.Tensor, sidelength: int) -> torch.Tensor:
    """Whether each sample ``k (..., ndim)`` lies inside ``|k_i| <= n / 2``.

    Points outside it read as 0 (gather) or are dropped (scatter), as
    ``_in_nyquist_cube`` / ``_in_nyquist_square`` in ``_common.mojo``.
    """
    return (k.abs() <= sidelength / 2 + NYQUIST_TOLERANCE).all(dim=-1)


def gather(array: torch.Tensor, k: torch.Tensor, interpolation: str) -> torch.Tensor:
    """Interpolate the rfft arrays ``(b, n, ..., n // 2 + 1)`` at ``k (b, ..., ndim)``.

    The stencil reads the DFT periodically (``_rfft_index``); samples outside the
    Nyquist box read as 0.
    """
    batch_size, sidelength = array.shape[0], array.shape[-2]
    flat = array.reshape(batch_size, -1)
    batch = torch.arange(batch_size, device=array.device)
    batch = batch.view(-1, *([1] * (k.dim() - 2)))
    out = torch.zeros(k.shape[:-1], dtype=array.dtype, device=array.device)
    for index, weight in _stencil(k, interpolation):
        index, conj = _rfft_index(index, sidelength)
        value = flat[batch, _flat_index(index, sidelength)]
        value = torch.where(conj, value.conj(), value)
        out = out + value * weight
    return out * _in_nyquist_box(k, sidelength)


def scatter(
    values: torch.Tensor,
    weights: torch.Tensor | None,
    k: torch.Tensor,
    sidelength: int,
    interpolation: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Splat ``values (b, ...)`` (and real ``weights``) into rfft arrays at ``k``.

    Returns the ``(b, n, ..., n // 2 + 1)`` data and weight arrays (``None``
    without ``weights``). The adjoint of :func:`gather` (samples outside the
    Nyquist box are dropped), plus the reconstruction's Hermitian double-insert: a
    write to the kx = 0 or kx = n/2 plane also adds its conjugate at the mirrored
    ``(-z, -y)`` point, which for a self-mirror point is the point itself
    (``_accumulate_3d`` / ``_accumulate_2d`` with ``friedel_double``).
    """
    batch_size, ndim = values.shape[0], k.shape[-1]
    in_box = _in_nyquist_box(k, sidelength)
    values = values * in_box
    if weights is not None:
        weights = weights * in_box
    shape = (batch_size, *(sidelength,) * (ndim - 1), sidelength // 2 + 1)
    size = math.prod(shape[1:])
    offset = torch.arange(batch_size, device=values.device) * size
    offset = offset.view(-1, *([1] * (k.dim() - 2)))
    data = torch.zeros(batch_size * size, dtype=values.dtype, device=values.device)
    weight_array = None
    if weights is not None:
        weight_array = torch.zeros(
            batch_size * size, dtype=weights.dtype, device=weights.device
        )

    def add(index, value, weight):
        flat = (offset + _flat_index(index, sidelength)).reshape(-1)
        data.index_add_(0, flat, value.reshape(-1))
        if weight_array is not None:
            weight_array.index_add_(0, flat, weight.reshape(-1))

    planes = torch.tensor(hermitian_planes(sidelength), device=values.device)
    for index, tap_weight in _stencil(k, interpolation):
        index, conj = _rfft_index(index, sidelength)
        value = torch.where(conj, values.conj(), values) * tap_weight
        weight = weights * tap_weight if weights is not None else None
        add(index, value, weight)

        *rest, x = index
        mirror = (*((sidelength - i) % sidelength for i in rest), x)
        mirrored = torch.isin(x, planes)
        add(
            mirror,
            value.conj() * mirrored,
            weight * mirrored if weight is not None else None,
        )

    data = data.reshape(shape)
    if weight_array is not None:
        weight_array = weight_array.reshape(shape)
    return data, weight_array
