"""Gridding (de-apodization) correction for the interpolation kernel.

Sampling the rfft volume at a non-integer coordinate is a convolution with the
interpolation kernel ``k``. A convolution in Fourier space is a multiplication in
real space by ``K``, the continuous Fourier transform of ``k`` -- so an extracted
projection is the projection of ``volume * K``, and an inserted reconstruction
comes out as ``reconstruction * K``. Both are corrected by **dividing** by ``K``:
the volume on the way in to an extraction, the reconstruction on the way out of
an insertion.

``K`` depends on which kernel was used, so it must follow the ``interpolation``
argument -- ``sinc**2`` is only correct for trilinear:

- **linear** -- the tent kernel, ``K(v) = sinc(v)**2``.
- **cubic** -- the Catmull-Rom kernel (Keys, ``a = -1/2``), whose transform is
  ``K(v) = sinc(v)**3 * (3*sinc(v) - 2*cos(pi*v))``. Derived by integrating the
  kernel directly, and verified against numerical quadrature to machine
  precision. It apodizes less than the tent (``K(0.25)`` is 0.94 vs 0.81), as
  expected of the more accurate interpolant.

Both kernels are separable, so the 3D correction is the product of the per-axis
1D transforms rather than a function of the frequency magnitude.
"""

from __future__ import annotations

import torch

_INTERPOLATIONS = ("linear", "cubic")


def _kernel_transform(frequency: torch.Tensor, interpolation: str) -> torch.Tensor:
    """Continuous Fourier transform of the interpolation kernel, ``K(v)``.

    ``frequency`` is a normalised real-space coordinate in ``[-0.5, 0.5)``. Both
    forms are written via :func:`torch.sinc` so they need no division and are
    exactly 1 at the origin.
    """
    sinc = torch.sinc(frequency)
    if interpolation == "linear":
        return sinc**2
    if interpolation == "cubic":
        return sinc**3 * (3 * sinc - 2 * torch.cos(torch.pi * frequency))
    raise ValueError(
        f"interpolation must be one of {sorted(_INTERPOLATIONS)}, got {interpolation!r}"
    )


def gridding_correction(
    sidelength: int,
    interpolation: str,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Cubic ``(d, d, d)`` de-apodization volume for ``interpolation``.

    Divide a centered real-space volume by this to undo the apodization the
    interpolation kernel imposes. Built as an outer product of the per-axis 1D
    transform, so it never materialises an intermediate coordinate grid.
    """
    frequency = torch.fft.fftshift(torch.fft.fftfreq(sidelength, device=device))
    k = _kernel_transform(frequency, interpolation)
    return k[:, None, None] * k[None, :, None] * k[None, None, :]
