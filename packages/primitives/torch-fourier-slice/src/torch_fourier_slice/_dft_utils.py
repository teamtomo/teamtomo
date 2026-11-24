import itertools
import math

import torch

# Sample points further than this past the Nyquist box read as 0 / are dropped. A
# small tolerance (in voxels) so rounding in the rotation doesn't drop points that
# lie exactly on the Nyquist plane, e.g. under a float32 90 degree rotation.
NYQUIST_TOLERANCE = 1e-3


def _in_nyquist_box(
    frequencies: torch.Tensor, image_shape: tuple[int, ...]
) -> torch.Tensor:
    """Whether each DFT sample frequency lies inside the Nyquist box `|f_i| <= 1/2`.

    Parameters
    ----------
    frequencies: torch.Tensor
        `(..., d)` array of DFT sample frequencies (cycles / pixel).
    image_shape: tuple[int, ...]
        Length `d` real space shape, used to express the tolerance in voxels.

    Returns
    -------
    in_box: torch.Tensor
        `(...)` boolean array.
    """
    shape = torch.as_tensor(
        image_shape, device=frequencies.device, dtype=frequencies.dtype
    )
    return torch.all(frequencies.abs() * shape <= shape / 2 + NYQUIST_TOLERANCE, dim=-1)


def _input_pixel_scale(
    freq_grid: torch.Tensor, image_shape: tuple[int, int]
) -> torch.Tensor:
    """How much each rfft pixel of a Hermitian image counts when backprojected.

    0 for the redundant half (ky < 0) of the kx = 0 column, 0.5 on the kx = 1/2
    column (even `w`) and for self-mirror pixels (ky = 0, or ky = -1/2 for even `h`)
    on both columns, 1 otherwise. See "Hermitian bookkeeping" in
    `torch_fourier_slice/experimental/README.md`, which covers both backends.

    Parameters
    ----------
    freq_grid: torch.Tensor
        `(h, w_rfft, d)` DFT sample frequencies of the image, yx in the last two
        entries of the last dimension.
    image_shape: tuple[int, int]
        Real space `(h, w)` of the image.

    Returns
    -------
    scale: torch.Tensor
        `(h, w_rfft)` weights in {0, 0.5, 1}.
    """
    h, w = image_shape
    y, x = freq_grid[..., -2], freq_grid[..., -1]
    self_mirror = y == 0
    if h % 2 == 0:
        self_mirror |= y == -0.5
    hermitian_column = x == 0
    if w % 2 == 0:
        hermitian_column |= x == 0.5
    scale = torch.ones_like(x)
    scale[hermitian_column] = 0.5
    scale[(x == 0) & (y < 0) & ~self_mirror] = 0
    scale[(x == 0) & (y > 0)] = 1
    return scale


def _sample_rfft_periodic(
    dft: torch.Tensor, coordinates: torch.Tensor, image_shape: tuple[int, ...]
) -> torch.Tensor:
    """Linearly interpolate an fftshifted rfft, reading past its edge periodically.

    The DFT of a sampled image repeats every `n` samples, so a stencil neighbour past
    the stored grid has the value of the stored sample one period away. Negative
    frequencies along the half transformed (last) dimension are read as the conjugate
    of the stored value at `-k` (Hermitian symmetry). For even `n` this is what fills
    in the Nyquist frequency `+n/2`, which an fftshifted DFT only stores at `-n/2`.

    Parameters
    ----------
    dft: torch.Tensor
        `(c, *full_dims, w_rfft)` fftshifted rfft (DC at `n // 2` on full dims).
    coordinates: torch.Tensor
        `(b, d)` array coordinates, as from
        `fftfreq_to_dft_coordinates(rfft=True, fftshift=True)`.
    image_shape: tuple[int, ...]
        Length `d` real space shape of the array the rfft was taken of.

    Returns
    -------
    samples: torch.Tensor
        `(b, c)` interpolated values.
    """
    ndim = coordinates.shape[-1]
    lower = torch.floor(coordinates)
    frac = coordinates - lower
    lower = lower.long()

    # rfft index is the frequency itself, wrap it into one period
    w = image_shape[-1]
    x_min = w // 2 - w + 1

    samples = torch.zeros(
        (coordinates.shape[0], dft.shape[0]), dtype=dft.dtype, device=dft.device
    )
    for corner in itertools.product((0, 1), repeat=ndim):
        idx = [lower[:, i] + corner[i] for i in range(ndim)]
        weight = math.prod(
            frac[:, i] if corner[i] else 1 - frac[:, i] for i in range(ndim)
        )
        x = torch.remainder(idx[-1] - x_min, w) + x_min
        conj = x < 0
        idx[-1] = x.abs()
        # full dims are fftshifted, -k sits at 2 * (n // 2) - i
        for i, n in enumerate(image_shape[:-1]):
            idx[i] = torch.remainder(
                torch.where(conj, 2 * (n // 2) - idx[i], idx[i]), n
            )
        values = dft[(slice(None), *idx)].T  # (b, c)
        values = torch.where(conj[:, None], values.conj(), values)
        samples = samples + weight[:, None] * values
    return samples


def _resample_rfft_edge(
    samples: torch.Tensor,
    dft: torch.Tensor,
    frequencies: torch.Tensor,
    coordinates: torch.Tensor,
    image_shape: tuple[int, ...],
) -> torch.Tensor:
    """Fill in samples whose interpolation stencil leaves the stored rfft.

    `sample_image_2d/3d` return 0 for any point past the edge of the stored array.
    Inside the Nyquist box the DFT values there are known, so resample those points
    periodically with `_sample_rfft_periodic`. Points outside the Nyquist box stay 0.

    Parameters
    ----------
    samples: torch.Tensor
        `(...)` samples, or `(..., c)` for a multichannel `dft`.
    dft: torch.Tensor
        `(*full_dims, w_rfft)` fftshifted rfft, or `(c, *full_dims, w_rfft)`.
    frequencies: torch.Tensor
        `(..., d)` DFT sample frequencies of the samples.
    coordinates: torch.Tensor
        `(..., d)` array coordinates of the samples.
    image_shape: tuple[int, ...]
        Length `d` real space shape of the array the rfft was taken of.

    Returns
    -------
    samples: torch.Tensor
        Samples with the edge points filled in.
    """
    ndim = coordinates.shape[-1]
    stored_shape = torch.as_tensor(
        dft.shape[-ndim:], device=coordinates.device, dtype=coordinates.dtype
    )
    outside = torch.any((coordinates < 0) | (coordinates > stored_shape - 1), dim=-1)
    edge = outside.nonzero(as_tuple=True)
    in_box = _in_nyquist_box(frequencies[edge], image_shape)
    edge = tuple(i[in_box] for i in edge)
    if edge[0].numel() == 0:
        return samples
    multichannel = dft.ndim == ndim + 1
    values = _sample_rfft_periodic(
        dft if multichannel else dft[None], coordinates[edge], image_shape
    )
    return samples.index_put(edge, values if multichannel else values[:, 0])
