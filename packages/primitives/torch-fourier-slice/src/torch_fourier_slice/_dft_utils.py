import itertools
import math

import torch


def _fftfreq_to_dft_coordinates(
    frequencies: torch.Tensor, image_shape: tuple[int, ...], rfft: bool
) -> torch.Tensor:
    """Convert DFT sample frequencies into array coordinates in a fftshifted DFT.

    Specifically:
    - `delta_fftfreq` is the step size in frequency space between adjacent DFT samples,
      `1 / n` for a dimension of length `n`.
    - after fftshift, DC sits at array index `n // 2` of each full dimension. This is
      `n / 2` for even `n` but `(n - 1) / 2` for odd `n`.
    - in the non-redundant half transform from `rfft`, DC sits at array index `0` of the
      last dimension.
    - we specify continuous coordinates in the DFT array coordinate system `[0, n - 1]`
      for subsequent sampling

    This functions converts from normalized DFT sample frequencies to DFT array
    coordinates, accounting for arbitrary dimensionaly and whether the DFT is the
    non-redundant half transform from `rfft`.


    Parameters
    ----------
    frequencies: torch.Tensor
        `(..., d)` array of multidimensional DFT sample frequencies
    image_shape: tuple[int, ...]
        Length `d` array of image dimensions. For an rfft, the last dimension is the
        length of the full (real space) dimension, not the half transform.
    rfft: bool
        Whether output should be compatible with an rfft (`True`) or a
        full DFT (`False`)

    Returns
    -------
    coordinates: torch.Tensor
        `(..., d)` array of coordinates into a fftshifted DFT.
    """
    # grab relevant dimension lengths (number of samples per dimension)
    image_shape_tensor: torch.Tensor = torch.as_tensor(
        image_shape, device=frequencies.device, dtype=frequencies.dtype
    )

    # array index of DC in each dimension
    # last dim is only non-redundant half in rfft case, which starts at DC
    dc_index = torch.div(image_shape_tensor, 2, rounding_mode="floor")
    if rfft is True:
        dc_index[-1] = 0

    # adjacent DFT samples are `1 / n` apart, so frequency `f` is `f * n` samples
    # away from DC
    return frequencies * image_shape_tensor + dc_index


def _rfft_full_width(h: int, w_rfft: int) -> int:
    """Recover the real space width of a square/cubic array from its rfft.

    The rfft halves the last dimension to `w // 2 + 1`, which is the same for an
    even `w` and for `w + 1`, so `w` can't be recovered from the half transform
    alone. Take it from the full dimension `h` before it when the two agree, which
    they do for square/cubic arrays of either parity, and fall back to the even
    width otherwise.

    Parameters
    ----------
    h: int
        Length of the full dimension before the half transformed one.
    w_rfft: int
        Length of the half transformed last dimension, `w // 2 + 1`.

    Returns
    -------
    w: int
        Length of the last dimension of the real space array.
    """
    return h if h // 2 + 1 == w_rfft else 2 * (w_rfft - 1)


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
        `(b, d)` array coordinates, as from `_fftfreq_to_dft_coordinates(rfft=True)`.
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
