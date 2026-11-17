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
