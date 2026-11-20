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
    coordinates = frequencies * image_shape_tensor + dc_index

    # `fftfreq(n) * n` isn't exact in float32, which puts the edge samples of some
    # sizes just outside the array, e.g. -2.4e-7 for the lowest frequency at n = 7.
    # Samplers zero, and insertion drops, anything outside the array, so snap
    # coordinates within `tolerance` of an edge onto it
    tolerance = 1e-3
    upper = image_shape_tensor - 1
    if rfft is True:
        upper[-1] = torch.div(image_shape_tensor[-1], 2, rounding_mode="floor")
    lower = torch.zeros_like(upper)
    coordinates = torch.where(
        (coordinates < lower) & (coordinates > lower - tolerance), lower, coordinates
    )
    coordinates = torch.where(
        (coordinates > upper) & (coordinates < upper + tolerance), upper, coordinates
    )
    return coordinates
