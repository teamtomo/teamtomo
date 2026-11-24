"""FFT-related shape utilities (rFFT output shape, FFT-friendly padding sizes)."""

import heapq
from typing import Sequence


def rfft_shape(input_shape: Sequence[int]) -> tuple[int, ...]:
    """Get the output shape of an ``rfft`` on an input with ``input_shape``."""
    out = list(input_shape)
    out[-1] = int((out[-1] / 2) + 1)
    return tuple(out)


def image_shape_from_dft_shape(
    dft_shape: Sequence[int],
    rfft: bool,
    *,
    image_shape: Sequence[int] | None = None,
    assume_square: bool = False,
) -> tuple[int, ...]:
    """Determine the real-space shape of an image from its DFT shape.

    Notes
    -----
    An rfft halves the last dimension to ``w // 2 + 1``, which is the same for real
    widths ``2k`` and ``2k + 1``, so the real-space width cannot be recovered from an
    rfft alone. Resolve it by passing ``image_shape`` or, where the image is known to
    be square/cubic, ``assume_square=True``.

    Parameters
    ----------
    dft_shape: Sequence[int]
        Shape of the DFT over the transformed dimensions.
    rfft: bool
        Whether the last dimension is the non-redundant half from an rfft.
    image_shape: Sequence[int] | None
        Real-space shape of the image. Checked against ``dft_shape`` and returned.
    assume_square: bool
        Take the real-space width from the second to last dimension of an rfft, for
        square/cubic images. Ignored if ``image_shape`` is given or ``rfft=False``.

    Returns
    -------
    image_shape: tuple[int, ...]
        Real-space shape of the image.

    Raises
    ------
    ValueError
        If ``image_shape`` does not give ``dft_shape``, if ``assume_square=True`` and
        the last two dimensions of ``dft_shape`` are not ``(n, n // 2 + 1)``, or if
        the width of an rfft is ambiguous.
    """
    dft_shape = tuple(int(n) for n in dft_shape)

    # full DFT of a real or complex image: no ambiguity
    if not rfft:
        if image_shape is not None:
            image_shape = tuple(int(n) for n in image_shape)
            if dft_shape != image_shape:
                raise ValueError(
                    f"Provided 'image_shape' ({image_shape}) does not match "
                    f"DFT shape ({dft_shape})."
                )
            return image_shape
        return dft_shape

    if image_shape is not None:
        image_shape = tuple(int(n) for n in image_shape)
        expected_shape = rfft_shape(image_shape)
        if dft_shape != expected_shape:
            raise ValueError(
                f"Expected DFT shape ({expected_shape}) implied by "
                f"'image_shape' ({image_shape}) does not match provided "
                f"DFT shape ({dft_shape})."
            )
        return image_shape

    # square/cubic images, e.g. Fourier slices, projections and their volumes
    if assume_square:
        if len(dft_shape) < 2 or dft_shape[-2] // 2 + 1 != dft_shape[-1]:
            raise ValueError(
                f"Expected last two dimensions of DFT shape ({dft_shape[-2:]}) "
                "to be (n, n // 2 + 1) for square/cubic images, but they are not."
            )
        n = dft_shape[-2]
        return (*dft_shape[:-1], n)

    raise ValueError(
        "rfft width is ambiguous without 'image_shape': real widths 2k and 2k + 1 "
        "both give k + 1 columns. Pass an explicit 'image_shape' to resolve it."
    )


def next_fft_size(n: int, factors: tuple[int, ...] = (2, 3, 5, 7)) -> int:
    """Smallest integer ``>= n`` whose prime factors lie in ``factors``.

    Such sizes are convenient for FFT implementations that pad to "FFT-friendly"
    lengths (mixed-radix transforms). For ``n <= 1``, returns ``1``.

    Parameters
    ----------
    n: int
        Minimum required size (typically a padded image edge length).
    factors: tuple[int, ...], optional
        List of allowed prime factors. Default is ``[2, 3, 5, 7]``.

    Returns
    -------
    int
        The smallest integer >= n whose prime factors lie in ``factors``.
    """
    if n <= 1:
        return 1
    if n <= 2:
        return 2

    heap = [1]
    seen = {1}
    max_int = 2**31
    upper_bound = min(n * 2, max_int)

    while heap:
        candidate = heapq.heappop(heap)

        # Return when the first larger candidate is found
        if candidate >= n:
            return candidate

        for prime in factors:
            next_val = candidate * prime
            if next_val > upper_bound:
                continue

            # Add new potential candidate into the heap
            if next_val not in seen:
                seen.add(next_val)
                heapq.heappush(heap, next_val)

    # Should never reach here.
    raise RuntimeError("Unreachable: no FFT size found")
