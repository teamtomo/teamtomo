"""FFT-related shape utilities (rFFT output shape, FFT-friendly padding sizes)."""

import heapq
from typing import Sequence


def rfft_shape(input_shape: Sequence[int]) -> tuple[int, ...]:
    """Get the output shape of an ``rfft`` on an input with ``input_shape``."""
    out = list(input_shape)
    out[-1] = int((out[-1] / 2) + 1)
    return tuple(out)


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
