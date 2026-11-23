"""Image sidelength/shape helpers for 1D profiles."""

from __future__ import annotations

import operator


def as_image_shape(image_sidelength: int | tuple[int, int]) -> tuple[int, int]:
    """``(h, w)`` from a sidelength (square image) or an ``(h, w)`` shape.

    Any integer scalar is accepted as a sidelength, including ``np.int64`` and 0-d
    integer tensors.
    """
    try:
        n = operator.index(image_sidelength)  # type: ignore[arg-type]
    except TypeError:
        h, w = image_sidelength[-2], image_sidelength[-1]  # type: ignore[index]
        return (int(h), int(w))
    return (n, n)


def profile_sidelength(image_sidelength: int | tuple[int, int]) -> int:
    """Sidelength ``n = min(h, w)`` that sets the 1D profile length ``n // 2 + 1``.

    This matches the shell count of ``rotational_average_dft_2d`` for non-square
    images.
    """
    return min(as_image_shape(image_sidelength))
