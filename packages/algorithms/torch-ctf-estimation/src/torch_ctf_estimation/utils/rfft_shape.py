"""Image sidelength/shape helpers for 1D profiles."""

from __future__ import annotations


def as_image_shape(image_sidelength: int | tuple[int, int]) -> tuple[int, int]:
    """``(h, w)`` from a sidelength (square image) or an ``(h, w)`` shape."""
    if isinstance(image_sidelength, int):
        return (image_sidelength, image_sidelength)
    h, w = image_sidelength[-2], image_sidelength[-1]
    return (int(h), int(w))


def profile_sidelength(image_sidelength: int | tuple[int, int]) -> int:
    """Sidelength ``n = min(h, w)`` that sets the 1D profile length ``n // 2 + 1``.

    This matches the shell count of ``rotational_average_dft_2d`` for non-square
    images.
    """
    return min(as_image_shape(image_sidelength))
