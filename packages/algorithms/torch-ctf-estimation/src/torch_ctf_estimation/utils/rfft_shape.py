"""Recover the real-space shape behind an rfft-layout spectrum."""

from __future__ import annotations


def resolve_rfft_image_shape(
    rfft_shape: tuple[int, int],
    image_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Real-space ``(h, w)`` for an rfft-layout array of shape ``(h, w // 2 + 1)``.

    The width of an rfft is ambiguous: real widths ``2k`` and ``2k + 1`` both give
    ``k + 1`` columns. Pass ``image_shape`` to resolve it. Without it, a square
    image is assumed when ``h // 2 + 1`` matches the rfft width (correct for odd
    and even ``h``). Otherwise the even width ``2 * (w_rfft - 1)`` is assumed.

    Parameters
    ----------
    rfft_shape : tuple[int, int]
        ``(h, w_rfft)``, the last two dimensions of the rfft-layout array.
    image_shape : tuple[int, int] | None, optional
        Real-space ``(h, w)``. Validated against ``rfft_shape`` when given.

    Returns
    -------
    tuple[int, int]
        Real-space ``(h, w)``.

    Raises
    ------
    ValueError
        If ``image_shape`` does not match ``rfft_shape``.
    """
    h, w_rfft = int(rfft_shape[0]), int(rfft_shape[1])
    if image_shape is None:
        if h // 2 + 1 == w_rfft:
            return (h, h)
        return (h, 2 * (w_rfft - 1))
    ih, iw = int(image_shape[-2]), int(image_shape[-1])
    if ih != h or iw // 2 + 1 != w_rfft:
        raise ValueError(
            f"image_shape {(ih, iw)} is incompatible with an rfft of shape "
            f"{(h, w_rfft)}; expected (h, w) with w // 2 + 1 == {w_rfft}."
        )
    return (ih, iw)
