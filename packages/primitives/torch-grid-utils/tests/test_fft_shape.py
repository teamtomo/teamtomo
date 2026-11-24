import pytest
import torch

from torch_grid_utils import image_shape_from_dft_shape, next_fft_size, rfft_shape
from torch_grid_utils.fftfreq_grid import rfft_shape as rfft_shape_from_fftfreq


def test_rfft_shape_2d():
    assert rfft_shape((8, 16)) == (8, 9)
    assert rfft_shape_from_fftfreq((8, 16)) == (8, 9)


def test_rfft_shape_3d():
    assert rfft_shape((4, 8, 16)) == (4, 8, 9)


def test_next_fft_size_edge_cases():
    assert next_fft_size(0) == 1
    assert next_fft_size(1) == 1


def test_next_fft_size_powers_of_small_primes():
    assert next_fft_size(7) == 7
    assert next_fft_size(8) == 8
    assert next_fft_size(9) == 9  # 3^2
    assert next_fft_size(10) == 10  # 2 * 5
    assert next_fft_size(14) == 14  # 2 * 7
    assert next_fft_size(49) == 49  # 7^2


def test_next_fft_size_uses_torch_meshgrid():
    """Sanity check: result uses only primes 2, 3, 5, 7 and is >= n."""
    for n in [17, 100, 127, 256, 1000]:
        m = next_fft_size(n)
        assert m >= n
        x = m
        for p in (2, 3, 5, 7):
            while x % p == 0:
                x //= p
        assert x == 1


def test_next_fft_size_matches_numpy_reference_small():
    """Cross-check a few values against explicit enumeration."""

    # minimal {2,3,5,7}-smooth >= n for small n
    def brute(n: int) -> int:
        if n <= 1:
            return 1
        m = n
        while True:
            x = m
            for p in (2, 3, 5, 7):
                while x % p == 0:
                    x //= p
            if x == 1:
                return m
            m += 1

    for n in range(2, 500):
        assert next_fft_size(n) == brute(n), f"n={n}"


@pytest.mark.parametrize(
    "image_shape", [(20, 20), (21, 21), (20, 21), (21, 20), (5, 20, 21), (7,)]
)
def test_image_shape_from_dft_shape_with_image_shape(image_shape):
    """An explicit image_shape is validated against the DFT and returned."""
    image = torch.zeros(image_shape)
    rfft_dft_shape = torch.fft.rfftn(image).shape
    fft_dft_shape = torch.fft.fftn(image).shape
    assert (
        image_shape_from_dft_shape(rfft_dft_shape, rfft=True, image_shape=image_shape)
        == image_shape
    )
    assert (
        image_shape_from_dft_shape(fft_dft_shape, rfft=False, image_shape=image_shape)
        == image_shape
    )


@pytest.mark.parametrize(
    ("dft_shape", "rfft", "image_shape"),
    [
        ((20, 11), True, (20, 22)),  # width 22 gives 12 columns
        ((20, 11), True, (21, 20)),  # height mismatch
        ((20, 11), True, (20, 20, 20)),  # dimensionality mismatch
        ((20, 20), False, (20, 21)),
    ],
)
def test_image_shape_from_dft_shape_rejects_incompatible_image_shape(
    dft_shape, rfft, image_shape
):
    with pytest.raises(ValueError, match="does not match"):
        image_shape_from_dft_shape(dft_shape, rfft=rfft, image_shape=image_shape)


@pytest.mark.parametrize("dft_shape", [(20, 20), (21, 20), (5, 20, 21)])
def test_image_shape_from_dft_shape_full_dft_is_unambiguous(dft_shape):
    assert image_shape_from_dft_shape(dft_shape, rfft=False) == dft_shape


@pytest.mark.parametrize("image_shape", [(20, 20), (21, 21), (8, 20, 20), (9, 21, 21)])
def test_image_shape_from_dft_shape_assume_square(image_shape):
    """Square/cubic images of either parity are recovered from their rfft."""
    dft_shape = torch.fft.rfftn(torch.zeros(image_shape)).shape
    assert (
        image_shape_from_dft_shape(dft_shape, rfft=True, assume_square=True)
        == image_shape
    )


@pytest.mark.parametrize("dft_shape", [(20, 12), (21, 10), (11,)])
def test_image_shape_from_dft_shape_assume_square_rejects_non_square(dft_shape):
    with pytest.raises(ValueError, match="square/cubic"):
        image_shape_from_dft_shape(dft_shape, rfft=True, assume_square=True)


@pytest.mark.parametrize("dft_shape", [(20, 11), (21, 11), (11,)])
def test_image_shape_from_dft_shape_ambiguous_rfft_raises(dft_shape):
    """(20, 11) is the rfft of both (20, 20) and (20, 21), so no width is assumed."""
    with pytest.raises(ValueError, match="ambiguous"):
        image_shape_from_dft_shape(dft_shape, rfft=True)
