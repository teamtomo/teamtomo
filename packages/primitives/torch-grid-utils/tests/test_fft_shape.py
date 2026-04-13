from torch_grid_utils import next_fft_size, rfft_shape
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
