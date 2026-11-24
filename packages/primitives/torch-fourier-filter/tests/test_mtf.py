import pytest
import torch

from torch_fourier_filter.mtf import (
    get_mtf_reference_path,
    list_mtf_references,
    make_mtf_grid,
    read_mtf,
)


def test_make_mtf_grid():
    # Test parameters
    image_shape_2d = (64, 64)
    image_shape_3d = (32, 64, 64)
    mtf_frequencies = torch.linspace(0, 0.5, 10)
    mtf_amplitudes = torch.linspace(1, 0, 10)

    mtf_grid_2d = make_mtf_grid(
        image_shape_2d, mtf_frequencies, mtf_amplitudes, rfft=True, fftshift=False
    )
    mtf_grid_3d = make_mtf_grid(
        image_shape_3d, mtf_frequencies, mtf_amplitudes, rfft=True, fftshift=False
    )

    # Test if the MTF grid values are within the expected range
    assert torch.all(mtf_grid_2d >= 0) and torch.all(mtf_grid_2d <= 1), (
        "MTF grid values out of range for 2D input"
    )
    assert torch.all(mtf_grid_3d >= 0) and torch.all(mtf_grid_3d <= 1), (
        "MTF grid values out of range for 3D input"
    )


def test_bundled_mtf_references():
    names = list_mtf_references()
    assert len(names) == 13
    assert all(name == name.lower() for name in names)
    for name in names:
        path = get_mtf_reference_path(name)
        assert path.is_file()
        frequencies, amplitudes = read_mtf(name)
        assert frequencies.shape == amplitudes.shape
        assert frequencies.ndim == 1
        # Some vendor curves sample slightly past Nyquist (e.g. 0.508)
        assert torch.all(frequencies >= 0) and torch.all(frequencies <= 0.51)
        assert torch.all(torch.diff(frequencies) > 0)


def test_mtf_reference_lookup_is_case_insensitive():
    assert get_mtf_reference_path("K2_300kV") == get_mtf_reference_path("k2_300kv")
    freqs_upper, amps_upper = read_mtf("Falcon4EC_300kV")
    freqs_lower, amps_lower = read_mtf("falcon4ec_300kv")
    assert torch.equal(freqs_upper, freqs_lower)
    assert torch.equal(amps_upper, amps_lower)


def test_read_mtf_from_path_matches_name():
    path = get_mtf_reference_path("k3_standard_300kv_fl2")
    assert all(
        torch.equal(a, b)
        for a, b in zip(read_mtf(str(path)), read_mtf("k3_standard_300kv_fl2"))
    )


def test_unknown_mtf_reference_raises():
    with pytest.raises(KeyError, match="Unknown MTF reference"):
        get_mtf_reference_path("not_a_camera")
