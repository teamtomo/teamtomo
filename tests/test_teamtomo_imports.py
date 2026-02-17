"""Check that all modules are importable under the teamtomo namespace."""

import pytest


def test_primitives_import():
    """Ensure teamtomo.primitives can be imported."""
    from teamtomo import primitives

    assert primitives is not None


def test_wip_imports():
    """Ensure teamtomo.wip modules can be imported."""
    from teamtomo import wip

    assert wip is not None


def test_individual_primitive_imports():
    """Test that individual primitive packages can be imported."""
    from teamtomo.primitives import (
        torch_affine_utils,
        torch_ctf,
        torch_cubic_spline_grids,
        torch_find_peaks,
        torch_fourier_filter,
        torch_fourier_rescale,
        torch_fourier_shell_correlation,
        torch_fourier_shift,
        torch_fourier_slice,
        torch_grid_utils,
        torch_image_interpolation,
        torch_so3,
        torch_subpixel_crop,
        torch_transform_image,
    )

    assert torch_affine_utils is not None
    assert torch_ctf is not None
    assert torch_cubic_spline_grids is not None
    assert torch_find_peaks is not None
    assert torch_fourier_filter is not None
    assert torch_fourier_rescale is not None
    assert torch_fourier_shell_correlation is not None
    assert torch_fourier_shift is not None
    assert torch_fourier_slice is not None
    assert torch_grid_utils is not None
    assert torch_image_interpolation is not None
    assert torch_so3 is not None
    assert torch_subpixel_crop is not None
    assert torch_transform_image is not None


def test_all_primitives_available():
    """Verify all primitive packages are in the namespace."""
    from teamtomo import primitives

    expected_packages = [
        "torch_affine_utils",
        "torch_ctf",
        "torch_cubic_spline_grids",
        "torch_find_peaks",
        "torch_fourier_filter",
        "torch_fourier_rescale",
        "torch_fourier_shell_correlation",
        "torch_fourier_shift",
        "torch_fourier_slice",
        "torch_grid_utils",
        "torch_image_interpolation",
        "torch_so3",
        "torch_subpixel_crop",
        "torch_transform_image",
    ]

    for pkg in expected_packages:
        assert hasattr(primitives, pkg), f"Missing package: {pkg}"


def test_wip_packages_available():
    """Verify WIP packages are in the namespace."""
    from teamtomo import wip

    assert hasattr(wip, "torch_tilt_series")
