"""Tests for prepare_image and patch/power-spectrum primitives."""

import warnings

import numpy as np
import pytest
import torch

from torch_ctf_estimation.utils.patches import (
    compute_patch_power_spectra,
    extract_ctf_patches,
    normalised_patch_positions,
)
from torch_ctf_estimation.utils.prepare_image import prepare_image_for_ctf
from torch_ctf_estimation.utils.rfft_shape import as_image_shape


def test_prepare_image_for_ctf_packs_and_rescales():
    image = torch.randn(64, 64)
    prepared, spacing = prepare_image_for_ctf(
        image,
        pixel_spacing_angstroms=1.0,
        target_pixel_spacing_angstroms=2.0,
        device=torch.device("cpu"),
    )
    assert prepared.ndim == 3
    assert prepared.shape[0] == 1
    assert spacing == 2.0


def test_prepare_image_never_upsamples():
    image = torch.randn(32, 32)
    _prepared, spacing = prepare_image_for_ctf(
        image,
        pixel_spacing_angstroms=4.0,
        target_pixel_spacing_angstroms=2.0,
        device=torch.device("cpu"),
    )
    assert spacing == 4.0


def test_extract_ctf_patches_overlapping():
    image = torch.randn(1, 64, 64)
    patches, centers, sidelength, whole = extract_ctf_patches(
        image, patch_sidelength=32, defocus_grid_resolution=(1, 2, 2)
    )
    assert not whole
    assert sidelength == (32, 32)
    assert patches.ndim == 5
    assert centers.shape[-1] == 3
    positions = normalised_patch_positions(centers, image.shape)
    assert positions.min() >= 0.0
    assert positions.max() <= 1.0


@pytest.mark.parametrize("shape", [(48, 48), (48, 61), (61, 48)])
def test_extract_ctf_patches_whole_image(shape):
    image = torch.randn(1, *shape)
    patches, _centers, patch_shape, whole = extract_ctf_patches(
        image, patch_sidelength=-1, defocus_grid_resolution=(1, 1, 1)
    )
    assert whole
    assert patch_shape == shape
    assert patches.shape[-2:] == shape


def test_compute_patch_power_spectra_shapes():
    patches = torch.randn(1, 2, 2, 16, 16)
    patch_ps, mean_ps = compute_patch_power_spectra(patches)
    assert patch_ps.shape == (1, 2, 2, 16, 9)
    assert mean_ps.shape == (16, 9)


def test_prepare_image_returns_achieved_spacing():
    """The returned spacing is what the rescale achieved, not the target."""
    image = torch.randn(409, 409)
    prepared, spacing = prepare_image_for_ctf(
        image,
        pixel_spacing_angstroms=1.0,
        target_pixel_spacing_angstroms=3.0,
        device=torch.device("cpu"),
    )
    assert prepared.shape[-2:] == (136, 136)
    assert spacing == pytest.approx(409 / 136)


def test_prepare_image_small_spacing_mismatch_does_not_warn():
    """h/w spacings 5e-4 apart (relative) are within tolerance: no warning."""
    image = torch.randn(2000, 900)  # 2000 / 667 vs 900 / 300 Angstrom
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _prepared, spacing = prepare_image_for_ctf(
            image,
            pixel_spacing_angstroms=1.0,
            target_pixel_spacing_angstroms=3.0,
            device=torch.device("cpu"),
        )
    assert spacing == pytest.approx(0.5 * (2000 / 667 + 3.0))


def test_prepare_image_large_spacing_mismatch_warns():
    """h/w spacings 1e-2 apart (relative) warn and return their mean."""
    image = torch.randn(100, 300)  # 100 / 33 vs 300 / 100 Angstrom
    with pytest.warns(UserWarning, match="differs between h"):
        _prepared, spacing = prepare_image_for_ctf(
            image,
            pixel_spacing_angstroms=1.0,
            target_pixel_spacing_angstroms=3.0,
            device=torch.device("cpu"),
        )
    assert spacing == pytest.approx(0.5 * (100 / 33 + 3.0))


@pytest.mark.parametrize(
    "sidelength", [48, np.int64(48), torch.tensor(48), torch.tensor(48).int()]
)
def test_as_image_shape_accepts_integer_scalars(sidelength):
    assert as_image_shape(sidelength) == (48, 48)


@pytest.mark.parametrize(
    "shape", [(48, 61), [48, 61], torch.Size([2, 48, 61]), np.array([48, 61])]
)
def test_as_image_shape_takes_last_two_of_a_shape(shape):
    assert as_image_shape(shape) == (48, 61)
