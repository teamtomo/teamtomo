"""Tests for prepare_image and patch/power-spectrum primitives."""

import torch

from torch_ctf_estimation.utils.patches import (
    compute_patch_power_spectra,
    extract_ctf_patches,
    normalised_patch_positions,
)
from torch_ctf_estimation.utils.prepare_image import prepare_image_for_ctf


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
    assert sidelength == 32
    assert patches.ndim == 5
    assert centers.shape[-1] == 3
    positions = normalised_patch_positions(centers, image.shape)
    assert positions.min() >= 0.0
    assert positions.max() <= 1.0


def test_extract_ctf_patches_whole_image():
    image = torch.randn(1, 48, 48)
    patches, _centers, sidelength, whole = extract_ctf_patches(
        image, patch_sidelength=-1, defocus_grid_resolution=(1, 1, 1)
    )
    assert whole
    assert sidelength == 48
    assert patches.shape[-2:] == (48, 48)


def test_compute_patch_power_spectra_shapes():
    patches = torch.randn(1, 2, 2, 16, 16)
    patch_ps, mean_ps = compute_patch_power_spectra(patches)
    assert patch_ps.shape == (1, 2, 2, 16, 9)
    assert mean_ps.shape == (16, 9)
