import numpy as np
import pytest
import torch
from torch_tilt_series import TiltSeries

import torch_reconstruct_tomogram
from torch_reconstruct_tomogram import (
    project_points,
    reconstruct_subvolume,
    reconstruct_tomogram,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_tilt_series(tmp_path, device="cpu", size=32):
    mrcfile = pytest.importorskip("mrcfile")
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    images = np.zeros((3, size, size), dtype=np.float32)
    c = size // 2
    images[:, c - 2 : c + 2, c - 2 : c + 2] = 1.0
    image_path = tmp_path / "images.mrc"
    mrcfile.write(image_path, images, overwrite=True)
    return TiltSeries(
        tilt_angles=tilt_angles,
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((3, 2)),
        image_path=image_path,
        pixel_spacing=1.0,
        device=device,
    )


def test_imports_with_version():
    assert isinstance(torch_reconstruct_tomogram.__version__, str)


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume(device, tmp_path):
    tilt_series = make_tilt_series(tmp_path, device)
    point_zyx = torch.tensor([0.0, 0.0, 0.0], device=device)
    subvolume = reconstruct_subvolume(tilt_series, point_zyx, sidelength=8)
    assert subvolume.shape == (8, 8, 8)
    assert subvolume.dtype == torch.float32
    assert device in str(subvolume.device)
    assert torch.isfinite(subvolume).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume_rank_polymorphic(device, tmp_path):
    tilt_series = make_tilt_series(tmp_path, device)

    # single point (3,) -> (d, h, w)
    point = torch.tensor([0.0, 0.0, 0.0], device=device)
    assert reconstruct_subvolume(tilt_series, point, sidelength=8).shape == (8, 8, 8)

    # batch (N, 3) -> (N, d, h, w)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)
    assert reconstruct_subvolume(tilt_series, points, sidelength=8).shape == (
        2,
        8,
        8,
        8,
    )

    # 2D grid (a, b, 3) -> (a, b, d, h, w)
    grid_2d = torch.zeros(2, 3, 3, device=device)
    assert reconstruct_subvolume(tilt_series, grid_2d, sidelength=8).shape == (
        2,
        3,
        8,
        8,
        8,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume_output_pixel_spacing(device, tmp_path):
    tilt_series = make_tilt_series(tmp_path, device)
    point = torch.tensor([0.0, 0.0, 0.0], device=device)

    subvolume_default = reconstruct_subvolume(tilt_series, point, sidelength=8)
    subvolume_explicit = reconstruct_subvolume(
        tilt_series, point, sidelength=8, output_pixel_spacing=1.0
    )
    assert torch.allclose(subvolume_default, subvolume_explicit)

    subvolume_coarse = reconstruct_subvolume(
        tilt_series, point, sidelength=8, output_pixel_spacing=2.0
    )
    assert subvolume_coarse.shape == (8, 8, 8)
    assert torch.isfinite(subvolume_coarse).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume_local_shifts(device, tmp_path):
    tilt_series = make_tilt_series(tmp_path, device)
    point = torch.tensor([0.0, 0.0, 0.0], device=device)

    def zero_local_shifts(projected_yx):
        return torch.zeros_like(projected_yx)

    subvolume_default = reconstruct_subvolume(tilt_series, point, sidelength=8)
    tilt_series.local_shifts_2d = zero_local_shifts
    subvolume_with_hook = reconstruct_subvolume(tilt_series, point, sidelength=8)
    assert torch.allclose(subvolume_default, subvolume_with_hook)


def test_project_points_local_shifts_are_angstroms_not_pixels():
    # pixel_spacing != 1 so an Angstrom-space shift and a pixel-space shift
    # would disagree if local_shifts were (still) being applied in pixels.
    shift_ang = torch.tensor([5.0, -3.0])

    def shift_fn(projected_yx_ang):
        return shift_ang.expand_as(projected_yx_ang)

    tilt_series_shifted = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
        pixel_spacing=2.0,
        local_shifts_2d=shift_fn,
    )
    tilt_series = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
        pixel_spacing=2.0,
    )
    point = torch.tensor([[0.0, 0.0, 0.0]])

    shifted_px = project_points(tilt_series_shifted, point)
    unshifted_px = project_points(tilt_series, point)
    # shift is applied in Angstroms, then the whole result is divided by
    # pixel_spacing so the pixel-space delta is shift_ang / pixel_spacing,
    # not shift_ang itself.
    expected_delta_px = shift_ang / tilt_series.pixel_spacing
    assert torch.allclose(
        (shifted_px - unshifted_px)[0, 0], expected_delta_px, atol=1e-5
    )


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume_preprocess_toggle(device, tmp_path):
    tilt_series = make_tilt_series(tmp_path, device)
    point = torch.tensor([0.0, 0.0, 0.0], device=device)

    preprocessed = reconstruct_subvolume(
        tilt_series, point, sidelength=8, preprocess=True
    )
    raw = reconstruct_subvolume(tilt_series, point, sidelength=8, preprocess=False)
    assert not torch.allclose(preprocessed, raw)


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram_output_pixel_spacing(device, tmp_path):
    tilt_series = make_tilt_series(tmp_path, device)
    volume = reconstruct_tomogram(
        tilt_series, (16, 16, 16), sidelength=8, output_pixel_spacing=2.0
    )
    assert volume.shape == (16, 16, 16)
    assert torch.isfinite(volume).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram(device, tmp_path):
    tilt_series = make_tilt_series(tmp_path, device)
    volume = reconstruct_tomogram(tilt_series, (16, 16, 16), sidelength=8)
    assert volume.shape == (16, 16, 16)
    assert volume.dtype == torch.float32
    assert device in str(volume.device)
    assert torch.isfinite(volume).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram_non_cubic(device, tmp_path):
    tilt_series = make_tilt_series(tmp_path, device)
    # shape not divisible by sidelength is still cropped to the requested shape
    volume = reconstruct_tomogram(tilt_series, (8, 24, 20), sidelength=8)
    assert volume.shape == (8, 24, 20)


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram_batch_size(device, tmp_path):
    tilt_series = make_tilt_series(tmp_path, device)
    recon_no_batch = reconstruct_tomogram(tilt_series, (16, 16, 16), sidelength=8)
    recon_with_batch = reconstruct_tomogram(
        tilt_series, (16, 16, 16), sidelength=8, batch_size=2
    )
    assert recon_no_batch.shape == (16, 16, 16)
    assert recon_with_batch.shape == (16, 16, 16)
    diff = torch.abs(recon_no_batch - recon_with_batch.to(recon_no_batch.device)).max()
    assert diff == 0.0


def test_reconstruct_subvolume_rotation_includes_tomo2sample(tmp_path, monkeypatch):
    """The Fourier-insertion rotation must be the tomogram -> detector
    rotation (projection_matrices' rotation composed with tomo2sample's),
    not just projection_matrices' sample -> detector rotation alone.
    Otherwise every patch is reconstructed oriented to sample space while
    being tiled into an axis-aligned tomogram-space grid, tearing apart at
    patch boundaries whenever sample2tomo carries a rotation. No-op when
    sample2tomo is the default identity, which is why this needs its own
    explicit test rather than relying on the (identity-only) tests above.
    """
    from torch_affine_utils.transforms_3d import Rz

    import torch_reconstruct_tomogram.reconstruct as reconstruct_module

    tilt_series = make_tilt_series(tmp_path)
    tilt_series.levelled2tomo = Rz(torch.tensor(37.0), zyx=True, device="cpu")

    captured = {}
    real_insert = reconstruct_module.insert_central_slices_rfft_3d_multichannel

    def spy(*args, **kwargs):
        captured["rotation_matrices"] = kwargs["rotation_matrices"].clone()
        return real_insert(*args, **kwargs)

    monkeypatch.setattr(
        reconstruct_module, "insert_central_slices_rfft_3d_multichannel", spy
    )

    reconstruct_subvolume(tilt_series, torch.tensor([0.0, 0.0, 0.0]), sidelength=8)

    expected_forward = (
        tilt_series.projection_matrices[:, :3, :3] @ tilt_series.tomo2sample[:3, :3]
    )
    expected = torch.linalg.pinv(expected_forward)
    assert torch.allclose(captured["rotation_matrices"], expected, atol=1e-5)
