import numpy as np
import pytest
import torch
from torch_affine_utils.transforms_3d import Rz, T

import torch_tilt_series
from torch_tilt_series import TiltSeries

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_tilt_series(device="cpu"):
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))
    return TiltSeries(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        device=device,
    )


def test_imports_with_version():
    assert isinstance(torch_tilt_series.__version__, str)


@pytest.mark.parametrize("device", DEVICES)
def test_construction(device):
    ts = make_tilt_series(device)
    assert ts.tilt_angles.shape == (3,)
    assert ts.tilt_angles.dtype == torch.float32
    assert device in str(ts.tilt_angles.device)
    assert ts.image_path is None
    assert ts.image_indices is None
    with pytest.raises(ValueError, match="pixel_spacing is not set"):
        _ = ts.pixel_spacing


@pytest.mark.parametrize("device", DEVICES)
def test_projection_matrices(device):
    ts = make_tilt_series(device)
    matrices = ts.projection_matrices
    assert matrices.shape == (3, 4, 4)
    assert matrices.dtype == torch.float32
    assert device in str(matrices.device)


def test_projection_matrix_identity_at_zero_geometry():
    ts = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
    )
    assert torch.allclose(ts.projection_matrices[0], torch.eye(4), atol=1e-6)


def test_x_tilt_default_is_identity():
    # x_tilts defaults to 0, so the geometry must be unchanged from the 2-angle model.
    ts = make_tilt_series()
    assert torch.allclose(ts.x_tilts, torch.zeros_like(ts.x_tilts))
    rx = ts.projection_matrices  # noqa: F841 - exercises the Rx code path at angle 0
    expected = TiltSeries(
        tilt_angles=torch.tensor([-30.0, 0.0, 30.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((3, 2)),
    ).projection_matrices
    assert torch.allclose(ts.projection_matrices, expected, atol=1e-6)


def test_x_tilt_rotates_about_x_axis():
    # A pure x-axis tilt (no stage tilt / no in-plane angle) must rotate points
    # about the X axis: the X coordinate is invariant, while (z, y) mix.
    ts = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
        x_tilts=90.0,
    )
    rot = ts.projection_matrices[0, :3, :3]  # zyx rotation block
    # point on the X axis (z=0, y=0, x=1) is unchanged by a rotation about X
    x_axis = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(rot @ x_axis, x_axis, atol=1e-6)
    # a point on the Y axis (z=0, y=1, x=0) maps onto the Z axis under a +90 deg turn
    y_axis = torch.tensor([0.0, 1.0, 0.0])
    mapped = rot @ y_axis
    assert torch.allclose(mapped[2], torch.tensor(0.0), atol=1e-6)  # x stays 0
    assert torch.isclose(torch.linalg.norm(mapped), torch.tensor(1.0), atol=1e-6)


def test_x_tilt_per_view_shape():
    ts = TiltSeries(
        tilt_angles=torch.tensor([-30.0, 0.0, 30.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((3, 2)),
        x_tilts=torch.tensor([-0.37, -0.30, -0.25]),
    )
    assert ts.projection_matrices.shape == (3, 4, 4)


def test_sample2scope_is_pure_rotation():
    ts = TiltSeries(
        tilt_angles=torch.tensor([-30.0, 0.0, 30.0]),
        tilt_axis_angle=torch.tensor([10.0, -5.0, 20.0]),
        sample_translations=torch.rand((3, 2)) * 10,
        x_tilts=torch.tensor([-0.4, 0.1, 0.3]),
    )
    m = ts.sample2scope
    assert m.shape == (3, 4, 4)
    # translation column is exactly zero
    assert torch.allclose(m[:, :3, 3], torch.zeros(3, 3), atol=1e-6)
    # 3x3 rotation block is orthogonal: R @ R.T == I
    rot = m[:, :3, :3]
    eye = torch.eye(3).expand(3, 3, 3)
    assert torch.allclose(rot @ rot.transpose(-1, -2), eye, atol=1e-5)


def test_scope2detector_only_mixes_yx():
    ts = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor([37.0]),
        sample_translations=torch.tensor([[3.0, -2.0]]),
    )
    m = ts.scope2detector[0]
    # z row: output z depends only on input z, unit coefficient, no shift
    assert torch.allclose(m[0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    # no y/x output depends on z input
    assert torch.allclose(m[1:3, 0], torch.zeros(2), atol=1e-6)
    # y/x block is a proper 2D rotation (orthogonal) plus the known shift
    rot_yx = m[1:3, 1:3]
    assert torch.allclose(rot_yx @ rot_yx.T, torch.eye(2), atol=1e-5)
    assert torch.allclose(m[1:3, 3], torch.tensor([3.0, -2.0]), atol=1e-6)


def test_projection_matrices_equals_composition():
    torch.manual_seed(0)
    n = 5
    ts = TiltSeries(
        tilt_angles=torch.rand(n) * 60 - 30,
        tilt_axis_angle=torch.rand(n) * 360,
        sample_translations=torch.randn(n, 2) * 5,
        x_tilts=torch.randn(n) * 2,
    )
    assert torch.allclose(
        ts.projection_matrices, ts.scope2detector @ ts.sample2scope, atol=1e-6
    )


def test_scope2sample_is_inverse_of_sample2scope():
    ts = TiltSeries(
        tilt_angles=torch.tensor([-30.0, 0.0, 30.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((3, 2)),
        x_tilts=torch.tensor([-0.4, 0.1, 0.3]),
    )
    eye = torch.eye(4).expand(3, 4, 4)
    assert torch.allclose(ts.sample2scope @ ts.scope2sample, eye, atol=1e-5)


def test_detector2scope_is_inverse_of_scope2detector():
    ts = TiltSeries(
        tilt_angles=torch.tensor([-30.0, 0.0, 30.0]),
        tilt_axis_angle=torch.tensor([10.0, -5.0, 20.0]),
        sample_translations=torch.randn(3, 2) * 5,
    )
    eye = torch.eye(4).expand(3, 4, 4)
    assert torch.allclose(ts.scope2detector @ ts.detector2scope, eye, atol=1e-5)


def test_tomo2sample_is_inverse_of_sample2tomo():
    ts_default = make_tilt_series()
    assert torch.allclose(
        ts_default.sample2tomo @ ts_default.tomo2sample, torch.eye(4), atol=1e-6
    )

    levelled2tomo = T(torch.tensor([1.0, -2.0, 3.0]), device="cpu") @ Rz(
        torch.tensor(40.0), zyx=True, device="cpu"
    )
    ts = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
        levelled2tomo=levelled2tomo,
    )
    assert torch.allclose(ts.sample2tomo @ ts.tomo2sample, torch.eye(4), atol=1e-5)


def test_sample2tomo_default_is_identity_no_behavior_change():
    ts = make_tilt_series()
    assert torch.allclose(ts.sample2tomo, torch.eye(4))
    assert torch.allclose(ts.tomo2sample, torch.eye(4))
    points = torch.tensor([[0.0, 7.0, -4.0], [1.0, -2.0, 3.0]])
    with_default = ts.project_points(points)
    ts_explicit = TiltSeries(
        tilt_angles=torch.tensor([-30.0, 0.0, 30.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((3, 2)),
        sample2levelled=torch.eye(4),
        levelled2tomo=torch.eye(4),
    )
    assert torch.allclose(with_default, ts_explicit.project_points(points), atol=1e-6)


def test_sample2tomo_pure_translation_shifts_projection():
    # sample2tomo = T(shift) maps p_sample -> p_sample + shift, i.e. a point
    # at sample-space position p has tomogram-space coordinate p + shift.
    # So tomo2sample = T(-shift): recovering the sample-space point from a
    # tomogram-space coordinate requires *subtracting* the shift.
    shift_zyx = torch.tensor([2.0, -3.0, 5.0])
    levelled2tomo = T(shift_zyx, device="cpu")
    ts_notomo = make_tilt_series()
    ts_tomo = TiltSeries(
        tilt_angles=torch.tensor([-30.0, 0.0, 30.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((3, 2)),
        levelled2tomo=levelled2tomo,
    )
    point_tomo = torch.tensor([[0.0, 0.0, 0.0]])
    point_sample_equivalent = point_tomo - shift_zyx
    assert torch.allclose(
        ts_tomo.project_points(point_tomo),
        ts_notomo.project_points(point_sample_equivalent),
        atol=1e-5,
    )


def test_sample2tomo_pure_rotation_matches_manual_transform():
    # A 90 degree Rz sample2tomo: points_tomo -> tomo2sample -> points_sample
    # should equal applying tomo2sample directly, computed independently.
    levelled2tomo = Rz(torch.tensor(90.0), zyx=True, device="cpu")
    ts_notomo = make_tilt_series()
    ts_tomo = TiltSeries(
        tilt_angles=torch.tensor([-30.0, 0.0, 30.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((3, 2)),
        levelled2tomo=levelled2tomo,
    )
    point_tomo = torch.tensor([[0.0, 1.0, 0.0]])  # pure +y in tomo space
    tomo2sample = torch.linalg.inv(levelled2tomo)
    point_tomo_w = torch.cat([point_tomo, torch.ones(1, 1)], dim=-1)
    point_sample = (point_tomo_w @ tomo2sample.T)[:, :3]
    assert torch.allclose(
        ts_tomo.project_points(point_tomo),
        ts_notomo.project_points(point_sample),
        atol=1e-5,
    )


def test_sample2tomo_is_composition_of_sample2levelled_and_levelled2tomo():
    sample2levelled = Rz(torch.tensor(15.0), zyx=True, device="cpu")
    levelled2tomo = T(torch.tensor([1.0, -2.0, 3.0]), device="cpu") @ Rz(
        torch.tensor(40.0), zyx=True, device="cpu"
    )
    ts = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
        sample2levelled=sample2levelled,
        levelled2tomo=levelled2tomo,
    )
    assert torch.allclose(ts.sample2tomo, levelled2tomo @ sample2levelled, atol=1e-6)
    assert torch.allclose(
        ts.tomo2sample, torch.linalg.inv(levelled2tomo @ sample2levelled), atol=1e-5
    )


def test_levelled2sample_is_inverse_of_sample2levelled():
    ts_default = make_tilt_series()
    assert torch.allclose(
        ts_default.sample2levelled @ ts_default.levelled2sample,
        torch.eye(4),
        atol=1e-6,
    )

    sample2levelled = Rz(torch.tensor(15.0), zyx=True, device="cpu")
    ts = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
        sample2levelled=sample2levelled,
    )
    assert torch.allclose(
        ts.sample2levelled @ ts.levelled2sample, torch.eye(4), atol=1e-5
    )


def test_tomo2levelled_is_inverse_of_levelled2tomo():
    ts_default = make_tilt_series()
    assert torch.allclose(
        ts_default.levelled2tomo @ ts_default.tomo2levelled, torch.eye(4), atol=1e-6
    )

    levelled2tomo = T(torch.tensor([1.0, -2.0, 3.0]), device="cpu")
    ts = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
        levelled2tomo=levelled2tomo,
    )
    assert torch.allclose(ts.levelled2tomo @ ts.tomo2levelled, torch.eye(4), atol=1e-5)


def test_sample2levelled_is_protected_from_levelled2tomo_reassignment():
    # The whole point of the split: reassigning levelled2tomo (an arbitrary,
    # user-owned reframing choice) must never touch sample2levelled (a fixed,
    # data-derived correction).
    sample2levelled = Rz(torch.tensor(15.0), zyx=True, device="cpu")
    ts = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
        sample2levelled=sample2levelled,
    )
    ts.levelled2tomo = T(torch.tensor([5.0, 0.0, 0.0]), device="cpu")
    assert torch.allclose(ts.sample2levelled, sample2levelled, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_project_points_origin(device):
    ts = make_tilt_series(device)
    points_zyx = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    projected_yx = ts.project_points(points_zyx)
    assert projected_yx.shape == (1, 3, 2)
    assert projected_yx.dtype == torch.float32
    assert device in str(projected_yx.device)
    assert torch.allclose(projected_yx, torch.zeros_like(projected_yx), atol=1e-5)


def test_project_points_value_at_zero_tilt():
    ts = make_tilt_series()
    point = torch.tensor([[0.0, 7.0, -4.0]])
    projected_yx = ts.project_points(point)
    # tilt index 1 has tilt_angle == 0, so the in-plane (y, x) is preserved
    assert torch.allclose(projected_yx[0, 1], torch.tensor([7.0, -4.0]), atol=1e-5)


def test_project_points_batch_shapes():
    ts = make_tilt_series()
    points = torch.zeros((5, 3))
    assert ts.project_points(points).shape == (5, 3, 2)


def test_project_points_output_zyxw_round_trips_to_sample_space():
    ts = make_tilt_series()
    point_sample = torch.tensor([[0.0, 7.0, -4.0]])
    zyxw = ts.project_points(point_sample, output_zyxw=True)
    assert zyxw.shape == (1, 3, 4)

    # dropping to yx must match the default (2D) output exactly
    yx = ts.project_points(point_sample)
    assert torch.allclose(zyxw[..., [1, 2]], yx, atol=1e-6)

    # round trip: detector2scope then scope2sample recovers the original point
    recovered = ts.scope2sample @ ts.detector2scope @ zyxw[..., None]
    recovered = recovered[..., :3, 0]  # (n_points, n_tilts, 3)
    expected = point_sample[:, None, :].expand(-1, 3, -1)
    assert torch.allclose(recovered, expected, atol=1e-4)


def test_project_points_local_shifts_applied_in_sample_space():
    ts = make_tilt_series()
    point = torch.tensor([[0.0, 7.0, -4.0]])
    shift = torch.tensor([1.0, -2.0, 3.0])

    def shift_fn(points_sample):
        return shift.expand_as(points_sample)

    shifted = ts.project_points(point, local_shifts=shift_fn)
    expected = ts.project_points(point + shift)
    assert torch.allclose(shifted, expected, atol=1e-5)


def test_pixel_spacing_setter():
    ts = make_tilt_series()
    with pytest.raises(ValueError, match="pixel_spacing is not set"):
        _ = ts.pixel_spacing
    ts.pixel_spacing = 3.5
    assert ts.pixel_spacing == 3.5
    ts.pixel_spacing = None
    with pytest.raises(ValueError, match="pixel_spacing is not set"):
        _ = ts.pixel_spacing


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_move():
    ts = make_tilt_series("cpu")
    assert "cpu" == str(ts.tilt_angles.device)
    ts.to("cuda")
    assert "cuda" in str(ts.tilt_angles.device)
    assert "cuda" in str(ts.tilt_axis_angle.device)
    assert "cuda" in str(ts.sample_translations.device)
    assert "cuda" in str(ts.x_tilts.device)
    assert "cuda" in str(ts.sample2levelled.device)
    assert "cuda" in str(ts.levelled2tomo.device)


def test_from_aretomo_output(tmp_path):
    pytest.importorskip("alnfile")

    aln = (
        "# AreTomo Alignment / Priims bprmMn\n"
        "# RawSize = 16 16 3\n"
        "# NumPatches = 0\n"
        "# AlphaOffset =     0.00\n"
        "# BetaOffset =      0.00\n"
        "# SEC     ROT      GMAG     TX       TY     SMEAN   SFIT  SCALE  BASE   TILT\n"
        "    1   0.0000  1.00000   1.000   2.000   1.00   1.00  1.00  0.00  -30.00\n"
        "    2   0.0000  1.00000   0.000   0.000   1.00   1.00  1.00  0.00    0.00\n"
        "    3   0.0000  1.00000  -1.000  -2.000   1.00   1.00  1.00  0.00   30.00\n"
    )
    aln_path = tmp_path / "ts.aln"
    aln_path.write_text(aln)

    pixel_spacing = 2.0
    ts = TiltSeries.from_aretomo_output(aln_path, pixel_spacing=pixel_spacing)

    assert torch.allclose(ts.tilt_angles, torch.tensor([-30.0, 0.0, 30.0]))
    assert torch.allclose(ts.tilt_axis_angle, torch.zeros(3))
    # tx/ty are stored as (y, x) in Angstroms == (ty, tx) * pixel_spacing
    expected = torch.tensor([[2.0, 1.0], [0.0, 0.0], [-2.0, -1.0]]) * pixel_spacing
    assert torch.allclose(ts.sample_translations, expected)
    # image loading metadata: never read here, only resolved
    assert ts.image_path == aln_path.with_suffix(".mrc")
    assert torch.equal(ts.image_indices, torch.tensor([0, 1, 2]))
    assert ts.pixel_spacing == pixel_spacing


def test_from_etomo_directory(tmp_path):
    pytest.importorskip("etomofiles")
    mrcfile = pytest.importorskip("mrcfile")

    (tmp_path / "ts.edf").write_text(
        "Setup.DatasetName=ts\nSetup.RawImageStackExt=st\nSetup.ImageRotationA=0.0\n"
    )
    (tmp_path / "ts.tlt").write_text("-30.0\n0.0\n30.0\n")
    (tmp_path / "ts.rawtlt").write_text("-30.0\n0.0\n30.0\n")
    (tmp_path / "ts.xtilt").write_text("0.0\n0.0\n0.0\n")
    (tmp_path / "ts.xf").write_text(
        "1.0 0.0 0.0 1.0  1.0  2.0\n"
        "1.0 0.0 0.0 1.0  0.0  0.0\n"
        "1.0 0.0 0.0 1.0 -1.0 -2.0\n"
    )
    # etomofiles.read() reads the MRC header to determine image count, even
    # though torch_tilt_series never reads the pixel data itself.
    images = np.zeros((3, 16, 16), dtype=np.float32)
    mrcfile.write(tmp_path / "ts.st", images, overwrite=True)

    pixel_spacing = 2.0
    ts = TiltSeries.from_etomo_directory(tmp_path, pixel_spacing=pixel_spacing)

    assert torch.allclose(ts.tilt_angles, torch.tensor([-30.0, 0.0, 30.0]))
    assert torch.allclose(ts.tilt_axis_angle, torch.zeros(3))
    expected = torch.tensor([[-4.0, -2.0], [0.0, 0.0], [4.0, 2.0]])
    assert torch.allclose(ts.sample_translations, expected)
    assert torch.allclose(ts.x_tilts, torch.zeros(3))
    # image loading metadata: never read here, only resolved
    assert ts.image_path == tmp_path / "ts.st"
    assert ts.image_indices is not None
    assert ts.image_indices.shape == (3,)
    assert ts.pixel_spacing == pixel_spacing


def test_from_etomo_directory_reads_xtilt(tmp_path):
    pytest.importorskip("etomofiles")
    mrcfile = pytest.importorskip("mrcfile")

    (tmp_path / "ts.edf").write_text(
        "Setup.DatasetName=ts\nSetup.RawImageStackExt=st\nSetup.ImageRotationA=0.0\n"
    )
    (tmp_path / "ts.tlt").write_text("-30.0\n0.0\n30.0\n")
    (tmp_path / "ts.rawtlt").write_text("-30.0\n0.0\n30.0\n")
    (tmp_path / "ts.xtilt").write_text("-0.37\n-0.30\n-0.25\n")
    (tmp_path / "ts.xf").write_text(
        "1.0 0.0 0.0 1.0  0.0  0.0\n"
        "1.0 0.0 0.0 1.0  0.0  0.0\n"
        "1.0 0.0 0.0 1.0  0.0  0.0\n"
    )
    images = np.zeros((3, 16, 16), dtype=np.float32)
    mrcfile.write(tmp_path / "ts.st", images, overwrite=True)

    ts = TiltSeries.from_etomo_directory(tmp_path, pixel_spacing=2.0)
    assert torch.allclose(ts.x_tilts, torch.tensor([-0.37, -0.30, -0.25]), atol=1e-5)
