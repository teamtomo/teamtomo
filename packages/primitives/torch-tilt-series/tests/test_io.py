import numpy as np
import pytest
import torch

from torch_tilt_series import TiltSeries, from_warp_xml, load_tilt_series_images


def test_load_tilt_series_images_uses_path_and_indices(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    raw = np.arange(4 * 8 * 8, dtype=np.float32).reshape(4, 8, 8)
    mrcfile.write(tmp_path / "stack.mrc", raw, overwrite=True)

    tilt_series = TiltSeries(
        tilt_angles=torch.tensor([0.0, 0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((2, 2)),
        image_path=tmp_path / "stack.mrc",
        image_indices=torch.tensor([3, 1]),
    )
    images = load_tilt_series_images(tilt_series)
    assert images.shape == (2, 8, 8)
    assert images.dtype == torch.float32
    assert torch.allclose(images[0], torch.as_tensor(raw[3]))
    assert torch.allclose(images[1], torch.as_tensor(raw[1]))


def test_load_tilt_series_images_without_indices_returns_full_stack(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    raw = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    mrcfile.write(tmp_path / "stack.mrc", raw, overwrite=True)

    tilt_series = TiltSeries(
        tilt_angles=torch.zeros(3),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((3, 2)),
        image_path=tmp_path / "stack.mrc",
    )
    images = load_tilt_series_images(tilt_series)
    assert torch.allclose(images, torch.as_tensor(raw))


def test_load_tilt_series_images_requires_image_path():
    tilt_series = TiltSeries(
        tilt_angles=torch.tensor([0.0]),
        tilt_axis_angle=torch.tensor(0.0),
        sample_translations=torch.zeros((1, 2)),
    )
    with pytest.raises(ValueError):
        load_tilt_series_images(tilt_series)


def _warp_xml(n_tilts=3, use_tilt=None, grid_x_value="0"):
    """Minimal Warp-style tilt-series XML: per-tilt lists, CTF block, 1x1xT grids."""
    angles = [-30.0, 0.0, 30.0][:n_tilts]
    use_tilt = use_tilt or ["True"] * n_tilts

    def lines(values):
        return "\n".join(str(v) for v in values)

    nodes = "".join(
        f'<Node X="0" Y="0" Z="{z}" Value="{grid_x_value}" />' for z in range(n_tilts)
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<TiltSeries ImageDimensionsAngstrom="4000, 4000">\n'
        f"<Angles>{lines(angles)}</Angles>\n"
        f"<UseTilt>{lines(use_tilt)}</UseTilt>\n"
        f"<AxisAngle>{lines([85.0] * n_tilts)}</AxisAngle>\n"
        f"<AxisOffsetX>{lines(10.0 * (i + 1) for i in range(n_tilts))}</AxisOffsetX>\n"
        f"<AxisOffsetY>{lines(-5.0 * (i + 1) for i in range(n_tilts))}</AxisOffsetY>\n"
        '<CTF><Param Name="PixelSize" Value="2.5" /></CTF>\n'
        f'<GridMovementX Width="1" Height="1" Depth="{n_tilts}">'
        f"{nodes}</GridMovementX>\n"
        f'<GridMovementY Width="1" Height="1" Depth="{n_tilts}">'
        + "".join(f'<Node X="0" Y="0" Z="{z}" Value="0" />' for z in range(n_tilts))
        + "</GridMovementY>\n"
        "</TiltSeries>"
    )


def test_from_warp_xml_reads_global_alignment(tmp_path):
    pytest.importorskip("warpxmlfile")
    path = tmp_path / "TS_1.xml"
    path.write_text(_warp_xml())

    tilt_series = from_warp_xml(path, image_path=tmp_path / "TS_1.st")
    assert torch.allclose(tilt_series.tilt_angles, torch.tensor([-30.0, 0.0, 30.0]))
    assert torch.allclose(tilt_series.tilt_axis_angle, torch.tensor([85.0] * 3))
    # (y, x) in Angstroms, straight from AxisOffsetY / AxisOffsetX
    assert torch.allclose(
        tilt_series.sample_translations,
        torch.tensor([[-5.0, 10.0], [-10.0, 20.0], [-15.0, 30.0]]),
    )
    assert tilt_series.pixel_spacing == 2.5
    assert tilt_series.image_path == tmp_path / "TS_1.st"
    # a binned stack: the pixel size is overridden, the Angstrom alignment is not
    binned = from_warp_xml(path, pixel_spacing=10.0)
    assert binned.pixel_spacing == 10.0
    assert torch.equal(binned.sample_translations, tilt_series.sample_translations)
    assert torch.equal(tilt_series.image_indices, torch.tensor([0, 1, 2]))
    # Warp tomogram z is flipped relative to sample z
    z_flip = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))
    assert torch.equal(tilt_series.levelled2tomo, z_flip)
    assert tilt_series.local_shifts_2d is None
    assert TiltSeries.from_warp_xml(path).tilt_angles.shape == (3,)


def test_from_warp_xml_drops_unused_tilts(tmp_path):
    pytest.importorskip("warpxmlfile")
    path = tmp_path / "TS_1.xml"
    path.write_text(_warp_xml(use_tilt=["True", "False", "True"]))

    tilt_series = from_warp_xml(path)
    assert torch.allclose(tilt_series.tilt_angles, torch.tensor([-30.0, 30.0]))
    assert torch.equal(tilt_series.image_indices, torch.tensor([0, 2]))
    assert tilt_series.sample_translations.shape == (2, 2)


def test_from_warp_xml_warns_on_local_grid_unless_callable_given(tmp_path):
    pytest.importorskip("warpxmlfile")
    path = tmp_path / "TS_1.xml"
    path.write_text(_warp_xml(grid_x_value="1.5"))

    with pytest.warns(UserWarning, match="GridMovementX"):
        from_warp_xml(path)

    shifts = torch.zeros(1, 3, 2)
    tilt_series = from_warp_xml(path, local_shifts_2d=lambda projected: shifts)
    assert tilt_series.local_shifts_2d is not None
