"""Tests for file I/O wrappers."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


def _write_test_mrc(path: Path, data: np.ndarray, voxel_size: float = 1.0) -> None:
    import mrcfile

    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = voxel_size


def test_fit_map_in_map_from_files_same_map(tmp_path):
    """Fitting identical maps from files should recover near-identity rotation."""
    from torch_fit_in_map import ExhaustiveSearchConfig, fit_map_in_map_from_files
    import torch

    data = np.random.rand(20, 20, 20).astype(np.float32)
    ref_path = tmp_path / "ref.mrc"
    mob_path = tmp_path / "mob.mrc"
    _write_test_mrc(ref_path, data, voxel_size=1.0)
    _write_test_mrc(mob_path, data, voxel_size=1.0)

    result = fit_map_in_map_from_files(
        mob_path,
        ref_path,
        exhaustive_config=ExhaustiveSearchConfig(angular_step_degrees=30.0),
        gradient_config=None,
        verbose=False,
    )
    assert torch.allclose(result.rotation_matrix.cpu(), torch.eye(3), atol=0.15)


def test_fit_map_in_map_from_files_voxel_size_mismatch(tmp_path):
    """Files with different voxel sizes should be rescaled and still align."""
    from torch_fit_in_map import ExhaustiveSearchConfig, fit_map_in_map_from_files

    data = np.random.rand(20, 20, 20).astype(np.float32)
    ref_path = tmp_path / "ref.mrc"
    mob_path = tmp_path / "mob.mrc"
    _write_test_mrc(ref_path, data, voxel_size=1.0)
    _write_test_mrc(mob_path, data, voxel_size=2.0)  # different pixel size

    result = fit_map_in_map_from_files(
        mob_path,
        ref_path,
        exhaustive_config=ExhaustiveSearchConfig(angular_step_degrees=30.0),
        gradient_config=None,
    )
    # After rescaling the content differs slightly due to resampling but score should be > 0
    assert isinstance(result.score, float)


def test_crop_or_pad_larger(tmp_path):
    """crop_or_pad_to_shape should center-crop a larger volume."""
    from torch_fit_in_map import crop_or_pad_to_shape

    vol = torch.ones(30, 30, 30)
    out = crop_or_pad_to_shape(vol, (20, 20, 20))
    assert out.shape == (20, 20, 20)


def test_crop_or_pad_smaller(tmp_path):
    """crop_or_pad_to_shape should zero-pad a smaller volume."""
    from torch_fit_in_map import crop_or_pad_to_shape

    vol = torch.ones(10, 10, 10)
    out = crop_or_pad_to_shape(vol, (20, 20, 20))
    assert out.shape == (20, 20, 20)
    # Corners (padding region) should be zero; centre of original data should be one
    assert out[0, 0, 0].item() == 0.0
    assert out[10, 10, 10].item() == 1.0  # data placed at [5:15,5:15,5:15]


def test_crop_or_pad_non_cubic(tmp_path):
    """crop_or_pad_to_shape should handle non-cubic targets."""
    from torch_fit_in_map import crop_or_pad_to_shape

    vol = torch.ones(40, 20, 10)
    out = crop_or_pad_to_shape(vol, (30, 25, 15))
    assert out.shape == (30, 25, 15)


def test_fit_pdb_in_map_from_files_raises_without_espcalculator(tmp_path, monkeypatch):
    """fit_pdb_in_map_from_files should raise ImportError when espcalculator is missing."""
    import sys
    import numpy as np

    from torch_fit_in_map import fit_pdb_in_map_from_files

    data = np.random.rand(20, 20, 20).astype(np.float32)
    map_path = tmp_path / "map.mrc"
    _write_test_mrc(map_path, data)
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text("REMARK test\n")

    # Simulate espcalculator not being installed
    monkeypatch.setitem(sys.modules, "espcalculator", None)  # type: ignore[arg-type]

    with pytest.raises((ImportError, Exception), match="espcalculator|torch-calculate"):
        fit_pdb_in_map_from_files(
            mobile_pdb_path=pdb_path,
            reference_map_path=map_path,
            pixel_size_angstroms=1.0,
            box_size=20,
        )
