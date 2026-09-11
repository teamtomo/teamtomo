"""Tests for CTF results JSON data I/O."""

import json
import os
import tempfile

import torch

from torch_ctf_estimation.estimate_ctf import estimate_ctf
from torch_ctf_estimation.models import (
    CTFFittingParams,
    CTFResultsOutput,
    OpticalParams,
)
from torch_ctf_estimation.utils.data_io import (
    read_results_json,
    results_to_output_model,
    write_results_json,
)


def _default_optical():
    return OpticalParams(
        pixel_spacing_angstroms=1.0,
        voltage_kev=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast_fraction=0.1,
    )


def _default_fitting(**kwargs):
    p = {
        "defocus_grid_resolution": (1, 2, 2),
        "frequency_fit_range_angstroms": (30.0, 5.0),
        "defocus_range_microns": (0.5, 3.0),
        "patch_sidelength": 128,
    }
    p.update(kwargs)
    return CTFFittingParams(**p)


def test_results_to_output_model_linear_defocus():
    """results_to_output_model with Defocus2DResults that has linear defocus model."""
    image = torch.randn(512, 512)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image,
        _default_optical(),
        _default_fitting(defocus_grid_resolution=(1, 1, 1), defocus_model="linear"),
        device=torch.device("cpu"),
    )
    output = results_to_output_model(result2d)
    assert isinstance(output, CTFResultsOutput)
    assert output.defocus_results.defocus_model_type == "linear"
    assert output.defocus_results.linear_defocus is not None
    assert output.defocus_results.grid_defocus is None
    assert output.defocus_results.linear_defocus.defocus_0 is not None
    assert output.defocus_results.defocus_u is not None
    assert output.defocus_results.defocus_v is not None
    assert "envelope_B" in output.model_dump()
    # envelope_B may be float or null when not estimated
    assert output.envelope_B is None or isinstance(output.envelope_B, (int, float))


def test_results_to_output_model_grid_defocus():
    """results_to_output_model with Defocus2DResults that has grid defocus model."""
    image = torch.randn(512, 512)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image,
        _default_optical(),
        _default_fitting(defocus_model="grid"),
        device=torch.device("cpu"),
    )
    output = results_to_output_model(result2d)
    assert output.defocus_results.defocus_model_type == "grid"
    assert output.defocus_results.grid_defocus is not None
    assert output.defocus_results.linear_defocus is None
    assert len(output.defocus_results.grid_defocus.shape) == 3
    assert len(output.defocus_results.grid_defocus.values) > 0
    assert "envelope_B" in output.model_dump()


def test_results_to_output_model_phase_shift_quadratic():
    """results_to_output_model with phase shift quadratic model."""
    image = torch.randn(512, 512)
    _mean_ps, _, result2d = estimate_ctf(
        image,
        _default_optical(),
        _default_fitting(optimize_phase_shift=True, phase_shift_model="quadratic"),
        device=torch.device("cpu"),
    )
    output = results_to_output_model(result2d)
    assert output.phase_shift_params is not None
    assert output.phase_shift_params.phase_shift_model_type == "quadratic"
    assert output.phase_shift_params.quadratic is not None
    assert output.phase_shift_params.grid is None
    assert hasattr(output.phase_shift_params.quadratic, "C")
    assert hasattr(output.phase_shift_params.quadratic, "alpha_rad")
    assert hasattr(output.phase_shift_params.quadratic, "g1")
    assert hasattr(output.phase_shift_params.quadratic, "k1")
    assert hasattr(output.phase_shift_params.quadratic, "g2")
    assert hasattr(output.phase_shift_params.quadratic, "k2")


def test_results_to_output_model_phase_shift_grid():
    """results_to_output_model with phase shift grid model."""
    image = torch.randn(512, 512)
    _mean_ps, _, result2d = estimate_ctf(
        image,
        _default_optical(),
        _default_fitting(optimize_phase_shift=True, phase_shift_model="grid"),
        device=torch.device("cpu"),
    )
    output = results_to_output_model(result2d)
    assert output.phase_shift_params is not None
    assert output.phase_shift_params.phase_shift_model_type == "grid"
    assert output.phase_shift_params.grid is not None
    assert output.phase_shift_params.quadratic is None
    assert "grid_u" in output.phase_shift_params.grid.model_dump()
    assert "grid_v" in output.phase_shift_params.grid.model_dump()


def test_results_to_output_model_envelope_B_present_when_set():
    """envelope_B is present in output when set on result (and null when not)."""
    image = torch.randn(512, 512)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image,
        _default_optical(),
        _default_fitting(),
        device=torch.device("cpu"),
    )
    output = results_to_output_model(result2d)
    dump = output.model_dump()
    assert "envelope_B" in dump
    # Value is either a number or None
    assert dump["envelope_B"] is None or isinstance(dump["envelope_B"], (int, float))


def test_write_results_json_and_read_results_json():
    """write_results_json to temp file and read_results_json round-trip."""
    image = torch.randn(512, 512)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image,
        _default_optical(),
        _default_fitting(),
        device=torch.device("cpu"),
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        write_results_json(result2d, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "defocus_results" in data
        assert "phase_shift_params" in data
        assert "envelope_B" in data
        assert "cross_correlation_final" in data
        assert data["cross_correlation_final"] is None or isinstance(
            data["cross_correlation_final"],
            (int, float),
        )
        assert data["defocus_results"]["defocus_model_type"] in ("grid", "linear")
        assert "defocus_u" in data["defocus_results"]
        assert "defocus_v" in data["defocus_results"]
        loaded = read_results_json(path)
        assert isinstance(loaded, CTFResultsOutput)
        assert loaded.defocus_results.defocus_u == data["defocus_results"]["defocus_u"]
        assert loaded.defocus_results.defocus_v == data["defocus_results"]["defocus_v"]
    finally:
        os.unlink(path)


def test_write_results_json_from_estimate_ctf():
    """Caller writes JSON after estimate_ctf; estimate_ctf itself does no I/O."""
    image = torch.randn(512, 512)
    _mean_ps, _result1d, result2d = estimate_ctf(
        image,
        _default_optical(),
        _default_fitting(),
        device=torch.device("cpu"),
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        write_results_json(result2d, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "defocus_results" in data
        assert "envelope_B" in data
    finally:
        os.unlink(path)
