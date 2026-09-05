"""Tests for plateau-style early stopping."""

from torch_ctf_estimation.models import CTFFittingParams, ThicknessParams
from torch_ctf_estimation.utils.early_stopping import make_early_stopper


def test_make_early_stopper_flat_losses_stop_after_window_plus_patience():
    stopper = make_early_stopper(patience=5, window_size=3, tolerance=1e-5)
    results = [stopper(1.0) for _ in range(8)]
    assert results[:7] == [False] * 7
    assert results[7] is True


def test_make_early_stopper_changing_losses_do_not_stop():
    stopper = make_early_stopper(patience=5, window_size=3, tolerance=1e-5)
    loss = 1.0
    for _ in range(20):
        assert stopper(loss) is False
        loss *= 0.5


def test_fitting_params_build_early_stopper_default_off():
    params = CTFFittingParams(
        defocus_grid_resolution=(1, 1, 1),
        frequency_fit_range_angstroms=(30.0, 5.0),
    )
    assert params.early_stopping is False
    assert params.build_early_stopper() is None


def test_fitting_params_build_early_stopper_when_enabled():
    params = CTFFittingParams(
        defocus_grid_resolution=(1, 1, 1),
        frequency_fit_range_angstroms=(30.0, 5.0),
        early_stopping=True,
        early_stopping_patience=2,
        early_stopping_window_size=2,
        early_stopping_tolerance=1e-4,
    )
    stopper = params.build_early_stopper()
    assert stopper is not None
    assert stopper(1.0) is False


def test_thickness_params_build_early_stopper_when_enabled():
    params = ThicknessParams(early_stopping=True)
    assert params.build_early_stopper() is not None
    assert ThicknessParams().build_early_stopper() is None
