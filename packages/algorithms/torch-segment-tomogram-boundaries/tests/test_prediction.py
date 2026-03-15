# tests/test_prediction.py
import warnings

import mrcfile
import numpy as np
import pytest
import torch

from torch_segment_tomogram_boundaries.predict import TomoSlabPredictor


def test_predictor_initialization(trained_checkpoint):
    """Test loading the model from a real checkpoint file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert trained_checkpoint.exists()
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint))
        assert predictor.model is not None
        assert isinstance(predictor.model, torch.nn.Module)
        assert predictor.target_shape_3d is not None


@pytest.mark.slow
def test_predict_from_numpy_array(trained_checkpoint):
    """Test prediction when the input is a NumPy array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint))
        # Input shape must match what the model was trained on to some extent
        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

        # Use a small slab size for speed
        result_mask = predictor.predict_binary(input_tomo, slab_size=3)

        assert isinstance(result_mask, np.ndarray)
        assert result_mask.shape == input_tomo.shape
        assert result_mask.dtype == np.uint8
        # The mask should be binary
        assert np.all(np.isin(result_mask, [0, 1]))


@pytest.mark.slow
def test_predict_from_file(trained_checkpoint, dummy_mrc_files, tmp_path):
    """Test prediction from an input file, writing to an output file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint))
        output_path = tmp_path / "output_mask.mrc"
        input_path = dummy_mrc_files["vol_path"]

        with mrcfile.open(input_path, permissive=True) as mrc:
            input_shape = mrc.data.shape

        mask = predictor.predict_binary(input_tomogram=input_path, slab_size=3)
        with mrcfile.new(output_path, overwrite=True) as mrc:
            mrc.set_data(mask)

        assert output_path.exists()
        with mrcfile.open(output_path) as mrc:
            assert mrc.data.shape == input_shape
            assert mrc.data.dtype == np.uint8
            assert np.all(np.isin(mrc.data, [0, 1]))


@pytest.mark.slow
def test_predict_probabilities_output(trained_checkpoint):
    """Test the output of the predict_probabilities method."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint))
        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

        prob_map = predictor.predict_probabilities(input_tomo, slab_size=3)

        assert isinstance(prob_map, np.ndarray)
        assert prob_map.shape == input_tomo.shape
        assert prob_map.dtype == np.float32
        assert prob_map.min() >= 0.0
        assert prob_map.max() <= 1.0


@pytest.mark.slow
def test_predict_with_no_slab_blending(trained_checkpoint):
    """Test prediction with slab_size=1 (no blending)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint))
        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

        # slab_size=1 disables blending
        result_mask = predictor.predict_binary(input_tomo, slab_size=1)

        assert result_mask.shape == input_tomo.shape
        assert np.all(np.isin(result_mask, [0, 1]))


@pytest.mark.slow
def test_predict_with_smoothing(trained_checkpoint):
    """Test prediction with Gaussian smoothing enabled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        predictor = TomoSlabPredictor(model_checkpoint_path=str(trained_checkpoint))
        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

        # Use a non-zero sigma to enable smoothing
        prob_map = predictor.predict_probabilities(
            input_tomo, slab_size=3, smoothing_sigma=1.5
        )

        assert prob_map.shape == input_tomo.shape
        assert prob_map.dtype == np.float32
        # A simple check: a smoothed output should not be identical to a non-smoothed one
        prob_map_no_smooth = predictor.predict_probabilities(
            input_tomo, slab_size=3, smoothing_sigma=None
        )
        assert not np.allclose(prob_map, prob_map_no_smooth, atol=1e-5)


@pytest.mark.slow
def test_predict_with_compile_disabled(trained_checkpoint):
    """Test that prediction works with torch.compile disabled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Initialize predictor with compile_model=False
        predictor = TomoSlabPredictor(
            model_checkpoint_path=str(trained_checkpoint), compile_model=False
        )
        input_tomo = np.random.rand(32, 64, 64).astype(np.float32)

        result_mask = predictor.predict_binary(input_tomo, slab_size=3)

        assert result_mask.shape == input_tomo.shape
        assert np.all(np.isin(result_mask, [0, 1]))
