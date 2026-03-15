"""Tomographic slab prediction and inference pipeline.

This module provides the TomoSlabPredictor class for running inference on trained
models to generate boundary masks from tomographic volumes.
"""

import gc
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import mrcfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch_segment_tomogram_boundaries import config
from torch_segment_tomogram_boundaries.losses import get_loss_function
from torch_segment_tomogram_boundaries.models import create_unet
from torch_segment_tomogram_boundaries.pl_model import SegmentationModel
from torch_segment_tomogram_boundaries.utils import threeD
from torch_segment_tomogram_boundaries.utils.common import get_device
from torch_segment_tomogram_boundaries.utils.twoD import robust_normalization

# Configure logging for prediction pipeline
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TomoSlabPredictor:
    """
    Predict tomographic slab boundaries using a trained deep learning model.

    This class provides an inference pipeline for generating boundary masks from 3D
    tomographic volumes. It loads a trained PyTorch Lightning model and applies it
    to new data.

    The prediction workflow includes:
    1. Model loading and setup from a checkpoint.
    2. Volume preprocessing and normalization.
    3. Multi-axis inference with slab blending for temporal consistency.
    4. Probability map averaging and optional smoothing.
    """

    def __init__(
        self, model_checkpoint_path: Union[str, Path], compile_model: bool = True
    ):
        """
        Initialize the predictor with a trained model checkpoint.

        Args:
            model_checkpoint_path: Path to the trained .ckpt file.
            compile_model: If True, compiles the model with `torch.compile` for
                           faster inference (requires PyTorch 2.0+).
        """
        self.device = get_device()
        self.model, self.target_shape_3d = self._load_model(
            model_checkpoint_path, compile_model
        )

    def _load_model(
        self, model_checkpoint_path: Union[str, Path], compile_model: bool
    ) -> Tuple[nn.Module, Tuple[int, int, int]]:
        """Loads the segmentation model from a checkpoint."""
        logging.info(f"Loading model from checkpoint: {model_checkpoint_path}")
        base_model = create_unet(**config.MODEL_CONFIG)
        loss_fn = get_loss_function(config.LOSS_CONFIG)

        model = SegmentationModel.load_from_checkpoint(
            model_checkpoint_path,
            map_location=self.device,
            model=base_model,
            loss_function=loss_fn,
        )
        model.eval().to(self.device)

        if compile_model and hasattr(torch, "compile"):
            logging.info("Compiling model for faster inference...")
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                logging.warning(
                    f"Model compilation failed: {e}. Using uncompiled model."
                )

        target_shape = model.hparams.get("target_shape")
        if not target_shape:
            raise ValueError(
                "`target_shape` not found in model checkpoint. Please retrain with an updated trainer."
            )

        logging.info(f"Using target shape from checkpoint for resizing: {target_shape}")
        return model, target_shape

    @torch.no_grad()
    def predict_probabilities(
        self,
        input_tomogram: Union[str, Path, np.ndarray],
        slab_size: int = 15,
        batch_size: int = 16,
        smoothing_sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Execute the prediction pipeline to generate a 3D probability map.

        Args:
            input_tomogram: Input tomogram as MRC file path or pre-loaded 3D numpy array.
            slab_size: Size of slab for temporal blending (must be odd). If 1, no blending.
            batch_size: Batch size for processing 2D slices during inference.
            smoothing_sigma: Standard deviation for 3D Gaussian smoothing. If None, no smoothing.

        Returns
        -------
            The predicted 3D probability map as a numpy array.
        """
        if isinstance(input_tomogram, (str, Path)):
            with mrcfile.open(input_tomogram, permissive=True) as mrc:
                original_data_np = mrc.data.astype(np.float32)
        else:
            original_data_np = input_tomogram.astype(np.float32)

        original_shape = original_data_np.shape
        logging.info(f"Input tomogram shape: {original_shape}")

        resized_volume = threeD.resize_and_pad_3d(
            torch.from_numpy(original_data_np),
            target_shape=self.target_shape_3d,
            mode="image",
        ).to(self.device)

        pred_xz = self._predict_single_axis_with_slab_blending(
            resized_volume, "XZ", slab_size, batch_size
        )
        pred_yz_permuted = self._predict_single_axis_with_slab_blending(
            resized_volume.permute(0, 2, 1), "YZ", slab_size, batch_size
        )
        pred_yz = pred_yz_permuted.permute(0, 2, 1)

        logging.info("Averaging predictions from both axes.")
        prob_map_tensor = (pred_xz + pred_yz) / 2.0

        if smoothing_sigma and smoothing_sigma > 0:
            logging.info(
                f"Applying 3D Gaussian smoothing with sigma={smoothing_sigma}..."
            )
            prob_map_tensor = threeD.gpu_gaussian_blur_3d(
                prob_map_tensor, smoothing_sigma, self.device
            )

        logging.info(f"Resizing prediction back to original shape {original_shape}...")
        prob_map_np = (
            F.interpolate(
                prob_map_tensor.unsqueeze(0).unsqueeze(0),
                size=original_shape,
                mode="area",
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info("Probability map prediction complete.")
        return prob_map_np

    def predict_binary(
        self,
        input_tomogram: Union[str, Path, np.ndarray],
        binarize_threshold: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        """
        Execute the prediction pipeline and return a binarized mask.

        Args:
            input_tomogram: Input tomogram as MRC file path or pre-loaded numpy array.
            binarize_threshold: Threshold for converting probability map to a binary mask.
            **kwargs: Additional keyword arguments passed to `predict_probabilities`,
                      e.g., `slab_size`, `batch_size`, `smoothing_sigma`.

        Returns
        -------
            The final binary slab mask as a 3D numpy array.
        """
        prob_map = self.predict_probabilities(input_tomogram=input_tomogram, **kwargs)
        logging.info(f"Binarizing probability map with threshold={binarize_threshold}")
        return (prob_map > binarize_threshold).astype(np.uint8)

    @torch.no_grad()
    def _predict_raw_slab(self, slab_3d: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Predict raw probabilities for a 3D slab using batched processing."""
        D, H, W = slab_3d.shape
        predictions = []
        for i in range(0, H, batch_size):
            end_idx = min(i + batch_size, H)
            batch_slices = [
                robust_normalization(slab_3d[:, j, :]).unsqueeze(0)
                for j in range(i, end_idx)
            ]
            batch_tensor = torch.stack(batch_slices, dim=0)
            pred_batch = self.model(batch_tensor)
            pred_batch = torch.sigmoid(pred_batch).squeeze(1)
            predictions.extend([pred_batch[k] for k in range(pred_batch.shape[0])])
        return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def _predict_single_axis_with_slab_blending(
        self, volume_3d: torch.Tensor, axis: str, slab_size: int, batch_size: int
    ) -> torch.Tensor:
        """Optimized slab blending prediction for better memory efficiency."""
        if slab_size <= 1:
            logging.info(f"Predicting along {axis} axis (no slab blending)...")
            return self._predict_raw_slab(volume_3d, batch_size)

        logging.info(
            f"Predicting with slab blending along {axis} axis (slab_size={slab_size})..."
        )
        num_slices = volume_3d.shape[1]
        final_slices = []
        hann_window = torch.hann_window(slab_size, periodic=False, device=self.device)
        half_slab = slab_size // 2

        for i in tqdm(
            range(num_slices),
            desc=f"Slab Blending ({axis} axis)",
            leave=False,
            ncols=80,
        ):
            start, end = max(0, i - half_slab), min(num_slices, i + half_slab + 1)
            pad_left = max(0, (i - half_slab) * -1)
            pad_right = max(0, (i + half_slab + 1) - num_slices)

            slab_3d = volume_3d[:, start:end, :]
            if pad_left > 0:
                slab_3d = torch.cat(
                    [volume_3d[:, :1, :].repeat(1, pad_left, 1), slab_3d], dim=1
                )
            if pad_right > 0:
                slab_3d = torch.cat(
                    [slab_3d, volume_3d[:, -1:, :].repeat(1, pad_right, 1)], dim=1
                )

            predicted_slab = self._predict_raw_slab(slab_3d, batch_size)

            current_window = hann_window.clone()
            if pad_left > 0:
                current_window[:pad_left] = 0
            if pad_right > 0:
                current_window[-pad_right:] = 0

            if current_window.sum() > 0:
                weights = current_window.view(1, -1, 1)
                final_slice = (predicted_slab * weights).sum(dim=1) / weights.sum()
            else:
                slice_input = (
                    robust_normalization(volume_3d[:, i, :]).unsqueeze(0).unsqueeze(0)
                )
                final_slice = (
                    torch.sigmoid(self.model(slice_input)).squeeze(0).squeeze(0)
                )

            final_slices.append(final_slice)

        return torch.stack(final_slices, dim=1)


def predict_probabilities(
    input_tomogram: Union[str, Path, np.ndarray],
    model_checkpoint_path: Union[str, Path],
    **kwargs,
) -> np.ndarray:
    """
    High-level function to predict a 3D probability map from a tomogram.

    This function provides a simple interface for inference, handling the
    instantiation of the predictor, model loading, and execution of the
    prediction pipeline.

    Args:
        input_tomogram: Input tomogram as MRC file path or pre-loaded numpy array.
        model_checkpoint_path: Path to the trained .ckpt file.
        **kwargs: Additional keyword arguments passed to the predictor's
                  `predict_probabilities` method, e.g., `slab_size`, `batch_size`,
                  `smoothing_sigma`, `compile_model`.

    Returns
    -------
        The predicted 3D probability map as a numpy array.
    """
    compile_model = kwargs.pop("compile_model", True)
    predictor = TomoSlabPredictor(
        model_checkpoint_path=model_checkpoint_path, compile_model=compile_model
    )
    return predictor.predict_probabilities(input_tomogram=input_tomogram, **kwargs)


def predict_binary(
    input_tomogram: Union[str, Path, np.ndarray],
    model_checkpoint_path: Union[str, Path],
    **kwargs,
) -> np.ndarray:
    """
    High-level function to predict a binary slab mask from a tomogram.

    This function provides a simple interface for inference, handling the
    instantiation of the predictor, model loading, and execution of the
    prediction pipeline.

    Args:
        input_tomogram: Input tomogram as MRC file path or pre-loaded numpy array.
        model_checkpoint_path: Path to the trained .ckpt file.
        **kwargs: Additional keyword arguments passed to the predictor's
                  `predict_binary` method, e.g., `binarize_threshold`,
                  `slab_size`, `batch_size`, `compile_model`.

    Returns
    -------
        The final binary slab mask as a 3D numpy array.
    """
    compile_model = kwargs.pop("compile_model", True)
    predictor = TomoSlabPredictor(
        model_checkpoint_path=model_checkpoint_path, compile_model=compile_model
    )
    return predictor.predict_binary(input_tomogram=input_tomogram, **kwargs)
