"""Data processing pipeline for tomographic training data preparation.

This module handles the conversion of raw 3D tomographic volumes and boundary
masks into standardized 2D training samples suitable for deep learning models.
It includes data loading, preprocessing, normalization, and train/val splitting.
"""

import gc
import logging
from pathlib import Path
from typing import List, Tuple

import mrcfile
import numpy as np
import torch
from tqdm import tqdm

from torch_segment_tomogram_boundaries import config, constants
from torch_segment_tomogram_boundaries.utils.common import get_device
from torch_segment_tomogram_boundaries.utils.threeD import resize_and_pad_3d
from torch_segment_tomogram_boundaries.utils.twoD import robust_normalization

log = logging.getLogger(__name__)


class TrainingDataGenerator:
    """
    Generate standardized 2D training samples from 3D tomographic data.

    This class processes 3D tomogram volumes and their corresponding boundary masks
    to create 2D training samples suitable for deep learning. It handles data loading,
    preprocessing, normalization, and train/validation splitting.

    The processing pipeline includes:
    - Loading and resizing 3D volumes to standard dimensions
    - Extracting 2D slices along Y- or X-axes with random slab averaging
    - Applying robust normalization
    - Splitting data into training and validation sets
    - Saving preprocessed samples as PyTorch tensor files

    Attributes
    ----------
    volume_dir : Path
        Directory containing input tomogram volumes (.mrc files).
    mask_dir : Path
        Directory containing input boundary masks (.mrc files).
    output_train_dir : Path
        Directory for saving training samples (.pt files).
    output_val_dir : Path
        Directory for saving validation samples (.pt files).
    validation_fraction : float
        Fraction of data reserved for validation (0.0-1.0).
    target_shape : Tuple[int, int, int]
        Target 3D shape for volume resizing (depth, height, width).
    device : torch.device
        PyTorch device for tensor operations.
    """

    def __init__(
        self,
        volume_dir: Path = config.TOMOGRAM_DIR,
        mask_dir: Path = config.MASKS_DIR,
        output_train_dir: Path = config.TRAIN_DATA_DIR,
        output_val_dir: Path = config.VAL_DATA_DIR,
        validation_fraction: float = constants.VALIDATION_FRACTION,
        target_volume_shape: Tuple[int, int, int] = constants.TARGET_VOLUME_SHAPE,
    ):
        """
        Initialize data generator with directories and processing parameters.

        Parameters
        ----------
        volume_dir : Path, default=constants.TOMOGRAM_DIR
            Directory containing input tomogram volumes (.mrc files).
        mask_dir : Path, default=constants.MASKS_DIR
            Directory containing corresponding boundary masks (.mrc files).
        output_train_dir : Path, default=constants.TRAIN_DATA_DIR
            Output directory for training samples (.pt files).
        output_val_dir : Path, default=constants.VAL_DATA_DIR
            Output directory for validation samples (.pt files).
        validation_fraction : float, default=config.VALIDATION_FRACTION
            Fraction of data to reserve for validation (0.0-1.0).
        target_volume_shape : Tuple[int, int, int], default=constants.TARGET_VOLUME_SHAPE
            Target 3D shape for volume standardization (depth, height, width).
        """
        self.volume_dir = Path(volume_dir)
        self.mask_dir = Path(mask_dir)
        self.output_train_dir = Path(output_train_dir)
        self.output_val_dir = Path(output_val_dir)
        self.validation_fraction = validation_fraction
        self.target_shape = target_volume_shape
        self.device = get_device(verbose=True)

    def _find_data_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find matching volume-mask pairs based on filename stems.

        Returns
        -------
        List[Tuple[Path, Path]]
            List of (volume_path, mask_path) pairs for matched files.

        Warnings
        --------
        Logs warning for volumes without matching masks.
        """
        pairs = []
        label_files_map = {f.stem: f for f in self.mask_dir.glob("*.mrc")}
        for vol_path in sorted(list(self.volume_dir.glob("*.mrc"))):
            if vol_path.stem in label_files_map:
                pairs.append((vol_path, label_files_map[vol_path.stem]))
            else:
                log.warning(f"No matching label for {vol_path.name}")
        return pairs

    def run(self) -> None:
        """
        Execute complete data preparation pipeline.

        This method orchestrates the full processing workflow:
        1. Create output directories if they don't exist
        2. Find and match volume-mask pairs
        3. Split data into training and validation sets
        4. Process each 3D volume to extract 2D training samples
        5. Save preprocessed samples as PyTorch tensor files

        The number of 2D slices extracted per volume is controlled by
        constants.NUM_SECTIONS_PER_VOLUME.

        Raises
        ------
        FileNotFoundError
            If no matching tomogram/mask pairs are found in input directories.
        """
        self.output_train_dir.mkdir(parents=True, exist_ok=True)
        self.output_val_dir.mkdir(parents=True, exist_ok=True)

        all_data_pairs = self._find_data_pairs()
        if not all_data_pairs:
            raise FileNotFoundError(
                f"No tomogram/mask pairs found in {self.volume_dir} and {self.mask_dir}."
            )

        log.info(
            f"All volumes will be resized and padded to the universal 3D shape: {self.target_shape}"
        )

        np.random.seed(42)
        np.random.shuffle(all_data_pairs)
        val_split_idx = int(len(all_data_pairs) * self.validation_fraction)
        train_pairs, val_pairs = (
            all_data_pairs[val_split_idx:],
            all_data_pairs[:val_split_idx],
        )
        log.info(
            f"Splitting into {len(train_pairs)} train and {len(val_pairs)} val volumes."
        )

        for split_name, data_pairs, output_dir in [
            ("TRAIN", train_pairs, self.output_train_dir),
            ("VAL", val_pairs, self.output_val_dir),
        ]:
            log.info(f"--- Processing {split_name} set ---")
            for vol_file, label_file in tqdm(
                data_pairs, desc=f"Processing {split_name} Tomograms"
            ):
                volume_std, label_std = None, None
                try:
                    volume = torch.from_numpy(
                        mrcfile.open(vol_file, permissive=True).data.astype(np.float32)
                    )
                    label = torch.from_numpy(
                        mrcfile.open(label_file, permissive=True).data.astype(np.int8)
                    )

                    volume_std = resize_and_pad_3d(
                        volume, self.target_shape, mode="image"
                    ).to(self.device)
                    label_std = resize_and_pad_3d(
                        label, self.target_shape, mode="label"
                    ).to(self.device)

                    for i in range(constants.NUM_SECTIONS_PER_VOLUME):
                        margin = np.random.randint(2, 7)
                        axis_to_slice = torch.randint(1, 3, (1,)).item()
                        D, H, W = volume_std.shape
                        if axis_to_slice == 1:  # Slice along Y-axis
                            slice_idx = torch.randint(margin, H - margin, (1,)).item()
                            vol_slab = volume_std[
                                :, slice_idx - margin : slice_idx + margin + 1, :
                            ]
                            ortho_img = torch.mean(vol_slab, dim=1)
                            ortho_label = label_std[:, slice_idx, :]
                        else:  # Slice along X-axis
                            slice_idx = torch.randint(margin, W - margin, (1,)).item()
                            vol_slab = volume_std[
                                :, :, slice_idx - margin : slice_idx + margin + 1
                            ]
                            ortho_img = torch.mean(vol_slab, dim=2)
                            ortho_label = label_std[:, :, slice_idx]

                        ortho_img_norm = robust_normalization(ortho_img)
                        single_channel_input = ortho_img_norm.unsqueeze(0)

                        save_path = (
                            output_dir
                            / f"{output_dir.name}_{vol_file.stem}_view_{i}.pt"
                        )
                        torch.save(
                            {
                                "image": single_channel_input.cpu(),
                                "label": ortho_label.cpu().long(),
                            },
                            save_path,
                        )

                except Exception as e:
                    log.error(
                        f"Failed to process {vol_file.name}. Error: {e}", exc_info=True
                    )
                finally:
                    del volume_std, label_std, volume, label
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
