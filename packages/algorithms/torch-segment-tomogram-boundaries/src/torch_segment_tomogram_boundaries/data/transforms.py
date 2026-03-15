"""Augmentation pipeline derived from Easymode cryoET utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.ndimage import rotate as nd_rotate

from torch_segment_tomogram_boundaries import constants
from torch_segment_tomogram_boundaries.data.easymode_fourier import (
    MissingWedgeMaskAndFourierAmplitudeMatching2D,
)


@dataclass
class DimensionConfig:
    buffer_height: int
    buffer_width: int
    target_height: int
    target_width: int


TRAIN_ROT_LIMIT_DEGREES = 10.0


def scale_to_0_1(img: np.ndarray) -> np.ndarray:
    """Scale numpy array to the [0, 1] range with safe guarding for zero variance."""
    img = img.astype(np.float32)
    min_val = float(img.min())
    max_val = float(img.max())
    if max_val - min_val > 1e-6:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = img - min_val
    return img.astype(np.float32)


class EasyModeAugmentor:
    """Easymode-inspired augmentation pipeline for 2D orthogonal slabs."""

    def __init__(self, is_training: bool = True) -> None:
        cfg = constants.AUGMENTATION_CONFIG
        self.dim_cfg = DimensionConfig(
            buffer_height=cfg["RESIZE_BUFFER_HEIGHT"],
            buffer_width=cfg["RESIZE_BUFFER_WIDTH"],
            target_height=cfg["TARGET_HEIGHT"],
            target_width=cfg["TARGET_WIDTH"],
        )
        self.is_training = is_training
        self.missing_wedge_aug = MissingWedgeMaskAndFourierAmplitudeMatching2D(
            amplitude_aug=True,
            missing_wedge_aug=True,
            missing_wedge_prob=0.4,
            amplitude_prob=0.5,
            sample_kernel_prob=0.7,
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        img_2d = np.squeeze(image).astype(np.float32)
        mask_2d = mask.astype(np.float32)

        img_2d = scale_to_0_1(img_2d)
        img_2d, mask_2d = self._resize_and_center_crop(img_2d, mask_2d)

        if self.is_training:
            img_2d, mask_2d = self._apply_training_pipeline(img_2d, mask_2d)

        img_2d = scale_to_0_1(img_2d)
        mask_2d = np.clip(mask_2d, 0.0, 1.0)

        return {
            "image": img_2d[..., None].astype(np.float32),
            "mask": mask_2d.astype(np.float32),
        }

    def _apply_training_pipeline(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            img, mask = self._horizontal_flip(img, mask)

        if np.random.rand() < 0.35:
            img, mask = self._continuous_rotate(img, mask)

        if np.random.rand() < 0.3:
            img, mask = self._random_zoom(img, mask)

        if np.random.rand() < 0.4:
            img = self.missing_wedge_aug(img)

        if np.random.rand() < 0.25:
            sigma = float(np.random.uniform(0.3, 1.5))
            img = gaussian_filter(img, sigma=sigma)

        return self._ensure_target_shape(img, mask)

    def _resize_and_center_crop(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.dim_cfg
        resized_img = cv2.resize(
            img, (cfg.buffer_width, cfg.buffer_height), interpolation=cv2.INTER_LINEAR
        )
        resized_mask = cv2.resize(
            mask, (cfg.buffer_width, cfg.buffer_height), interpolation=cv2.INTER_NEAREST
        )

        start_y = (cfg.buffer_height - cfg.target_height) // 2
        start_x = (cfg.buffer_width - cfg.target_width) // 2
        end_y = start_y + cfg.target_height
        end_x = start_x + cfg.target_width

        cropped_img = resized_img[start_y:end_y, start_x:end_x]
        cropped_mask = resized_mask[start_y:end_y, start_x:end_x]
        return cropped_img.astype(np.float32), cropped_mask.astype(np.float32)

    @staticmethod
    def _horizontal_flip(
        img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Preserve beam orientation by only mirroring along the width axis (axis=1).
        img = np.flip(img, axis=1)
        mask = np.flip(mask, axis=1)
        return img, mask

    @staticmethod
    def _continuous_rotate(
        img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        angle = float(
            np.random.uniform(-TRAIN_ROT_LIMIT_DEGREES, TRAIN_ROT_LIMIT_DEGREES)
        )
        img_rot = nd_rotate(
            img,
            angle=angle,
            axes=(0, 1),
            order=1,
            mode="reflect",
            reshape=False,
            prefilter=False,
        )
        mask_rot = nd_rotate(
            mask,
            angle=angle,
            axes=(0, 1),
            order=0,
            mode="nearest",
            reshape=False,
            prefilter=False,
        )
        return img_rot.astype(np.float32), mask_rot.astype(np.float32)

    def _random_zoom(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        factor = float(np.random.uniform(0.9, 1.1))
        zoom_img = zoom(img, factor, order=1)
        zoom_mask = zoom(mask, factor, order=0)
        zoom_img, zoom_mask = self._match_shape(zoom_img, zoom_mask, img.shape)
        return zoom_img.astype(np.float32), zoom_mask.astype(np.float32)

    @staticmethod
    def _match_shape(
        img: np.ndarray, mask: np.ndarray, target_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        img_adj = EasyModeAugmentor._center_pad_or_crop(
            img, target_shape, mode="reflect"
        )
        mask_adj = EasyModeAugmentor._center_pad_or_crop(
            mask, target_shape, mode="edge"
        )
        return img_adj, mask_adj

    @staticmethod
    def _center_pad_or_crop(
        arr: np.ndarray, target_shape: Tuple[int, int], mode: str
    ) -> np.ndarray:
        result = arr
        current_h, current_w = result.shape
        target_h, target_w = target_shape

        # Adjust height
        if current_h < target_h:
            pad_total = target_h - current_h
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            result = np.pad(result, ((pad_top, pad_bottom), (0, 0)), mode=mode)
        elif current_h > target_h:
            start = (current_h - target_h) // 2
            end = start + target_h
            result = result[start:end, :]

        # Adjust width
        current_h, current_w = result.shape
        if current_w < target_w:
            pad_total = target_w - current_w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            result = np.pad(result, ((0, 0), (pad_left, pad_right)), mode=mode)
        elif current_w > target_w:
            start = (current_w - target_w) // 2
            end = start + target_w
            result = result[:, start:end]

        return result.astype(np.float32)

    def _ensure_target_shape(
        self, img: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_shape = (self.dim_cfg.target_height, self.dim_cfg.target_width)
        if img.shape != target_shape:
            img, mask = self._match_shape(img, mask, target_shape)
        return img, mask


def get_transforms(
    is_training: bool = True, use_balanced_crop: bool = True
) -> EasyModeAugmentor:
    """Factory aligning with the previous public API."""
    return EasyModeAugmentor(is_training=is_training)
