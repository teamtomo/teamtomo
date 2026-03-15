"""Utility functions and Fourier-domain augmentations inspired by Easymode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import numpy.fft as fft
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


def hypotenuse_ndim(axes: List[np.ndarray], offset: float = 0.5) -> np.ndarray:
    """Compute the n-dimensional hypotenuse with an optional offset."""
    if len(axes) == 2:
        return np.hypot(
            axes[0] - max(axes[0].shape) * offset,
            axes[1] - max(axes[1].shape) * offset,
        )
    return np.hypot(
        hypotenuse_ndim(axes[1:], offset),
        axes[0] - max(axes[0].shape) * offset,
    )


def rotational_kernel(arr: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Create a rotationally symmetric kernel from a 1D reference array."""
    func = interp1d(
        np.arange(len(arr), dtype=np.float32),
        arr.astype(np.float32),
        bounds_error=False,
        fill_value=0.0,
    )
    axes = np.ogrid[tuple(slice(0, int(np.ceil(s / 2))) for s in shape)]
    kernel = hypotenuse_ndim(axes, offset=0.0).astype(np.float32)
    kernel = func(kernel)

    for idx, size in enumerate(shape):
        padding: List[Tuple[int, int]] = [(0, 0)] * len(shape)
        padding[idx] = (int(np.floor(size / 2)), 0)
        mode = "reflect" if size % 2 else "symmetric"
        kernel = np.pad(kernel, padding, mode=mode)
    return kernel.astype(np.float32)


def get_line_plot(
    n_points: int,
    smooth_sigma: float,
    step_sigma: float,
    offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a smoothed random walk used for spectrum modulation."""
    x = np.linspace(0, n_points - 1, n_points, dtype=np.float32)
    y = np.cumsum(np.random.randn(n_points).astype(np.float32) * step_sigma)
    y_smoothed = gaussian_filter1d(y, sigma=smooth_sigma)
    return x, np.abs(y_smoothed + offset)


def normalize_and_fft_patch(patch: np.ndarray) -> np.ndarray:
    """Normalize a patch and project it into Fourier space."""
    patch = patch.astype(np.float32)
    patch = patch - patch.min()
    max_val = patch.max()
    if max_val > 0:
        patch = patch / max_val
    transformed = fft.fftn(patch)
    return fft.fftshift(transformed)


def fft_patch_to_real(fft_patch: np.ndarray) -> np.ndarray:
    """Return the real component after an inverse FFT."""
    shifted = fft.fftshift(fft_patch)
    inverse = fft.ifftn(shifted)
    return np.real(inverse).astype(np.float32)


def sample_scalar(
    value: Union[Tuple[float, float], List[float], Callable[..., Any], float],
    *args: Any,
) -> float:
    """Sample a scalar from a range or callable."""
    if isinstance(value, (tuple, list)):
        low, high = value
        return float(np.random.uniform(low, high))
    if callable(value):
        return float(value(*args))
    return float(value)


def wedge_mask_2d(
    shape: Tuple[int, int], angle: float, orientation_deg: float
) -> np.ndarray:
    """Boolean mask retaining frequencies within a wedge around an orientation."""
    height, width = shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    y_coords = y_coords - (height - 1) / 2.0
    x_coords = x_coords - (width - 1) / 2.0
    angles = (np.degrees(np.arctan2(y_coords, x_coords)) + 360.0) % 360.0

    lower = (orientation_deg - angle / 2.0) % 360.0
    upper = (orientation_deg + angle / 2.0) % 360.0
    if lower < upper:
        keep = (angles >= lower) & (angles <= upper)
    else:
        keep = (angles >= lower) | (angles <= upper)
    return keep


@dataclass
class MissingWedgeParams:
    smooth_sigma: float
    step_sigma: float
    offset: float
    missing_angle: float
    do_mw_transform: bool
    do_amp_transform: bool
    do_sample_kernel: bool
    orientation: float


def _default_scale_func(shape: Tuple[int, int], axis: int) -> float:
    """Default scaling function for Fourier augmentation."""
    return float(
        np.exp(np.random.uniform(np.log(shape[axis] / 8.0), np.log(shape[axis])))
    )


class MissingWedgeMaskAndFourierAmplitudeMatching2D:
    """Combined Fourier augmentation inspired by Easymode's Membrain pipeline."""

    def __init__(
        self,
        amplitude_aug: bool = True,
        missing_wedge_aug: bool = True,
        smooth_sigma_range: Union[Tuple[float, float], float] = (2.0, 4.0),
        step_sigma_range: Union[Tuple[float, float], float] = (0.1, 4.0),
        offset_range: Union[Tuple[float, float], float] = (2.0, 10.0),
        missing_angle_range: Union[Tuple[float, float], float] = (50.0, 85.0),
        missing_wedge_prob: float = 0.5,
        amplitude_prob: float = 0.5,
        sample_kernel_prob: float = 0.5,
        scale: Callable[[Tuple[int, int], int], float] | None = None,
        loc: Union[Tuple[float, float], float] = (0.0, 1.0),
    ) -> None:
        self.amplitude_aug = amplitude_aug
        self.missing_wedge_aug = missing_wedge_aug
        self.smooth_sigma_range = smooth_sigma_range
        self.step_sigma_range = step_sigma_range
        self.offset_range = offset_range
        self.missing_angle_range = missing_angle_range
        self.missing_wedge_prob = missing_wedge_prob
        self.amplitude_prob = amplitude_prob
        self.sample_kernel_prob = sample_kernel_prob
        self.scale = scale or _default_scale_func
        self.loc = loc

    def _draw_params(self) -> MissingWedgeParams:
        smooth_sigma = (
            self.smooth_sigma_range
            if isinstance(self.smooth_sigma_range, float)
            else np.random.uniform(*self.smooth_sigma_range)
        )
        step_sigma = (
            self.step_sigma_range
            if isinstance(self.step_sigma_range, float)
            else np.random.uniform(*self.step_sigma_range)
        )
        offset = (
            self.offset_range
            if isinstance(self.offset_range, float)
            else np.random.uniform(*self.offset_range)
        )
        missing_angle = (
            self.missing_angle_range
            if isinstance(self.missing_angle_range, float)
            else np.random.uniform(*self.missing_angle_range)
        )

        do_mw_transform = bool(np.random.rand() < self.missing_wedge_prob)
        do_amp_transform = bool(np.random.rand() < self.amplitude_prob)
        do_sample_kernel = bool(np.random.rand() < self.sample_kernel_prob)
        if do_mw_transform:
            do_sample_kernel = True
        orientation = float(np.random.uniform(0.0, 180.0))

        return MissingWedgeParams(
            smooth_sigma=float(smooth_sigma),
            step_sigma=float(step_sigma),
            offset=float(offset),
            missing_angle=float(missing_angle),
            do_mw_transform=do_mw_transform,
            do_amp_transform=do_amp_transform,
            do_sample_kernel=do_sample_kernel,
            orientation=orientation,
        )

    def _generate_kernel(self, img_shape: Tuple[int, int]) -> np.ndarray:
        """Generate a Gaussian kernel for interpolating augmented content."""
        scale_values = [sample_scalar(self.scale, img_shape, axis) for axis in range(2)]
        loc_values = [sample_scalar(self.loc, img_shape, axis) for axis in range(2)]
        loc_values = np.array(loc_values) * np.array(img_shape, dtype=np.float32)

        coords = [
            np.linspace(-loc_values[i], img_shape[i] - loc_values[i], img_shape[i])
            for i in range(2)
        ]
        meshgrid = np.meshgrid(*coords, indexing="ij")
        kernel = np.exp(
            -0.5 * sum((meshgrid[i] / scale_values[i]) ** 2 for i in range(2))
        )
        return kernel.astype(np.float32)

    @staticmethod
    def run_interpolation(
        img: np.ndarray, img_modified: np.ndarray, kernel: np.ndarray
    ) -> np.ndarray:
        """Interpolate between the original and modified images using a kernel."""
        return img + kernel * (img_modified - img)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        params = self._draw_params()
        if not (self.amplitude_aug or self.missing_wedge_aug):
            return img

        fft_patch = normalize_and_fft_patch(img)
        if self.amplitude_aug and params.do_amp_transform:
            _, fake_spectrum = get_line_plot(
                n_points=max(8, int(img.shape[0] / 2)),
                smooth_sigma=params.smooth_sigma,
                step_sigma=params.step_sigma,
                offset=params.offset,
            )
            equal_kernel = rotational_kernel(fake_spectrum, img.shape)
            fft_patch *= equal_kernel

        if self.missing_wedge_aug and params.do_mw_transform:
            missing_mask = wedge_mask_2d(
                img.shape, params.missing_angle, params.orientation
            )
            fft_patch = np.where(missing_mask, fft_patch, 0.0)

        real_patch = fft_patch_to_real(fft_patch)
        real_patch = real_patch - real_patch.mean()
        std = real_patch.std()
        if std > 1e-6:
            real_patch = real_patch / std

        if params.do_sample_kernel:
            centered_img = img - img.mean()
            img_std = centered_img.std()
            if img_std > 1e-6:
                centered_img = centered_img / img_std
            kernel = self._generate_kernel(img.shape)
            real_patch = self.run_interpolation(centered_img, real_patch, kernel)

        real_patch = real_patch - real_patch.mean()
        std = real_patch.std()
        if std > 1e-6:
            real_patch = real_patch / std
        return real_patch.astype(np.float32)
