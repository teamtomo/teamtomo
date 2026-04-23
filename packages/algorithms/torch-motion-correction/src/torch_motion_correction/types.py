"""Helper dataclasses for grouping related parameters in motion correction."""

from dataclasses import dataclass

import torch

from torch_motion_correction.patch_grid import patch_grid_centers
from torch_motion_correction.patch_utils import ImagePatchIterator


@dataclass
class FourierFilterConfig:
    """Configuration for Fourier-space filtering applied during motion estimation.

    Parameters
    ----------
    b_factor : float
        B-factor in Angstroms^2 used to downweight high-frequency content.
        Default is 500.
    frequency_range : tuple[float, float]
        Bandpass frequency range as (low_cutoff, high_cutoff) in Angstroms.
        Default is (300, 10).
    """

    b_factor: float = 500
    frequency_range: tuple[float, float] = (300, 10)


@dataclass
class PatchSamplingConfig:
    """Configuration for patch extraction from a movie.

    Groups patch shape and overlap fraction and provides a convenience method to
    construct an :class:`~torch_motion_correction.patch_utils.ImagePatchIterator`.

    Parameters
    ----------
    patch_shape : tuple[int, int]
        Patch dimensions as (height, width) in pixels.
    overlap : float
        Fraction of the patch size to use as overlap between adjacent patches.
        Must be in [0, 1). Default is 0.5 (50% overlap).
    distribute_patches : bool
        Whether to distribute patches evenly across the image. Default is True.
    """

    patch_shape: tuple[int, int]
    overlap: float = 0.5
    distribute_patches: bool = True

    @property
    def patch_step(self) -> tuple[int, int]:
        """Step size between patches derived from patch_shape and overlap."""
        ph, pw = self.patch_shape
        return (
            max(1, int(ph * (1 - self.overlap))),
            max(1, int(pw * (1 - self.overlap))),
        )

    def get_patch_iterator(
        self,
        image: torch.Tensor,
        device: torch.device | None = None,
    ) -> ImagePatchIterator:
        """Build an :class:`~torch_motion_correction.patch_utils.ImagePatchIterator`.

        Parameters
        ----------
        image : torch.Tensor
            Movie frames with shape (t, H, W).
        device : torch.device, optional
            Device on which to place the patch position tensors.

        Returns
        -------
        ImagePatchIterator
            Iterator that yields (patch_batch, patch_centers) mini-batches.
        """
        t, h, w = image.shape
        ph, pw = self.patch_shape
        step_h, step_w = self.patch_step
        patch_positions = patch_grid_centers(
            image_shape=(t, h, w),
            patch_shape=(1, ph, pw),
            patch_step=(1, step_h, step_w),
            distribute_patches=self.distribute_patches,
            device=device,
        )
        return ImagePatchIterator(
            image=image,
            patch_size=self.patch_shape,
            control_points=patch_positions,
        )


@dataclass
class OptimizationConfig:
    """Configuration for the gradient-based optimization in local motion estimation.

    Parameters
    ----------
    max_iterations : int
        Maximum number of optimization iterations. Default is 100.
    optimizer_type : str
        Optimizer to use. One of "adam", "sgd", "rmsprop", "lbfgs".
        Default is "adam".
    loss_type : str
        Loss function to minimize. One of "mse", "ncc", "cc". Default is "mse".
    grid_type : str
        Cubic spline interpolation type for the deformation field being optimized.
        Either "catmull_rom" or "bspline". Default is "catmull_rom".
    optimizer_kwargs : dict, optional
        Additional keyword arguments forwarded to the optimizer constructor.
    early_stopping : bool
        Whether to enable early stopping. Default is False.
    early_stopping_patience : int
        Number of steps without significant improvement before stopping. Default is 5.
    early_stopping_window_size : int
        Number of recent loss values to average for smoothing. Default is 3.
    early_stopping_tolerance : float
        Minimum relative improvement in the smoothed loss to reset the patience
        counter. Default is 1e-5.
    """

    max_iterations: int = 100
    optimizer_type: str = "adam"
    loss_type: str = "mse"
    grid_type: str = "catmull_rom"
    optimizer_kwargs: dict | None = None
    early_stopping_patience: int = 5
    early_stopping_window_size: int = 3
    early_stopping_tolerance: float = 1e-5
    early_stopping: bool = False


@dataclass
class XCRefinementConfig:
    """Post-processing configuration for cross-correlation motion estimation.

    Parameters
    ----------
    sub_pixel_refinement : bool
        Whether to apply parabolic sub-pixel refinement to cross-correlation
        peaks. Default is True.
    temporal_smoothing : bool
        Whether to apply Savitzky-Golay temporal smoothing across frames for
        each patch position. Default is True.
    smoothing_window_size : int
        Window size for temporal smoothing (must be odd). Default is 5.
    outlier_rejection : bool
        Whether to replace outlier patch shifts with the mean of valid patches.
        Default is True.
    outlier_threshold : float
        Z-score threshold (standard deviations from median) above which a patch
        shift is considered an outlier. Default is 3.0.
    """

    sub_pixel_refinement: bool = True
    temporal_smoothing: bool = True
    smoothing_window_size: int = 5
    outlier_rejection: bool = True
    outlier_threshold: float = 3.0
