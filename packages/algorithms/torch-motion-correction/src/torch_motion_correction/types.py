"""Helper dataclasses for grouping related parameters in motion correction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import einops
import pandas as pd
import torch
import torch.nn.functional as F
from torch_cubic_spline_grids import CubicBSplineGrid3d, CubicCatmullRomGrid3d
from torch_grid_utils.patch_grid import patch_grid_centers

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
class DeformationField:
    """A spatiotemporal deformation field with its interpolation type.

    Wraps the raw tensor data together with the grid interpolation type so that
    the type information travels with the data rather than being passed separately
    to each consuming function.

    Parameters
    ----------
    data : torch.Tensor
        Deformation field tensor with shape (2, nt, nh, nw) where 2 corresponds
        to (y, x) shifts in Angstroms, nt is the number of time points, and
        nh, nw are the spatial control point dimensions.
    grid_type : str
        Interpolation type for the deformation field. Either "catmull_rom" or
        "bspline". Default is "catmull_rom".
    """

    data: torch.Tensor
    grid_type: str = "catmull_rom"

    @property
    def resolution(self) -> tuple[int, int, int]:
        """Return (nt, nh, nw) resolution of the deformation field."""
        _, nt, nh, nw = self.data.shape
        return (nt, nh, nw)

    @property
    def shape(self) -> torch.Size:
        """Shape of the underlying data tensor."""
        return self.data.shape

    @property
    def device(self) -> torch.device:
        """Device of the underlying data tensor."""
        return self.data.device

    def to(self, device: torch.device | str) -> DeformationField:
        """Return a new DeformationField with data moved to the given device."""
        return DeformationField(data=self.data.to(device), grid_type=self.grid_type)

    def _as_cubic_grid(self):
        """Return the data as a cubic spline grid object for evaluation."""
        if self.grid_type == "catmull_rom":
            return CubicCatmullRomGrid3d.from_grid_data(self.data).to(self.data.device)
        elif self.grid_type == "bspline":
            return CubicBSplineGrid3d.from_grid_data(self.data).to(self.data.device)
        else:
            raise ValueError(
                f"Invalid grid_type: {self.grid_type!r}. "
                "Must be 'catmull_rom' or 'bspline'."
            )

    def evaluate_at(self, tyx: torch.Tensor) -> torch.Tensor:
        """Evaluate interpolated shifts at arbitrary (t, y, x) coordinates.

        Parameters
        ----------
        tyx : torch.Tensor
            (..., 3) tensor of normalized (t, y, x) coordinates in [0, 1].

        Returns
        -------
        torch.Tensor
            (..., 2) tensor of (y, x) shifts in Angstroms.
        """
        return self._as_cubic_grid()(tyx)

    def evaluate_at_t(
        self,
        t: float,
        grid_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Evaluate a dense shift grid at a single normalized timepoint.

        Parameters
        ----------
        t : float
            Normalized timepoint in [0, 1].
        grid_shape : tuple[int, int]
            (h, w) number of sample points in each spatial dimension.

        Returns
        -------
        torch.Tensor
            (2, h, w) tensor of (y, x) shifts in Angstroms.
        """
        h, w = grid_shape
        y = torch.linspace(0, 1, steps=h, device=self.device)
        x = torch.linspace(0, 1, steps=w, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        yx_grid = einops.rearrange([yy, xx], "yx h w -> (h w) yx")
        tyx_grid = F.pad(yx_grid, (1, 0), value=t)  # (h*w, 3)
        shifts = self.evaluate_at(tyx_grid)  # (h*w, 2)

        return einops.rearrange(shifts, "(h w) tyx -> tyx h w", h=h, w=w)

    def resample(self, target_resolution: tuple[int, int, int]) -> DeformationField:
        """Return a new DeformationField resampled to a different resolution.

        Parameters
        ----------
        target_resolution : tuple[int, int, int]
            (nt, nh, nw) resolution of the output deformation field.

        Returns
        -------
        DeformationField
            Resampled field with the same ``grid_type`` as the original.
        """
        nt, nh, nw = target_resolution
        t = torch.linspace(0, 1, steps=nt)
        y = torch.linspace(0, 1, steps=nh)
        x = torch.linspace(0, 1, steps=nw)
        tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
        tyx = einops.rearrange([tt, yy, xx], "tyx nt nh nw -> nt nh nw tyx")
        new_data = self.evaluate_at(tyx.to(self.device))
        new_data = einops.rearrange(new_data, "nt nh nw tyx -> tyx nt nh nw")

        return DeformationField(data=new_data, grid_type=self.grid_type)

    @classmethod
    def from_frame_shifts(
        cls,
        shifts: torch.Tensor,
        pixel_spacing: float,
        device: torch.device | None = None,
        grid_type: str = "catmull_rom",
    ) -> DeformationField:
        """Create a DeformationField from per-frame whole-image shifts.

        Parameters
        ----------
        shifts : torch.Tensor
            (t, 2) tensor of (y, x) shifts in pixels.
        pixel_spacing : float
            Pixel spacing in Angstroms per pixel.
        device : torch.device, optional
            Device for the output tensor.
        grid_type : str
            Interpolation type. Default is 'catmull_rom'.

        Returns
        -------
        DeformationField
            Deformation field with shape (2, t, 1, 1).
        """
        if device is not None:
            shifts = shifts.to(device)
        data = einops.rearrange(shifts * pixel_spacing, "t c -> c t 1 1")
        return cls(data=data, grid_type=grid_type)

    def to_csv(self, output_path: str | Path) -> None:
        """Write the deformation field to a CSV file.

        Parameters
        ----------
        output_path : str or Path
            Destination CSV path. Parent directories are created if needed.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.data.detach().cpu()
        _, t, h, w = data.shape
        t_idx, h_idx, w_idx = torch.meshgrid(
            torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij"
        )
        df = pd.DataFrame(
            {
                "t": t_idx.flatten().numpy(),
                "h": h_idx.flatten().numpy(),
                "w": w_idx.flatten().numpy(),
                "y_shift": data[0].flatten().numpy(),
                "x_shift": data[1].flatten().numpy(),
            }
        )
        df.to_csv(output_path, index=False)

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        device: torch.device | None = None,
        grid_type: str = "catmull_rom",
    ) -> DeformationField:
        """Load a DeformationField from a CSV file written by :meth:`to_csv`.

        Parameters
        ----------
        csv_path : str or Path
            Path to the input CSV file.
        device : torch.device, optional
            Device for the output tensor.
        grid_type : str
            Interpolation type. Default is 'catmull_rom'.

        Returns
        -------
        DeformationField
        """
        df = pd.read_csv(csv_path)

        unique_t = sorted(df["t"].unique())
        unique_h = sorted(df["h"].unique())
        unique_w = sorted(df["w"].unique())
        t_to_idx = {v: i for i, v in enumerate(unique_t)}
        h_to_idx = {v: i for i, v in enumerate(unique_h)}
        w_to_idx = {v: i for i, v in enumerate(unique_w)}

        t, h, w = len(unique_t), len(unique_h), len(unique_w)
        data = torch.zeros((2, t, h, w), dtype=torch.float32)
        t_idxs = df["t"].map(t_to_idx).to_numpy()
        h_idxs = df["h"].map(h_to_idx).to_numpy()
        w_idxs = df["w"].map(w_to_idx).to_numpy()
        data[0, t_idxs, h_idxs, w_idxs] = torch.tensor(
            df["y_shift"].to_numpy(), dtype=torch.float32
        )
        data[1, t_idxs, h_idxs, w_idxs] = torch.tensor(
            df["x_shift"].to_numpy(), dtype=torch.float32
        )
        if device is not None:
            data = data.to(device)
        return cls(data=data, grid_type=grid_type)


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
    n_iterations : int
        Number of optimization iterations. Default is 100.
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
    """

    n_iterations: int = 100
    optimizer_type: str = "adam"
    loss_type: str = "mse"
    grid_type: str = "catmull_rom"
    optimizer_kwargs: dict | None = None


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
