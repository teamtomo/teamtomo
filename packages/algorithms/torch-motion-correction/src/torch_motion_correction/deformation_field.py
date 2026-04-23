"""Helper class for representing and evaluating deformation fields.

Terminology disambiguation
--------------------------
- A "field" refers to the `DeformationField` class or an instance of it. These represent
  motion (deformation) parameterizations.
- A "grid" refers to the underlying cubic spline grid (e.g. `CubicCatmullRomGrid3d`)
  that parameterizes the deformation field. These objects perform the actual
  interpolation and evaluation of shifts at coordinates.
- Variables "data" refer to the raw tensor data of shape (2, nt, nh, nw). Both the
    `DeformationField` and the underlying grid classes wrap this tensor data.
"""

from pathlib import Path
from typing import Optional

import einops
import pandas as pd
import torch
import torch.nn.functional as F
from torch_cubic_spline_grids import CubicBSplineGrid3d, CubicCatmullRomGrid3d


class DeformationField:
    """A spatio-temporal deformation field represented by cubic spline grids.

    Wraps the tensor data along with the grid type and shift units into a convenient
    class. Functional methods are also included for convenience for initializing,
    evaluating, and resampling deformation fields.

    Parameters
    ----------
    data: torch.Tensor
        Deformation field tensor with shape (2, nt, nh, nw) where the first dimension
        corresponds to (y, x) shifts in Angstroms, nt is the number of timepoints,
        and (nh, nw) is the spatial grid resolution.
    grid_type: str
        Interpolation type for the deformation field. One of "catmull_rom" or
        "bspline". Default is "catmull_rom".
    """

    data: torch.Tensor
    grid_type: str
    _grid: CubicCatmullRomGrid3d | CubicBSplineGrid3d

    def __init__(
        self,
        data: torch.Tensor,
        grid_type: str = "catmull_rom",
    ):
        self.data = data
        self.grid_type = grid_type

        # CubicSplineGrid.data is a property backed by an nn.Parameter. Passing a
        # Parameter to from_grid_data triggers nn.Module.__setattr__ which tries to
        # register_parameter('data', ...) and conflicts with the property descriptor.
        # We pass a plain tensor (shared storage) to avoid this conflict.
        grid_data = data.data if isinstance(data, torch.nn.Parameter) else data

        if grid_type == "catmull_rom":
            self._grid = CubicCatmullRomGrid3d.from_grid_data(grid_data)
        elif grid_type == "bspline":
            self._grid = CubicBSplineGrid3d.from_grid_data(grid_data)
        else:
            raise ValueError(
                f"Unsupported grid type: {grid_type!r}. "
                "Must be 'catmull_rom' or 'bspline'."
            )

    # --- Convenience methods for evaluation --------------

    def __call__(self, tyx: torch.Tensor) -> torch.Tensor:
        """Evaluate at normalized (t, y, x) coordinates, preserving gradients."""
        return self._grid(tyx)

    def parameters(self):
        """Return optimizable parameters of the underlying spline grid."""
        return self._grid.parameters()

    @property
    def resolution(self) -> tuple[int, int, int]:
        """Return the resolution of the deformation field as (nt, nh, nw)."""
        _, nt, nh, nw = self.data.shape
        return (nt, nh, nw)

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the deformation field tensor."""
        return self.data.shape

    @property
    def device(self) -> torch.device:
        """Return the device of the deformation field tensor."""
        return self.data.device

    def to(self, device: torch.device | str) -> "DeformationField":
        """Return a copy of the deformation field on the specified device."""
        return DeformationField(
            data=self.data.to(device),
            grid_type=self.grid_type,
        )

    def evaluate_at_t(self, t: float, grid_shape: tuple[int, int]) -> torch.Tensor:
        """Evaluate the deformation field at a single timepoint for a spatial grid.

        Parameters
        ----------
        t: float
            Normalized timepoint in [0, 1] at which to evaluate the field.
        grid_shape: tuple[int, int]
            Spatial grid shape as (height, width).

        Returns
        -------
        torch.Tensor
            (2, height, width) tensor of (y, x) shifts in Angstroms.
        """
        h, w = grid_shape
        y = torch.linspace(0, 1, steps=h, device=self.device)
        x = torch.linspace(0, 1, steps=w, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        yx_grid = einops.rearrange([yy, xx], "yx h w -> (h w) yx")
        tyx_grid = F.pad(yx_grid, (1, 0), value=t)  # (h*w, 3)
        shifts = self(tyx_grid)  # (h*w, 2)
        return einops.rearrange(shifts, "(h w) yx -> yx h w", h=h, w=w)

    # --- Creating or modifying deformation fields --------------

    def resample(self, target_resolution: tuple[int, int, int]) -> "DeformationField":
        """Return a new DeformationField resampled to a target resolution.

        Parameters
        ----------
        target_resolution: tuple[int, int, int]
            Target resolution as (nt, nh, nw).

        Returns
        -------
        DeformationField
            New field resampled to the target resolution with the same grid type.
        """
        nt, nh, nw = target_resolution
        t = torch.linspace(0, 1, steps=nt)
        y = torch.linspace(0, 1, steps=nh)
        x = torch.linspace(0, 1, steps=nw)
        tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
        tyx = einops.rearrange([tt, yy, xx], "tyx nt nh nw -> nt nh nw tyx")
        new_data = self(tyx.to(self.device))
        new_data = einops.rearrange(new_data, "nt nh nw yx -> yx nt nh nw")
        return DeformationField(data=new_data, grid_type=self.grid_type)

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
    def from_frame_shifts(
        cls,
        shifts: torch.Tensor,
        pixel_spacing: float,
        device: torch.device | None = None,
        grid_type: str = "catmull_rom",
    ) -> "DeformationField":
        """Initialize a deformation field from per-frame whole-image shifts.

        Parameters
        ----------
        shifts: torch.Tensor
            (nt, 2) tensor of (y, x) shifts for each frame in pixels.
        pixel_spacing: float
            Pixel spacing in Angstroms to convert shifts to Angstroms.
        device: torch.device | None
            Device for the output tensor. Defaults to the device of ``shifts``.
        grid_type: str
            Grid type. One of "catmull_rom" or "bspline". Default is "catmull_rom".

        Returns
        -------
        DeformationField
            Deformation field with shape (2, nt, 1, 1).
        """
        if device is None:
            device = shifts.device
        data = einops.rearrange(shifts * pixel_spacing, "t c -> c t 1 1").to(device)
        return cls(data=data, grid_type=grid_type)

    @classmethod
    def from_initial_field(
        cls,
        resolution: tuple[int, int, int],
        initial_field: Optional["DeformationField"],
        grid_type: str,
        device: torch.device,
    ) -> tuple["DeformationField", "DeformationField"]:
        """Create a new optimizable field and a frozen base field.

        The new (optimizable) field starts at zero shifts. The base field holds the
        (resampled) shifts from ``initial_field``, or zeros if ``initial_field`` is
        None.

        Parameters
        ----------
        resolution : tuple[int, int, int]
            (nt, nh, nw) control-point resolution for both fields.
        initial_field : DeformationField or None
            Existing field to use as the frozen base. When None the base field
            starts at zero shifts.
        grid_type : str
            "catmull_rom" or "bspline".
        device : torch.device
            Device to place both fields on.

        Returns
        -------
        new_field : DeformationField
            Zero-initialised, optimisable field. Pass ``.parameters()`` to an
            optimizer.
        base_field : DeformationField
            Frozen field holding the initial (resampled) shifts.
        """
        if grid_type not in ("catmull_rom", "bspline"):
            raise ValueError(
                f"Unsupported grid type: {grid_type!r}. "
                "Must be 'catmull_rom' or 'bspline'."
            )
        grid_class = (
            CubicCatmullRomGrid3d if grid_type == "catmull_rom" else CubicBSplineGrid3d
        )

        new_raw_grid = grid_class(resolution=resolution, n_channels=2).to(device)
        new_field = cls(data=next(new_raw_grid.parameters()), grid_type=grid_type)
        new_field._grid = new_raw_grid

        if initial_field is None:
            base_data = torch.zeros(size=(2, *resolution), device=device)
        else:
            base_data = initial_field.resample(resolution).data.to(device)
            base_data -= torch.mean(base_data)
        base_raw_grid = grid_class.from_grid_data(base_data).to(device)
        base_field = cls(data=base_data, grid_type=grid_type)
        base_field._grid = base_raw_grid

        return new_field, base_field

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        device: torch.device | None = None,
        grid_type: str = "catmull_rom",
    ) -> "DeformationField":
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
