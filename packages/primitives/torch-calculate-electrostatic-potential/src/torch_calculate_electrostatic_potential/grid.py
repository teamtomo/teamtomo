"""Grid configuration and per-atom stencil geometry for 2D/3D potential grids."""

from __future__ import annotations

from dataclasses import dataclass

import einops
import numpy as np
import torch


def default_sublattice_radius(pixel_spacing: float) -> float:
    """Return the default per-atom stencil radius for a given voxel spacing.

    The radius is ``max(5.0, 3.0 * pixel_spacing)`` Angstroms. This matches the
    sizing used by ``torch-fit-in-map`` and ``ttsim3d`` when no explicit radius
    is supplied.
    """
    if pixel_spacing <= 0:
        raise ValueError("pixel_spacing must be positive")
    return max(5.0, 3.0 * pixel_spacing)


@dataclass
class _Stencil:
    """Precomputed local sublattice inserted around every atom."""

    dimensions: torch.Tensor  # (ndim,) int64
    coordinates: torch.Tensor  # (K, ndim)
    flat_indices: torch.Tensor  # (K,) int64
    center_cubic_index: torch.Tensor  # (ndim,) int64
    center_coordinate: torch.Tensor  # (ndim,)


class GridConfig:
    """Configuration for a 2D or 3D scattering-potential grid.

    Notes
    -----
    Dimensionality (2 or 3) is inferred from the length of `grid_shape`. Stencil is
    computed and cached on first access.

    Attributes
    ----------
    grid_shape : torch.Tensor
        Shape of the grid in voxels, (ndim,) int64.
    voxel_size : torch.Tensor
        Size of each voxel in Angstroms, (ndim,) float32.
    left_bottom_point : torch.Tensor
        Coordinates of the left-bottom corner of the grid, (ndim,) float32.
    right_upper_point : torch.Tensor
        Coordinates of the right-upper corner of the grid, (ndim,) float32.
    sublattice_radius : float
        Radius of the sublattice stencil around each atom, in Angstroms.
    equal_length : bool
        If True (default), all axes are symmetrically padded during
        construction to the longest axis length. This **mutates**
        ``grid_shape``, ``left_bottom_point``, and ``right_upper_point`` in
        place. If False, each axis keeps the requested voxel count.
    dtype : torch.dtype
        Data type for grid computations, by default torch.float32.
    device : torch.device
        Device for grid computations, by default torch.device("cpu").
    """

    def __init__(
        self,
        grid_shape: tuple[int, ...] | torch.Tensor,
        voxel_size: tuple[float, ...] | torch.Tensor,
        left_bottom_point: tuple[float, ...] | torch.Tensor | None = None,
        right_upper_point: tuple[float, ...] | torch.Tensor | None = None,
        sublattice_radius: float = 10.0,
        equal_length: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        """Initialization method for GridConfig.

        Parameters
        ----------
        grid_shape : tuple[int, ...] | torch.Tensor
            Shape of the grid in voxels, (ndim,) int64.
        voxel_size : tuple[float, ...] | torch.Tensor
            Size of each voxel in Angstroms, (ndim,) float32.
        left_bottom_point : tuple[float, ...] | torch.Tensor | None, optional
            Coordinates of the left-bottom corner of the grid, (ndim,) float32. If None,
            defaults to (0, 0, 0) for 3D or (0, 0) for 2D.
        right_upper_point : tuple[float, ...] | torch.Tensor | None, optional
            Coordinates of the right-upper corner of the grid, (ndim,) float32. If None,
            defaults to (grid_shape - 1) * voxel_size.
        sublattice_radius : float, optional
            Radius of the sublattice stencil around each atom, in Angstroms. Default is
            10.0.
        equal_length : bool, optional
            If True (default), every axis is symmetrically padded during
            construction to match the voxel count of the longest axis. This
            updates ``grid_shape``, ``left_bottom_point``, and
            ``right_upper_point`` in place before the config is used.
        dtype : torch.dtype, optional
            Data type for grid computations, by default torch.float32.
        device : torch.device, optional
            Device for grid computations, by default torch.device("cpu").
        """
        if device is None:
            device = torch.device("cpu")

        self.grid_shape = torch.as_tensor(
            grid_shape, dtype=torch.int64, device=device
        ).clone()
        self.ndim = int(self.grid_shape.numel())
        self.voxel_size = torch.as_tensor(voxel_size, dtype=dtype, device=device)

        # Validate inputs for dimensionality and size
        if self.ndim not in (2, 3):
            raise ValueError(f"grid_shape must have length 2 or 3, got {self.ndim}")

        if self.voxel_size.shape != self.grid_shape.shape:
            raise ValueError(
                f"grid_shape ({tuple(self.grid_shape.tolist())}) and voxel_size "
                f"({tuple(self.voxel_size.shape)}) must have the same length"
            )

        self.dtype = dtype
        self.device = device

        if left_bottom_point is None or right_upper_point is None:
            left_bottom_point = torch.zeros(self.ndim, dtype=dtype, device=device)
            right_upper_point = ((self.grid_shape - 1) * self.voxel_size).to(dtype)
        self.left_bottom_point = torch.as_tensor(
            left_bottom_point, dtype=dtype, device=device
        ).clone()
        self.right_upper_point = torch.as_tensor(
            right_upper_point, dtype=dtype, device=device
        ).clone()

        self.equal_length = equal_length
        if equal_length:
            self._constrain_to_equal_length()

        self.sublattice_radius = sublattice_radius
        self._strides = self._compute_strides()
        self._stencil: _Stencil | None = None

    def _constrain_to_equal_length(self) -> None:
        """Symmetrically pad shorter axes up to the longest axis length.

        Mutates ``grid_shape``, ``left_bottom_point``, and ``right_upper_point``.
        """
        max_extent = int(self.grid_shape.max().item())
        for axis in range(self.ndim):
            extra = max_extent - int(self.grid_shape[axis].item())

            if extra <= 0:
                continue

            extend_lo = extra // 2
            extend_hi = extra - extend_lo
            self.left_bottom_point[axis] -= extend_lo * self.voxel_size[axis]
            self.right_upper_point[axis] += extend_hi * self.voxel_size[axis]
            self.grid_shape[axis] = max_extent

    @property
    def grid_flat_size(self) -> int:
        """Total number of voxels in the grid."""
        return int(self.grid_shape.prod().item())

    @property
    def stencil(self) -> _Stencil:
        """Obtain the pre-computed stencil, or compute if not already constructed."""
        if self._stencil is None:
            self._stencil = self._build_stencil()
        return self._stencil

    def _compute_strides(self) -> torch.Tensor:
        """Row-major flat-index strides (last axis fastest)."""
        strides = torch.ones(self.ndim, dtype=torch.int64, device=self.device)

        for axis in range(self.ndim - 2, -1, -1):
            strides[axis] = strides[axis + 1] * self.grid_shape[axis + 1]

        return strides

    def convert_cubic_index_to_flat_index(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert (..., ndim) cubic indices to (...) flat indices."""
        return (indices.to(torch.int64) * self._strides).sum(dim=-1)

    def _build_stencil(self) -> _Stencil:
        """Build stencil for voxel coordinates and indices."""
        radii_voxels = [
            int(np.ceil((self.sublattice_radius / self.voxel_size[axis].item()) - 0.5))
            for axis in range(self.ndim)
        ]

        # Find extent in each dimension
        extents = []
        for axis in range(self.ndim):
            extent = min(2 * radii_voxels[axis] + 1, int(self.grid_shape[axis].item()))
            extents.append(extent if extent % 2 == 1 else extent - 1)

        dimensions = torch.tensor(extents, dtype=torch.int64, device=self.device)
        axes = [
            torch.arange(extent, device=self.device, dtype=self.dtype)
            for extent in extents
        ]

        # Build meshgrid and flat coordinate indices into that grid
        mesh = torch.meshgrid(*axes, indexing="ij")
        cubic_indices = torch.stack(
            [axis_values.ravel() for axis_values in mesh], dim=-1
        )  # (K, ndim)

        # Compute center cubic index and center coordinate of the stencil
        center_cubic_index = dimensions // 2
        center_coordinate = (
            self.left_bottom_point + self.voxel_size * center_cubic_index.to(self.dtype)
        )

        coordinates = cubic_indices * self.voxel_size + self.left_bottom_point
        flat_indices = self.convert_cubic_index_to_flat_index(cubic_indices)

        return _Stencil(
            dimensions=dimensions,
            coordinates=coordinates,
            flat_indices=flat_indices,
            center_cubic_index=center_cubic_index,
            center_coordinate=center_coordinate,
        )

    def _find_closest_voxel(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest voxel cubic index and center coordinate for each point."""
        offset = points - self.left_bottom_point

        cubic_indices = torch.floor((offset + self.voxel_size / 2) / self.voxel_size)
        cubic_indices = cubic_indices.to(torch.int64)
        cubic_indices = cubic_indices.clamp(
            min=torch.zeros(self.ndim, dtype=torch.int64, device=self.device),
            max=self.grid_shape - 1,
        )

        cubic_indices = cubic_indices.to(self.dtype)
        centers = cubic_indices * self.voxel_size + self.left_bottom_point
        centers = centers.to(self.dtype)

        return cubic_indices, centers

    def get_atom_anchors(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-atom stencil anchor, clamped so that anchor + stencil stays in-grid.

        Parameters
        ----------
        points : torch.Tensor
            Atom positions, shape (..., ndim).

        Returns
        -------
        anchor_flat : torch.Tensor
            Flat indices of the anchor voxels, shape (...,).
        anchor_coord : torch.Tensor
            Coordinates of the anchor voxels, shape (..., ndim).
        """
        stencil = self.stencil
        closest_cubic_index, closest_center = self._find_closest_voxel(points)

        transform_idx = closest_cubic_index - stencil.center_cubic_index
        transform_idx = transform_idx.clamp(
            min=torch.zeros(self.ndim, dtype=torch.int64, device=self.device),
            max=self.grid_shape - stencil.dimensions,
        )
        anchor_flat = self.convert_cubic_index_to_flat_index(transform_idx)

        transform_coord = closest_center - stencil.center_coordinate
        max_transform = self.voxel_size * (self.grid_shape - stencil.dimensions)
        max_transform = max_transform.to(self.dtype)

        transform_coord = transform_coord.clamp(
            min=torch.zeros(self.ndim, dtype=self.dtype, device=self.device),
            max=max_transform,
        )

        return anchor_flat, transform_coord

    def get_atom_stencil_voxels(
        self, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Absolute stencil voxel flat indices and coordinates around each atom.

        Parameters
        ----------
        points : torch.Tensor
            Atom positions, shape (..., ndim).

        Returns
        -------
        flat_indices : torch.Tensor
            Flat indices of the stencil voxels, shape (..., K).
        coordinates : torch.Tensor
            Coordinates of the stencil voxels, shape (..., K, ndim).
        """
        stencil = self.stencil

        anchor_flat, anchor_coord = self.get_atom_anchors(points)

        flat_indices = (
            einops.rearrange(anchor_flat, "... -> ... 1") + stencil.flat_indices
        )
        coordinates = (
            einops.rearrange(anchor_coord, "... d -> ... 1 d") + stencil.coordinates
        )

        return flat_indices, coordinates

    @classmethod
    def from_grid_shape_and_voxel_size(
        cls,
        grid_shape: tuple[int, ...],
        voxel_size: tuple[float, ...] | torch.Tensor,
        left_bottom_point: tuple[float, ...] | torch.Tensor | None = None,
        right_upper_point: tuple[float, ...] | torch.Tensor | None = None,
        *,
        center_zyx: tuple[float, float, float] | torch.Tensor | None = None,
        center_yx: tuple[float, float] | torch.Tensor | None = None,
        sublattice_radius: float = 10.0,
        equal_length: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> GridConfig:
        """Create a grid with optional explicit 3D or 2D center coordinates.

        ``center_zyx`` and ``center_yx`` locate the center voxel for odd grid
        dimensions and the midpoint between center voxels for even dimensions.
        """
        if device is None:
            device = torch.device("cpu")
        centers = [center for center in (center_zyx, center_yx) if center is not None]
        if len(centers) > 1:
            raise ValueError("provide only one of center_zyx or center_yx")
        if centers:
            if left_bottom_point is not None or right_upper_point is not None:
                raise ValueError("center coordinates cannot be combined with corners")
            ndim = len(grid_shape)
            expected_ndim = 3 if center_zyx is not None else 2
            if ndim != expected_ndim:
                center_name = "center_zyx" if center_zyx is not None else "center_yx"
                raise ValueError(
                    f"{center_name} requires a {expected_ndim}D grid_shape"
                )
            shape_tensor = torch.as_tensor(grid_shape, dtype=torch.int64, device=device)
            voxel_tensor = torch.as_tensor(voxel_size, dtype=dtype, device=device)
            center_tensor = torch.as_tensor(centers[0], dtype=dtype, device=device)
            half_span = (shape_tensor - 1).to(dtype) * voxel_tensor / 2
            left_bottom_point = center_tensor - half_span
            right_upper_point = center_tensor + half_span
        return cls(
            grid_shape=grid_shape,
            voxel_size=voxel_size,
            left_bottom_point=left_bottom_point,
            right_upper_point=right_upper_point,
            sublattice_radius=sublattice_radius,
            equal_length=equal_length,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_grid_shape_and_corner_points(
        cls,
        grid_shape: tuple[int, ...] | torch.Tensor,
        left_bottom_point: tuple[float, ...] | torch.Tensor,
        right_upper_point: tuple[float, ...] | torch.Tensor,
        sublattice_radius: float = 10.0,
        equal_length: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> GridConfig:
        """Helper to create a GridConfig from grid shape and corner points."""
        if device is None:
            device = torch.device("cpu")

        grid_shape_t = torch.as_tensor(grid_shape, dtype=torch.int64, device=device)
        left_bottom_point_t = torch.as_tensor(
            left_bottom_point, dtype=dtype, device=device
        )
        right_upper_point_t = torch.as_tensor(
            right_upper_point, dtype=dtype, device=device
        )
        voxel_size = (right_upper_point_t - left_bottom_point_t) / (grid_shape_t - 1)
        return cls(
            grid_shape=grid_shape_t,
            voxel_size=voxel_size,
            left_bottom_point=left_bottom_point_t,
            right_upper_point=right_upper_point_t,
            sublattice_radius=sublattice_radius,
            equal_length=equal_length,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_voxel_size_and_corner_points(
        cls,
        voxel_size: tuple[float, ...],
        left_bottom_point: tuple[float, ...],
        right_upper_point: tuple[float, ...],
        sublattice_radius: float = 10.0,
        equal_length: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> GridConfig:
        """Helper to create a GridConfig from voxel size and corner points."""
        if device is None:
            device = torch.device("cpu")

        voxel_size_t = torch.as_tensor(voxel_size, dtype=dtype, device=device)
        left_bottom_point_t = torch.as_tensor(
            left_bottom_point, dtype=dtype, device=device
        )
        right_upper_point_t = torch.as_tensor(
            right_upper_point, dtype=dtype, device=device
        )
        span = right_upper_point_t - left_bottom_point_t
        grid_shape = torch.ceil(span / voxel_size_t).to(torch.int64) + 1
        recomputed_voxel_size = span / (grid_shape - 1)
        return cls(
            grid_shape=grid_shape,
            voxel_size=recomputed_voxel_size,
            left_bottom_point=left_bottom_point_t,
            right_upper_point=right_upper_point_t,
            sublattice_radius=sublattice_radius,
            equal_length=equal_length,
            dtype=dtype,
            device=device,
        )
