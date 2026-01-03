"""Utilities for extracting a grid of patches from an image."""

from typing import Any, cast

import torch

from ._patch_grid_centers import patch_grid_centers
from ._patch_grid_indices import patch_grid_indices


def patch_grid(
    images: torch.Tensor,
    patch_shape: tuple[int, int] | tuple[int, int, int],
    patch_step: tuple[int, int] | tuple[int, int, int],
    distribute_patches: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a grid of patches from an image.

    Parameters
    ----------
    images: torch.Tensor
        (..., h, w) array of images.
    patch_shape: tuple[int, int] | tuple[int, int, int]
        (patch_h, patch_w) or (patch_d, patch_h, patch_w) of patches to be extracted.
    patch_step: tuple[int, int] | tuple[int, int, int]
        The target distance between patch centers in dimensions `h` and `w`.
    distribute_patches: bool
        Whether to distribute patches across the entire dimension length (`True`)
        or leave a gap at the end of each dimension (`False`).

    Returns
    -------
    patches, patch_centers: tuple[torch.Tensor, torch.Tensor]
        `(..., grid_h, grid_w, patch_h, patch_w)` grid of 2D patches
        and `(..., grid_h, grid_w, 2)` array of coordinates of patch centers
        in image dimensions `h` and `w`.
    """
    if len(patch_shape) != len(patch_step):
        raise ValueError(
            "patch shape and step must have the same number of dimensions."
        )
    ndim = len(patch_shape)
    if ndim == 2:
        patches, patch_centers = _patch_grid_2d(
            images=images,
            patch_shape=cast(tuple[int, int], patch_shape),
            patch_step=cast(tuple[int, int], patch_step),
            distribute_patches=distribute_patches,
        )
    elif ndim == 3:
        patches, patch_centers = _patch_grid_3d(
            images=images,
            patch_shape=cast(tuple[int, int, int], patch_shape),
            patch_step=cast(tuple[int, int, int], patch_step),
            distribute_patches=distribute_patches,
        )
    else:
        raise NotImplementedError()
    return patches, patch_centers


def _patch_grid_2d(
    images: torch.Tensor,
    patch_shape: tuple[int, int],
    patch_step: tuple[int, int],
    distribute_patches: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a grid of 2D patches from 2D image(s).

    Parameters
    ----------
    images: torch.Tensor
        `(..., h, w)` array of 2D images.
    patch_shape: tuple[int, int]
        `(patch_h, patch_w)` of patches to be extracted.
    patch_step: tuple[int, int]
        The target distance between patch centers in dimensions `h` and `w`.
    distribute_patches: bool
        Whether to distribute patches across the entire dimension length (`True`)
        or leave a gap at the end of each dimension (`False`).

    Returns
    -------
    patches, patch_centers: tuple[torch.Tensor, torch.Tensor]
        `(..., grid_h, grid_w, patch_h, patch_w)` grid of 2D patches
        and `(..., grid_h, grid_w, 2)` array of coordinates of patch centers
        in image dimensions `h` and `w`.
    """
    patch_centers = patch_grid_centers(
        image_shape=images.shape[-2:],
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=distribute_patches,
        device=images.device,
    )
    indices_result = patch_grid_indices(
        image_shape=images.shape[-2:],
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=distribute_patches,
        device=images.device,
    )
    patch_idx_h, patch_idx_w = cast(tuple[torch.Tensor, torch.Tensor], indices_result)
    patches = images[..., patch_idx_h, patch_idx_w]
    return patches, patch_centers


def _patch_grid_3d(
    images: torch.Tensor,
    patch_shape: tuple[int, int, int],
    patch_step: tuple[int, int, int],
    distribute_patches: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract a grid of 3D patches from 3D image(s).

    Parameters
    ----------
    images: torch.Tensor
        `(..., h, w)` array of 3D images.
    patch_shape: tuple[int, int, int]
        `(patch_h, patch_w)` of patches to be extracted.
    patch_step: tuple[int, int]
        The target distance between patch centers in dimensions `h` and `w`.
    distribute_patches: bool
        Whether to distribute patches across the entire dimension length (`True`)
        or leave a gap at the end of each dimension (`False`).

    Returns
    -------
    patches, patch_centers: tuple[torch.Tensor, torch.Tensor]
        `(..., grid_d, grid_h, grid_w, patch_d, patch_h, patch_w)` grid of 2D patches
        and `(..., grid_h, grid_w, 2)` array of coordinates of patch centers
        in image dimensions `h` and `w`.
    """
    patch_centers = patch_grid_centers(
        image_shape=images.shape[-3:],
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=distribute_patches,
        device=images.device,
    )
    indices_result = patch_grid_indices(
        image_shape=images.shape[-3:],
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=distribute_patches,
        device=images.device,
    )
    patch_idx_d, patch_idx_h, patch_idx_w = cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor], indices_result
    )
    patches = images[..., patch_idx_d, patch_idx_h, patch_idx_w]
    return patches, patch_centers


class LazyPatchGrid:
    """
    A lazy patch grid that computes patches on-demand to avoid memory overhead.

    Presents the same interface as a pre-computed patch grid but only extracts
    patches when they are actually accessed.
    """

    def __init__(
        self,
        images: torch.Tensor,
        patch_shape: tuple[int, int] | tuple[int, int, int],
        patch_step: tuple[int, int] | tuple[int, int, int],
        distribute_patches: bool = True,
    ) -> None:
        """
        Initialization from image shape, patch size, and patch step.

        Parameters
        ----------
        images: torch.Tensor
            (..., h, w) or (..., d, h, w) array of images.
        patch_shape: tuple[int, int] | tuple[int, int, int]
            (patch_h, patch_w) or (patch_d, patch_h, patch_w) of patches.
        patch_step: tuple[int, int] | tuple[int, int, int]
            The target distance between patch centers.
        distribute_patches: bool
            Whether to distribute patches across the entire dimension length.
        """
        self.images = images
        self.patch_shape = patch_shape
        self.patch_step = patch_step
        self.distribute_patches = distribute_patches

        # Determine if this is 2D or 3D
        self.ndim = len(patch_shape)

        # Pre-compute patch centers and indices - these are lightweight
        if self.ndim == 2:
            self.patch_centers = patch_grid_centers(
                image_shape=images.shape[-2:],
                patch_shape=patch_shape,
                patch_step=patch_step,
                distribute_patches=distribute_patches,
                device=images.device,
            )

            indices_result = patch_grid_indices(
                image_shape=images.shape[-2:],
                patch_shape=patch_shape,
                patch_step=patch_step,
                distribute_patches=distribute_patches,
                device=images.device,
            )
            self.patch_idx_h, self.patch_idx_w = cast(
                tuple[torch.Tensor, torch.Tensor], indices_result
            )

            # Store grid dimensions
            self.grid_shape = self.patch_centers.shape[:-1]  # (grid_h, grid_w)

        elif self.ndim == 3:
            self.patch_centers = patch_grid_centers(
                image_shape=images.shape[-3:],
                patch_shape=patch_shape,
                patch_step=patch_step,
                distribute_patches=distribute_patches,
                device=images.device,
            )

            indices_result = patch_grid_indices(
                image_shape=images.shape[-3:],
                patch_shape=patch_shape,
                patch_step=patch_step,
                distribute_patches=distribute_patches,
                device=images.device,
            )
            self.patch_idx_d, self.patch_idx_h, self.patch_idx_w = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor], indices_result
            )

            # Store grid dimensions
            self.grid_shape = self.patch_centers.shape[:-1]  # (grid_d, grid_h, grid_w)
        else:
            raise NotImplementedError("Only 2D and 3D patches supported")

        # Full shape would be (..., *grid_shape, *patch_shape)
        self._shape = images.shape[: -self.ndim] + self.grid_shape + patch_shape

        # Cache for storing computed patches
        self._cache: dict[Any, torch.Tensor] = {}
        self._cache_keys: set[Any] = set()

    @property
    def shape(self) -> torch.Size:
        """Return the shape that the full patch grid would have."""
        return torch.Size(self._shape)

    @property
    def device(self) -> torch.device:
        """Return the device of the underlying images."""
        return self.images.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the underlying images."""
        return self.images.dtype

    def __getitem__(self, key: int | slice | tuple | Any) -> torch.Tensor:
        """
        Extract patches on-demand based on indexing.

        This is where the magic happens - only the requested patches are computed.

        Parameters
        ----------
        key: int | slice | tuple | Any
            Indexing key (int, slice, tuple, etc.) to extract patches from.

        Returns
        -------
        patches: torch.Tensor
            Patches at the specified indices.
        """
        # Convert key to a hashable cache key
        cache_key = self._make_cache_key(key)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Extract patches based on the key
        if self.ndim == 2:
            patches = self._extract_patches_2d(key)
        else:  # ndim == 3
            patches = self._extract_patches_3d(key)

        # Cache the result
        self._cache[cache_key] = patches
        self._cache_keys.add(cache_key)

        # Limit cache size to prevent memory bloat
        if len(self._cache) > 50:  # Conservative limit for patches
            self._evict_cache()

        return patches

    def _make_cache_key(self, key: int | slice | tuple | Any) -> tuple | int | Any:
        """
        Convert indexing key to a hashable cache key.

        Parameters
        ----------
        key: int | slice | tuple | Any
            Indexing key to convert.

        Returns
        -------
        cache_key: tuple | int | Any
            Hashable cache key.
        """
        if isinstance(key, tuple):
            return tuple(
                (s.start, s.stop, s.step)
                if isinstance(s, slice)
                else s
                if isinstance(s, (int, type(Ellipsis)))
                else tuple(s.tolist())
                if isinstance(s, torch.Tensor)
                else s
                for s in key
            )
        elif isinstance(key, slice):
            return (key.start, key.stop, key.step)
        elif isinstance(key, torch.Tensor):
            return tuple(key.tolist())
        else:
            return key

    def _evict_cache(self) -> None:
        """
        Remove oldest cache entries to free memory.

        Returns
        -------
        None
        """
        keys_to_remove = list(self._cache_keys)[: len(self._cache_keys) // 2]
        for key in keys_to_remove:
            self._cache.pop(key, None)
            self._cache_keys.discard(key)

    def _extract_patches_2d(self, key: int | slice | tuple | Any) -> torch.Tensor:
        """
        Extract 2D patches for the given key.

        Parameters
        ----------
        key: int | slice | tuple | Any
            Indexing key to extract patches from.

        Returns
        -------
        patches: torch.Tensor
            Patches at the specified indices.
        """
        # Handle different indexing patterns
        if isinstance(key, tuple) and len(key) >= 2:
            # Extract grid indices (last 2 dimensions of the key)
            batch_key = key[:-2] if len(key) > 2 else ()
            grid_key = key[-2:]
            gh_key, gw_key = grid_key
        elif isinstance(key, (int, slice)):
            batch_key = ()
            gh_key, gw_key = key, slice(None)
        else:
            batch_key = ()
            gh_key, gw_key = slice(None), slice(None)

        # Get subset of patch indices
        patch_idx_h_subset = self.patch_idx_h[gh_key]
        patch_idx_w_subset = self.patch_idx_w[gw_key]

        # Select image subset
        if batch_key:
            image_subset = self.images[batch_key]
        else:
            image_subset = self.images

        # Extract patches using advanced indexing
        patches = image_subset[..., patch_idx_h_subset, patch_idx_w_subset]
        return patches

    def _extract_patches_3d(self, key: int | slice | tuple | Any) -> torch.Tensor:
        """
        Extract 3D patches for the given key.

        Parameters
        ----------
        key: int | slice | tuple | Any
            Indexing key to extract patches from.

        Returns
        -------
        patches: torch.Tensor
            Patches at the specified indices.
        """
        # Handle different indexing patterns
        if isinstance(key, tuple) and len(key) >= 3:
            batch_key = key[:-3] if len(key) > 3 else ()
            grid_key = key[-3:]
            gd_key, gh_key, gw_key = grid_key
        elif isinstance(key, tuple) and len(key) == 2:
            batch_key = ()
            gd_key = slice(None)
            gh_key, gw_key = key
        elif isinstance(key, (int, slice)):
            batch_key = ()
            gd_key, gh_key, gw_key = key, slice(None), slice(None)
        else:
            batch_key = ()
            gd_key, gh_key, gw_key = slice(None), slice(None), slice(None)

        # For the complex indexing structure, we need to use the original
        # patch_grid approach. The patch indices have shapes like
        # (gd, 1, 1, pd, 1, 1), (1, gh, 1, 1, ph, 1), etc.

        # Build the indexing arrays properly
        gd, gh, gw = self.grid_shape

        # Handle the complex indexing by creating the right selection
        if isinstance(gd_key, torch.Tensor):
            # Random indexing case
            if isinstance(gh_key, torch.Tensor) and isinstance(gw_key, torch.Tensor):
                # All are tensor indices - use advanced indexing

                # Select image subset
                if batch_key:
                    image_subset = self.images[batch_key]
                else:
                    image_subset = self.images

                # Extract patches one by one and stack them
                patches_list = []
                for i in range(len(gd_key)):
                    gd_idx = gd_key[i].item()
                    gh_idx = gh_key[i].item()
                    gw_idx = gw_key[i].item()

                    # Get the specific patch indices for this grid position
                    patch_d_idx = self.patch_idx_d[gd_idx]
                    patch_h_idx = self.patch_idx_h[gh_idx]
                    patch_w_idx = self.patch_idx_w[gw_idx]

                    # Extract the single patch
                    patch = image_subset[..., patch_d_idx, patch_h_idx, patch_w_idx]
                    patches_list.append(patch)

                # Stack all patches
                patches = torch.stack(
                    patches_list, dim=-4
                )  # Stack in the patch dimension
                return patches
            else:
                raise NotImplementedError("Mixed tensor/slice indexing not supported")
        else:
            # Original slice-based approach should work
            patch_idx_d_subset = self.patch_idx_d[gd_key]
            patch_idx_h_subset = self.patch_idx_h[gh_key]
            patch_idx_w_subset = self.patch_idx_w[gw_key]

            # Select image subset
            if batch_key:
                image_subset = self.images[batch_key]
            else:
                image_subset = self.images

            # Extract patches using advanced indexing
            patches = image_subset[
                ..., patch_idx_d_subset, patch_idx_h_subset, patch_idx_w_subset
            ]
            return patches

    def random_subset(self, n_patches: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random subset of patches - optimized for your use case.

        Parameters
        ----------
        n_patches: int
            Number of random patches to extract

        Returns
        -------
        patches: torch.Tensor
            Random subset of patches
        centers: torch.Tensor
            Corresponding patch centers
        """
        # Use a simpler approach based on the working generator logic
        if self.ndim == 2:
            gh, gw = self.grid_shape
            # Generate random grid positions
            idx_gh = torch.randint(0, gh, (n_patches,), device=self.device)
            idx_gw = torch.randint(0, gw, (n_patches,), device=self.device)

            # Extract patches using direct slicing (like in the generator)
            patches_list = []
            centers_list = []

            patch_shape_2d = cast(tuple[int, int], self.patch_shape)
            ph, pw = patch_shape_2d

            for i in range(n_patches):
                gi, gj = idx_gh[i].item(), idx_gw[i].item()
                center_h, center_w = self.patch_centers[gi, gj]

                # Calculate patch boundaries (same logic as generator)
                h_start = max(0, int(center_h - ph // 2))
                h_end = min(self.images.shape[-2], h_start + ph)
                h_start = max(0, h_end - ph)

                w_start = max(0, int(center_w - pw // 2))
                w_end = min(self.images.shape[-1], w_start + pw)
                w_start = max(0, w_end - pw)

                # Extract patch using slicing
                patch = self.images[..., h_start:h_end, w_start:w_end]
                patches_list.append(patch)
                centers_list.append(self.patch_centers[gi, gj])

            patches = torch.stack(patches_list, dim=-3)  # Stack in patch dimension
            centers = torch.stack(centers_list, dim=0)

        else:  # ndim == 3
            gd, gh, gw = self.grid_shape
            # Generate random grid positions
            idx_gd = torch.randint(0, gd, (n_patches,), device=self.device)
            idx_gh = torch.randint(0, gh, (n_patches,), device=self.device)
            idx_gw = torch.randint(0, gw, (n_patches,), device=self.device)

            # Extract patches using direct slicing (like in the generator)
            patches_list = []
            centers_list = []

            patch_shape_3d = cast(tuple[int, int, int], self.patch_shape)
            pd, ph, pw = patch_shape_3d

            for i in range(n_patches):
                gi, gj, gk = idx_gd[i].item(), idx_gh[i].item(), idx_gw[i].item()
                center_d, center_h, center_w = self.patch_centers[gi, gj, gk]

                # Same logic as the working generator
                if pd == 1:
                    # Time series case - don't slice time dimension
                    h_start = max(0, int(center_h - ph // 2))
                    h_end = min(self.images.shape[-2], h_start + ph)
                    h_start = max(0, h_end - ph)

                    w_start = max(0, int(center_w - pw // 2))
                    w_end = min(self.images.shape[-1], w_start + pw)
                    w_start = max(0, w_end - pw)

                    # Extract patch and add singleton dimension
                    patch = self.images[..., h_start:h_end, w_start:w_end]
                    patch = patch.unsqueeze(-3)  # Add singleton spatial dimension
                else:
                    # True 3D case
                    d_start = max(0, int(center_d - pd // 2))
                    d_end = min(self.images.shape[-3], d_start + pd)
                    d_start = max(0, d_end - pd)

                    h_start = max(0, int(center_h - ph // 2))
                    h_end = min(self.images.shape[-2], h_start + ph)
                    h_start = max(0, h_end - ph)

                    w_start = max(0, int(center_w - pw // 2))
                    w_end = min(self.images.shape[-1], w_start + pw)
                    w_start = max(0, w_end - pw)

                    patch = self.images[
                        ..., d_start:d_end, h_start:h_end, w_start:w_end
                    ]

                patches_list.append(patch)
                centers_list.append(self.patch_centers[gi, gj, gk])

            patches = torch.stack(patches_list, dim=-4)  # Stack in patch dimension
            centers = torch.stack(centers_list, dim=0)

        return patches, centers

    def get_patch_centers_subset(self, *indices: tuple) -> torch.Tensor:
        """Get patch centers for a subset of the grid."""
        return self.patch_centers[indices]

    def get_patches_at_indices(
        self, idx_gh: torch.Tensor, idx_gw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get patches at specific grid indices to match non-lazy behavior.

        Parameters
        ----------
        idx_gh: torch.Tensor
            Grid indices in height dimension
        idx_gw: torch.Tensor
            Grid indices in width dimension

        Returns
        -------
        patches: torch.Tensor
            Patches at the specified indices
        centers: torch.Tensor
            Corresponding patch centers
        """
        if self.ndim == 2:
            # Extract patches using the same logic as non-lazy version
            patches_list = []
            centers_list = []

            patch_shape_2d = cast(tuple[int, int], self.patch_shape)
            ph, pw = patch_shape_2d

            for i in range(len(idx_gh)):
                gi, gj = idx_gh[i].item(), idx_gw[i].item()
                center_h, center_w = self.patch_centers[gi, gj]

                # Calculate patch boundaries (same logic as random_subset)
                h_start = max(0, int(center_h - ph // 2))
                h_end = min(self.images.shape[-2], h_start + ph)
                h_start = max(0, h_end - ph)

                w_start = max(0, int(center_w - pw // 2))
                w_end = min(self.images.shape[-1], w_start + pw)
                w_start = max(0, w_end - pw)

                # Extract patch using slicing
                patch = self.images[..., h_start:h_end, w_start:w_end]

                # Add singleton dimension to match expected shape (t, 1, ph, pw)
                patch = patch.unsqueeze(-3)

                patches_list.append(patch)
                centers_list.append(self.patch_centers[gi, gj])

            # Stack patches and centers
            patches = torch.stack(patches_list, dim=-4)  # (t, n_patches, 1, ph, pw)
            centers = torch.stack(centers_list, dim=0)  # (n_patches, 2)

        else:  # ndim == 3
            # For 3D case - similar logic but with depth dimension
            patches_list = []
            centers_list = []

            patch_shape_3d = cast(tuple[int, int, int], self.patch_shape)
            pd, ph, pw = patch_shape_3d

            for i in range(len(idx_gh)):
                # For 3D case, we need to handle the depth index too
                # But for now, assuming we're using the 2D grid approach for time series
                gi, gj = idx_gh[i].item(), idx_gw[i].item()
                # Use first depth index (0) for time series case
                gk = 0 if len(self.grid_shape) == 3 else None

                if gk is not None:
                    center_d, center_h, center_w = self.patch_centers[gk, gi, gj]
                else:
                    center_h, center_w = self.patch_centers[gi, gj]
                    center_d = 0

                # Same logic as random_subset for 3D
                if pd == 1:
                    # Time series case
                    h_start = max(0, int(center_h - ph // 2))
                    h_end = min(self.images.shape[-2], h_start + ph)
                    h_start = max(0, h_end - ph)

                    w_start = max(0, int(center_w - pw // 2))
                    w_end = min(self.images.shape[-1], w_start + pw)
                    w_start = max(0, w_end - pw)

                    patch = self.images[..., h_start:h_end, w_start:w_end]
                    patch = patch.unsqueeze(-3)  # Add singleton dimension
                else:
                    # True 3D case
                    d_start = max(0, int(center_d - pd // 2))
                    d_end = min(self.images.shape[-3], d_start + pd)
                    d_start = max(0, d_end - pd)

                    h_start = max(0, int(center_h - ph // 2))
                    h_end = min(self.images.shape[-2], h_start + ph)
                    h_start = max(0, h_end - ph)

                    w_start = max(0, int(center_w - pw // 2))
                    w_end = min(self.images.shape[-1], w_start + pw)
                    w_start = max(0, w_end - pw)

                    patch = self.images[
                        ..., d_start:d_end, h_start:h_end, w_start:w_end
                    ]

                patches_list.append(patch)
                if gk is not None:
                    centers_list.append(self.patch_centers[gk, gi, gj])
                else:
                    centers_list.append(self.patch_centers[gi, gj])

            patches = torch.stack(patches_list, dim=-4)  # Stack in patch dimension
            centers = torch.stack(centers_list, dim=0)

        return patches, centers

    def clear_cache(self) -> None:
        """Clear the internal cache to free memory."""
        self._cache.clear()
        self._cache_keys.clear()

    def __repr__(self) -> str:
        """Return a string representation of the LazyPatchGrid."""
        return (
            f"LazyPatchGrid(shape={self.shape}, "
            f"patch_shape={self.patch_shape}, "
            f"grid_shape={self.grid_shape}, "
            f"cached_items={len(self._cache)})"
        )


def patch_grid_lazy(
    images: torch.Tensor,
    patch_shape: tuple[int, int] | tuple[int, int, int],
    patch_step: tuple[int, int] | tuple[int, int, int],
    distribute_patches: bool = True,
) -> tuple[LazyPatchGrid, torch.Tensor]:
    """
    Create a lazy patch grid that extracts patches on-demand.

    Parameters
    ----------
    images: torch.Tensor
        (..., h, w) or (..., d, h, w) array of images.
    patch_shape: tuple[int, int] | tuple[int, int, int]
        Shape of patches to extract.
    patch_step: tuple[int, int] | tuple[int, int, int]
        Step size between patch centers.
    distribute_patches: bool
        Whether to distribute patches evenly across the image.

    Returns
    -------
    lazy_patches: LazyPatchGrid
        Lazy patch grid object that computes patches on-demand.
    patch_centers: torch.Tensor
        Pre-computed patch centers.

    Examples
    --------
    >>> # Create lazy patch grid - no patches computed yet!
    >>> lazy_patches, centers = patch_grid_lazy(images, patch_shape, patch_step)
    >>> # Only compute specific patches when needed
    >>> subset_patches = lazy_patches[0:10, 0:20]  # Memory efficient!
    >>> # Perfect for optimization loops
    >>> for iteration in range(n_iterations):
    ...     patches, centers = lazy_patches.random_subset(n_patches_per_batch)
    ...     # Use patches in optimization...
    """
    lazy_grid = LazyPatchGrid(
        images=images,
        patch_shape=patch_shape,
        patch_step=patch_step,
        distribute_patches=distribute_patches,
    )

    return lazy_grid, lazy_grid.patch_centers
