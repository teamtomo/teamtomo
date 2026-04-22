"""Correlation calculations for local resolution estimation.

Builds a spherical or circular window mask, a permutation null from random
window pairs and cosine similarity, then scans a stepped grid and assigns
p-values against that null (:func:`compute_local_resolution_pvalues`).
"""

import torch


def _create_bool_mask(
    window_size_float: float,
    ndim: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Create a spherical (3D) or circular (2D) mask for the local window.

    Parameters
    ----------
    window_size_float
        Radius in voxels (may be non-integer; voxels outside this distance from the
        window center are set to 0).
    ndim
        Number of spatial dimensions (2 or 3).
    device
        Device for the returned mask (same as the rest of the correlation pipeline).

    Returns
    -------
    mask : torch.Tensor
        Float tensor of shape ``(window,)*ndim`` with ``1`` inside the sphere/disk
        and ``0`` outside.
    window_size_int : int
        Integer half-width ``ceil(window_size_float)``; side length is
        ``2 * window_size_int + 1``.
    """
    ws = torch.as_tensor(window_size_float, dtype=torch.float32, device=device)
    window_size_int = int(torch.ceil(ws).item())
    window = window_size_int * 2 + 1
    center = window // 2

    axis = torch.arange(window, dtype=torch.float32, device=device)
    grids = torch.meshgrid(*[axis] * ndim, indexing="ij")
    indices = torch.stack(grids, dim=0)
    distances = torch.linalg.norm(indices - float(center), dim=0)

    mask = torch.ones((window,) * ndim, dtype=torch.float32, device=device)
    mask[distances > window_size_float] = 0.0

    return mask, window_size_int


def _window_offsets_from_mask(
    bool_mask: torch.Tensor,
    window_half_width: int,
    ndim: int,
) -> torch.Tensor:
    """Relative offsets for each mask voxel (for gather with a window center).

    Parameters
    ----------
    bool_mask
        Mask from :func:`_create_bool_mask`; non-zero entries are included.
    window_half_width
        Integer half-width of the window grid (subtract from flat indices to get
        offsets relative to the center voxel).
    ndim
        Spatial dimensionality (2 or 3).

    Returns
    -------
    torch.Tensor
        Shape ``(M, ndim)``, ``long``, on the same device as ``bool_mask``. Each row
        is an offset to add to a center index when extracting the window.
    """
    coords = torch.nonzero(bool_mask != 0, as_tuple=False)
    half = torch.tensor(
        [window_half_width] * ndim,
        dtype=torch.long,
        device=coords.device,
    )
    return coords.to(dtype=torch.long) - half


def _extract_windows_3d(
    padded_map: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Extract masked voxel values from a 3D map at multiple positions.

    Parameters
    ----------
    padded_map : torch.Tensor
        Input 3D map of shape (D, H, W).
    positions : torch.Tensor
        Center positions in padded map, shape (N, 3), dtype long.
    offsets : torch.Tensor
        Relative offsets of valid voxels, shape (M, 3), dtype long.

    Returns
    -------
    values : torch.Tensor
        Extracted values of shape (N, M).
    """
    abs_idx = positions[:, None, :] + offsets[None, :, :]  # (N, M, 3)
    return padded_map[abs_idx[..., 0], abs_idx[..., 1], abs_idx[..., 2]]


def _extract_windows_2d(
    padded_map: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Extract masked pixel values from a 2D map at multiple positions.

    Parameters
    ----------
    padded_map : torch.Tensor
        Input 2D map of shape (H, W).
    positions : torch.Tensor
        Center positions in padded map, shape (N, 2), dtype long.
    offsets : torch.Tensor
        Relative offsets of valid voxels, shape (M, 2), dtype long.

    Returns
    -------
    values : torch.Tensor
        Extracted values of shape (N, M).
    """
    abs_idx = positions[:, None, :] + offsets[None, :, :]  # (N, M, 2)
    return padded_map[abs_idx[..., 0], abs_idx[..., 1]]


def batched_cosine_similarity(vals1: torch.Tensor, vals2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between paired row vectors (e.g. masked local windows).

    For each row ``b``, computes dot product over ``i`` divided by the product of
    L2 norms of ``vals1[b]`` and ``vals2[b]``.
    Rows with zero norm in either vector yield ``0``. Means are **not** subtracted,
    so this is cosine similarity, not Pearson correlation.

    Parameters
    ----------
    vals1, vals2
        Same shape ``(B, M)``, typically values gathered at ``B`` window centers with
        ``M`` mask voxels per window.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)``, same device and dtype as inputs.
    """
    numerator = (vals1 * vals2).sum(dim=1)
    denom1 = (vals1 * vals1).sum(dim=1)
    denom2 = (vals2 * vals2).sum(dim=1)
    denominator = torch.sqrt(denom1) * torch.sqrt(denom2)
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


def build_null_distribution(
    max_radius: tuple[int, ...],
    padded_half_map1_list: list[torch.Tensor],
    padded_half_map2_list: list[torch.Tensor],
    window_offsets: torch.Tensor,
    n_random_maps: int,
    reference_dist_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build permutation null distribution of local cosine similarities.

    Randomly samples window pairs from the half-maps and computes cosine
    similarity to form a null distribution for permutation testing.
    Window shape is encoded in ``window_offsets`` (built by the caller).

    Parameters
    ----------
    max_radius
        Inclusive lower bound for valid window centers along each axis (padding).
    padded_half_map1_list
        Padded half-map 1 (list for API symmetry; index 0 is used).
    padded_half_map2_list
        One entry per random map: permuted half-map 2 variants.
    window_offsets
        Relative voxel offsets within the window, shape ``(M, ndim)``, ``long``,
        on ``device``.
    n_random_maps
        Number of random permutation maps.
    reference_dist_size
        Number of samples in the null distribution.
    device
        GPU or CPU device.

    Returns
    -------
    torch.Tensor
        1D float tensor of permuted correlation coefficients.
    """
    ndim = len(padded_half_map1_list[0].shape)
    padded_half_map1 = padded_half_map1_list[0]

    extract_fn = _extract_windows_3d if ndim == 3 else _extract_windows_2d

    # Split reference_dist_size across n_random_maps surrogates (ceil per map).
    cycle_length = (reference_dist_size + n_random_maps - 1) // n_random_maps

    permuted_cor_coeffs = torch.empty(
        reference_dist_size, dtype=torch.float32, device=device
    )

    # Independent random centers for map1 vs map2 (null: no spatial alignment).
    # After stack: (reference_dist_size, ndim) for each map.
    cols1: list[torch.Tensor] = []
    cols2: list[torch.Tensor] = []
    for d in range(ndim):
        lo = max_radius[d]
        hi = int(padded_half_map1.shape[d]) - max_radius[d] - 1
        cols1.append(
            torch.randint(
                lo,
                hi,
                (reference_dist_size,),
                device=device,
                dtype=torch.long,
            )
        )
        cols2.append(
            torch.randint(
                lo,
                hi,
                (reference_dist_size,),
                device=device,
                dtype=torch.long,
            )
        )
    all_indices1 = torch.stack(cols1, dim=1)  # (reference_dist_size, ndim)
    all_indices2 = torch.stack(cols2, dim=1)  # (reference_dist_size, ndim)

    # Each chunk uses padded_half_map2_list[map_idx] for the permuted half-map 2.
    for map_idx in range(n_random_maps):
        start = map_idx * cycle_length
        end = min(start + cycle_length, reference_dist_size)
        if start >= reference_dist_size:
            break

        padded_half_map2 = padded_half_map2_list[map_idx]

        positions1 = all_indices1[start:end]
        positions2 = all_indices2[start:end]

        # Batched window extraction: (batch, M) where M = number of mask voxels
        vals1 = extract_fn(padded_half_map1, positions1, window_offsets)
        vals2 = extract_fn(padded_half_map2, positions2, window_offsets)

        permuted_cor_coeffs[start:end] = batched_cosine_similarity(vals1, vals2)

    return permuted_cor_coeffs


def run_local_correlation_raw(
    corr_box_size: tuple[int, ...],
    max_radius: tuple[int, ...],
    padded_half_map1: torch.Tensor,
    padded_half_map2: torch.Tensor,
    step_size: tuple[int, ...],
    window_offsets: torch.Tensor,
    result_array: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> None:
    """Compute correlation at each grid position..

    Uses only the masked voxels, no p-value conversion. Result is written in-place
    to ``result_array``.  Window geometry is encoded in ``window_offsets``
    (built by the caller).

    Parameters
    ----------
    corr_box_size
        Size of the correlation box (per dimension).
    max_radius
        Half-window offset (padding) per dimension.
    padded_half_map1
        Padded half-map 1 (2D or 3D, float32).
    padded_half_map2
        Padded half-map 2 (2D or 3D, float32).
    step_size
        Step size per dimension.
    window_offsets
        Relative voxel offsets within the window, shape ``(M, ndim)``, ``long``
        (equivalent to precomputing mask indices; built from the sphere mask).
    result_array
        Output tensor (modified in-place) for correlation values.
    device
        Target device (e.g. ``torch.device('cuda:0')`` or ``torch.device('cpu')``).
    batch_size
        Number of positions to process simultaneously. Tune for GPU memory.
    """
    ndim = len(padded_half_map1.shape)

    # Window geometry comes from the caller (sphere/disk mask as offset list).

    # Stepped centers in padded-map coords; stride matches output grid (caller).
    grid_axes = [
        torch.arange(
            len(range(0, int(corr_box_size[d]), int(step_size[d]))), device=device
        )
        for d in range(ndim)
    ]
    grid_meshes = torch.meshgrid(*grid_axes, indexing="ij")
    grid = torch.stack(grid_meshes, dim=-1).reshape(-1, ndim)  # (N centerpoints, ndim)

    for d in range(ndim):
        grid[:, d] = max_radius[d] + grid[:, d] * step_size[d]

    # Flat output indices mesh over the output grid (not padded-map indices).
    result_axes = [
        torch.arange(
            len(range(0, int(corr_box_size[d]), int(step_size[d]))), device=device
        )
        for d in range(ndim)
    ]
    result_meshes = torch.meshgrid(*result_axes, indexing="ij")
    result_grid = torch.stack(result_meshes, dim=-1).reshape(-1, ndim)  # (N, ndim)

    num_positions = grid.shape[0]

    extract_fn = _extract_windows_3d if ndim == 3 else _extract_windows_2d

    # Process in batches
    fsc_all = torch.empty(num_positions, dtype=torch.float32, device=device)

    for start in range(0, num_positions, batch_size):
        end = min(start + batch_size, num_positions)
        batch_positions = grid[start:end].long()

        vals1 = extract_fn(padded_half_map1, batch_positions, window_offsets)  # (B, M)
        vals2 = extract_fn(padded_half_map2, batch_positions, window_offsets)  # (B, M)

        fsc_all[start:end] = batched_cosine_similarity(vals1, vals2)

    # --- Write results back (vectorized for common case) ---
    if ndim == 3:
        result_array[
            result_grid[:, 0].long(),
            result_grid[:, 1].long(),
            result_grid[:, 2].long(),
        ] = fsc_all
    elif ndim == 2:
        result_array[
            result_grid[:, 0].long(),
            result_grid[:, 1].long(),
        ] = fsc_all


def run_local_correlation(
    corr_box_size: tuple[int, ...],
    max_radius: tuple[int, ...],
    padded_half_map1: torch.Tensor,
    padded_half_map2: torch.Tensor,
    step_size: tuple[int, ...],
    permuted_map: torch.Tensor,
    window_offsets: torch.Tensor,
    result_array: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> None:
    """Compute correlation at each grid position and convert to p-values.

    Uses only the masked voxels, then converts to a p-value via the
    permutation distribution. Result is written in-place to ``result_array``.
    Window geometry is encoded in ``window_offsets`` (built by the caller).

    Parameters
    ----------
    corr_box_size
        Size of the correlation box (per dimension).
    max_radius
        Half-window offset (padding) per dimension.
    padded_half_map1
        Padded half-map 1 (2D or 3D, float32).
    padded_half_map2
        Padded half-map 2 (2D or 3D, float32).
    step_size
        Step size per dimension.
    permuted_map
        Precomputed local correlation (cosine) coefficients from permutation
        testing — the null / reference distribution for p-values.
    window_offsets
        Relative voxel offsets within the window, shape ``(M, ndim)``, ``long``
        (equivalent to precomputing mask indices; built from the sphere mask).
    result_array
        Output tensor (modified in-place) for p-values.
    device
        Target device (e.g. ``torch.device('cuda:0')`` or ``torch.device('cpu')``).
    batch_size
        Number of positions to process simultaneously. Tune for GPU memory.
    """
    ndim = len(padded_half_map1.shape)

    # Window geometry comes from the caller (sphere/disk mask as offset list).

    # Stepped centers in padded-map coords; stride matches output grid (caller).
    grid_axes = [
        torch.arange(
            len(range(0, int(corr_box_size[d]), int(step_size[d]))), device=device
        )
        for d in range(ndim)
    ]
    grid_meshes = torch.meshgrid(*grid_axes, indexing="ij")
    grid = torch.stack(grid_meshes, dim=-1).reshape(-1, ndim)  # (N centerpoints, ndim)

    for d in range(ndim):
        grid[:, d] = max_radius[d] + grid[:, d] * step_size[d]

    # Flat output indices mesh over the output grid (not padded-map indices).
    result_axes = [
        torch.arange(
            len(range(0, int(corr_box_size[d]), int(step_size[d]))), device=device
        )
        for d in range(ndim)
    ]
    result_meshes = torch.meshgrid(*result_axes, indexing="ij")
    result_grid = torch.stack(result_meshes, dim=-1).reshape(-1, ndim)  # (N, ndim)

    num_positions = grid.shape[0]

    extract_fn = _extract_windows_3d if ndim == 3 else _extract_windows_2d

    # Process in batches
    fsc_all = torch.empty(num_positions, dtype=torch.float32, device=device)

    for start in range(0, num_positions, batch_size):
        end = min(start + batch_size, num_positions)
        batch_positions = grid[start:end].long()

        vals1 = extract_fn(padded_half_map1, batch_positions, window_offsets)  # (B, M)
        vals2 = extract_fn(padded_half_map2, batch_positions, window_offsets)  # (B, M)

        fsc_all[start:end] = batched_cosine_similarity(vals1, vals2)

    # ECDF-style p-value: proportion of null correlations strictly larger than observed
    # (implemented via sorted null + searchsorted; see right=True convention).
    sorted_permuted, _ = torch.sort(permuted_map.to(dtype=torch.float32))
    permuted_len = float(sorted_permuted.numel())

    insert_indices = torch.searchsorted(sorted_permuted, fsc_all, right=True)
    pvalues = (permuted_len - insert_indices.float()) / permuted_len

    # --- Write results back (vectorized for common case) ---
    if ndim == 3:
        result_array[
            result_grid[:, 0].long(),
            result_grid[:, 1].long(),
            result_grid[:, 2].long(),
        ] = pvalues
    elif ndim == 2:
        result_array[
            result_grid[:, 0].long(),
            result_grid[:, 1].long(),
        ] = pvalues


def compute_local_correlation(
    sample1_filtered: torch.Tensor,
    sample2_filtered: torch.Tensor,
    window_size: float,
    corrected_box_size: tuple[int, ...],
    max_radius: tuple[int, ...],
    step_size: tuple[int, ...],
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Compute local per-resolution correlation values (cosine similarity) directly.

    Works on both GPU and CPU transparently.

    Parameters
    ----------
    sample1_filtered
        Padded half-map 1 (2D or 3D, float32).
    sample2_filtered
        Padded half-map 2 (2D or 3D, float32).
    window_size
        Window radius (float; will be ceiled for integer indexing).
    corrected_box_size
        Size of the region to process per dimension.
    max_radius
        Padding offset per dimension (allowing measurements at edges).
    step_size
        Step size per dimension.
    device
        GPU or CPU device.
    batch_size
        Number of positions to process simultaneously. Tune for GPU memory.

    Returns
    -------
    correlation_map : torch.Tensor
        Map of local correlation values form cosine similarity (``float32``).
    """
    ndim = len(sample1_filtered.shape)

    # Window geometry for this shell (radius = window_size voxels).
    bool_mask, window_half_width = _create_bool_mask(window_size, ndim, device)
    window_offsets = _window_offsets_from_mask(bool_mask, window_half_width, ndim)

    # Correlation map lives on the stepped grid inside the padded valid region.
    output_shape = tuple(
        len(
            range(
                max_radius[i],
                max_radius[i] + corrected_box_size[i],
                step_size[i],
            )
        )
        for i in range(ndim)
    )

    result_array = torch.zeros(output_shape, dtype=torch.float32, device=device)

    # At each grid point: compute cosine similarity
    run_local_correlation_raw(
        corrected_box_size,
        max_radius,
        sample1_filtered,
        sample2_filtered,
        step_size,
        window_offsets,
        result_array,
        device,
        batch_size,
    )

    return result_array


def compute_local_resolution_pvalues(
    sample1_filtered: torch.Tensor,
    sample2_filtered: torch.Tensor,
    permutated_sample2_filtered: list[torch.Tensor],
    permutated_sample1_filtered: list[torch.Tensor],
    window_size: float,
    corrected_box_size: tuple[int, ...],
    max_radius: tuple[int, ...],
    step_size: tuple[int, ...],
    n_random_maps: int,
    reference_dist_size: int,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Compute local resolution p-values via permutation testing.

    Works on both GPU and CPU transparently.

    Parameters
    ----------
    sample1_filtered
        Padded half-map 1 (2D or 3D, float32).
    sample2_filtered
        Padded half-map 2 (2D or 3D, float32).
    permutated_sample2_filtered
        Permuted half-map 2 variants for building the null distribution.
    permutated_sample1_filtered
        Permuted half-map 1 variants for building the null distribution.
    window_size
        Window radius (float; will be ceiled for integer indexing).
    corrected_box_size
        Size of the region to process per dimension.
    max_radius
        Padding offset per dimension (allowing measurements at edges).
    step_size
        Step size per dimension.
    n_random_maps
        Number of random map iterations for permutation testing.
    reference_dist_size
        Size of the reference distribution.
    device
        GPU or CPU device.
    batch_size
        Number of positions to process simultaneously. Tune for GPU memory.

    Returns
    -------
    pValueMap : torch.Tensor
        Map of p-values from the permutation test (``float16``).
    """
    ndim = len(sample1_filtered.shape)

    # Window geometry for this shell (radius = window_size voxels).
    bool_mask, window_half_width = _create_bool_mask(window_size, ndim, device)
    window_offsets = _window_offsets_from_mask(bool_mask, window_half_width, ndim)

    # Null distribution: cosine similarity of random windows (map1 vs permuted map2).
    permuted_map = build_null_distribution(
        max_radius,
        permutated_sample1_filtered,
        permutated_sample2_filtered,
        window_offsets,
        n_random_maps,
        reference_dist_size,
        device,
    )

    # P-value map lives on the stepped grid inside the padded valid region.
    output_shape = tuple(
        len(
            range(
                max_radius[i],
                max_radius[i] + corrected_box_size[i],
                step_size[i],
            )
        )
        for i in range(ndim)
    )

    # Default 1.0; run_local_correlation overwrites interior with p-values.
    result_array = torch.ones(output_shape, dtype=torch.float32, device=device)

    # At each grid point: observed cosine similarity vs sorted null -> p-value.
    run_local_correlation(
        corrected_box_size,
        max_radius,
        sample1_filtered,
        sample2_filtered,
        step_size,
        permuted_map,
        window_offsets,
        result_array,
        device,
        batch_size,
    )

    return result_array.to(dtype=torch.float16)
