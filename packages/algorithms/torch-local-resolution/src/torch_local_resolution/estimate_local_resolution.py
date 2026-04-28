"""Local resolution estimation: validated public API and core routine.

:func:`compute_resolution` sets up geometry, prepares rFFTs and permutation
surrogates, applies per-shell bandpasses, and fills per-batch p-value maps via
:mod:`utils_correlations`.
"""

from typing import cast

import numpy as np
import torch
from torch_fourier_filter.bandpass import bandpass_filter_hyptan
from torch_fourier_filter.phase_randomize import phase_permutation

from . import utils, utils_correlations
from .input_models import ComputeResolutionInput


def _bandpass_halfmaps(
    fft_pairs: list[
        tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]
    ],
    bandpass_filter: torch.Tensor,
    b: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    """Apply bandpass and invert both half-maps for a single batch element."""
    fft1, fft2, padded_image_shape_b, fft_crop_b = fft_pairs[b]

    sample1_filtered = utils.apply_bandpass_and_invert(
        fft1, bandpass_filter, padded_image_shape_b, fft_crop_b
    )
    sample2_filtered = utils.apply_bandpass_and_invert(
        fft2, bandpass_filter, padded_image_shape_b, fft_crop_b
    )

    return (sample1_filtered, sample2_filtered, padded_image_shape_b, fft_crop_b)


def _prepare_fft_pairs(
    batch_half_map1: torch.Tensor,
    batch_half_map2: torch.Tensor,
    n_batch: int,
    pad: int,
    device: torch.device,
    n_random_maps: int = 0,
    do_phase_permutation: bool = False,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]],
    list[list[torch.Tensor]],
]:
    """Compute FFT pairs for each batch element and, optionally, permutation surrogates."""
    fft_pairs = []
    permutation_maps_fft_all = []

    for b in range(n_batch):
        fft1, fft2, padded_image_shape, fft_crop = utils.prepare_halfmaps_for_fft(
            utils.spatial_map_bzyx(batch_half_map1, b),
            utils.spatial_map_bzyx(batch_half_map2, b),
            pad=pad,
            device=device,
        )
        fft_pairs.append((fft1, fft2, padded_image_shape, fft_crop))

        if n_random_maps <= 0:
            continue

        perm_ffts = []
        for _ in range(n_random_maps):
            # Surrogate half-map 2: phase_permutation on fft2, or real-space shuffle
            # on the padded grid then rfftn.
            if do_phase_permutation:
                perm_ffts.append(
                    phase_permutation(
                        fft2,
                        image_shape=padded_image_shape,
                        rfft=True,
                        cuton=0,
                        fftshift=False,
                        device=device,
                    )
                )
            else:
                vol_b = utils.spatial_map_bzyx(batch_half_map2, b)
                t_flat = vol_b.flatten().float().to(device)
                idx = (
                    torch.randperm(int(np.prod(padded_image_shape)), device=device)
                    % t_flat.numel()
                )
                permutation_map = t_flat[idx].reshape(padded_image_shape)
                fft3 = torch.fft.rfftn(
                    permutation_map, dim=list(range(len(padded_image_shape)))
                )
                perm_ffts.append(fft3)
        permutation_maps_fft_all.append(perm_ffts)

    return fft_pairs, permutation_maps_fft_all


def _prepare_device_and_geometry(
    inp: ComputeResolutionInput,
) -> tuple[
    torch.device,
    torch.Tensor,
    torch.Tensor,
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
]:
    """Move half-maps to device and compute shared shell / output grid geometry."""
    device: torch.device = cast("torch.device", inp.device)
    batch_half_map1 = inp.batch_half_map1.to(device, non_blocking=device.type == "cuda")
    batch_half_map2 = inp.batch_half_map2.to(device, non_blocking=device.type == "cuda")

    # Spatial dimensions from (B,Z,Y,X) and batch size.
    n_batch = batch_half_map1.shape[0]
    ref_shape = utils.spatial_shape_from_bzyx(batch_half_map1)

    # Geometry shared by every resolution shell (same padding and output grid).
    # pad: voxels per side so the largest spherical window fits in the grid.
    # max_radius: first valid center index is pad (windows stay inside the box).
    # output_shape: correlation on a strided grid (step_size), not every voxel.
    pad = int(np.ceil(np.max(inp.windows_radii)))
    corrected_box_size = ref_shape
    dim = len(ref_shape)
    max_radius = tuple(pad for _ in range(dim))
    step_size_dim = tuple(inp.step_size for _ in range(dim))
    output_shape = tuple(len(range(0, s, inp.step_size)) for s in ref_shape)

    return (
        device,
        batch_half_map1,
        batch_half_map2,
        n_batch,
        corrected_box_size,
        max_radius,
        step_size_dim,
        output_shape,
        pad,
    )


def compute_correlation(inp: ComputeResolutionInput) -> torch.Tensor:
    """Compute per-shell correlation from a validated :class:`ComputeResolutionInput`.

    Parameters
    ----------
    inp : ComputeResolutionInput
        Validated half-maps and hyperparameters. Build via
        :class:`~toch_local_resolution.input_models.ComputeResolutionInput` or call
        :func:`estimate_local_resolution`.

    Returns
    -------
    loc_corr_map : torch.Tensor
        Map with correlation values per batch element, per sampling location, per
        investigated shell, in the order of given input shells. Shape is
        (n_batch, len(windows_radii), *output_shape). Map size will be reduced
        compared to input map according to ``step_size``.
    """
    (
        device,
        batch_half_map1,
        batch_half_map2,
        n_batch,
        corrected_box_size,
        max_radius,
        step_size_dim,
        output_shape,
        pad,
    ) = _prepare_device_and_geometry(inp)

    # Per batch element: rFFT both half-maps and build Fourier-domain surrogates.
    fft_pairs, _ = _prepare_fft_pairs(
        batch_half_map1,
        batch_half_map2,
        n_batch,
        pad,
        device,
    )

    # Padded FFT grid shape is the same for every b when input shapes match.
    _, _, padded_image_shape, _ = fft_pairs[0]

    # Target resolutions (Angstrom) -> spatial frequency -> shell [low, high] in 1/A.
    # Bandpass below expects cycles/pixel; shell edges are scaled by apix there.
    spatial_frequencies = 1 / np.array(inp.resolutions)
    shells = utils.calculate_shells(inp.apix, spatial_frequencies, inp.shell_size)

    # Accumulator: one correlation value map per batch row and per shell
    # on the stepped grid.
    loc_corr_map = torch.zeros(
        (n_batch, len(inp.windows_radii), *output_shape),
        device=device,
        dtype=torch.float32,
    )

    # Outer loop: each iteration is one resolution shell and matching window radius.
    for index_i, i in enumerate(inp.windows_radii):
        window_size = i

        # One tangential-hyperbolic bandpass for this shell; shared across batch.
        low_cpp = float(shells[index_i][0] * inp.apix)
        high_cpp = float(shells[index_i][1] * inp.apix)
        bandpass_filter = bandpass_filter_hyptan(
            low=low_cpp,
            high=high_cpp,
            falloff=inp.falloff,
            image_shape=padded_image_shape,
            rfft=True,
            fftshift=False,
            device=device,
        )

        for b in range(n_batch):
            # Band-limit real half-maps and each surrogate; crop off FFT zero-pad.
            sample1_filtered, sample2_filtered, _, _ = _bandpass_halfmaps(
                fft_pairs, bandpass_filter, b
            )

            # Local window mask, permutation null, grid scan -> correlation values.
            loc_corr_map[b, index_i] = utils_correlations.compute_local_correlation(
                sample1_filtered,
                sample2_filtered,
                window_size,
                corrected_box_size,
                max_radius,
                step_size_dim,
                device,
                inp.batch_size,
            )

    return loc_corr_map


def compute_resolution(inp: ComputeResolutionInput) -> torch.Tensor:
    """Compute per-shell p-value map from a validated :class:`ComputeResolutionInput`.

    Parameters
    ----------
    inp : ComputeResolutionInput
        Validated half-maps and hyperparameters. Build via
        :class:`~toch_local_resolution.input_models.ComputeResolutionInput` or call
        :func:`estimate_local_resolution`.

    Returns
    -------
    loc_res_map : torch.Tensor
        Map with p-values per batch element, per sampling location, per investigated
        shell, in the order of given input shells. Shape is
        (n_batch, len(windows_radii), *output_shape). Map size will be reduced
        compared to input map according to ``step_size``.
    """
    (
        device,
        batch_half_map1,
        batch_half_map2,
        n_batch,
        corrected_box_size,
        max_radius,
        step_size_dim,
        output_shape,
        pad,
    ) = _prepare_device_and_geometry(inp)

    # For default n_random_maps = 0: Estimate required number of random maps
    if inp.n_random_maps == 0:
        inp.n_random_maps = utils.estimate_random_maps(
            inp.reference_dist_size,
            np.max(inp.windows_radii),
            utils.spatial_map_bzyx(batch_half_map1, 0).shape,
        )

    # Per batch element: rFFT both half-maps and build Fourier-domain surrogates.
    fft_pairs, permutation_maps_fft_all = _prepare_fft_pairs(
        batch_half_map1,
        batch_half_map2,
        n_batch,
        pad,
        device,
        n_random_maps=inp.n_random_maps,
        do_phase_permutation=inp.do_phase_permutation,
    )

    # Padded FFT grid shape is the same for every b when input shapes match.
    _, _, padded_image_shape, _ = fft_pairs[0]

    # Target resolutions (Angstrom) -> spatial frequency -> shell [low, high] in 1/A.
    # Bandpass below expects cycles/pixel; shell edges are scaled by apix there.
    spatial_frequencies = 1 / np.array(inp.resolutions)
    shells = utils.calculate_shells(inp.apix, spatial_frequencies, inp.shell_size)

    # Accumulator: one p-value map per batch row and per shell on the stepped grid.
    loc_res_map = torch.zeros(
        (n_batch, len(inp.windows_radii), *output_shape),
        device=device,
        dtype=torch.float16,
    )

    # Outer loop: each iteration is one resolution shell and matching window radius.
    for index_i, i in enumerate(inp.windows_radii):
        window_size = i

        # One tangential-hyperbolic bandpass for this shell; shared across batch.
        low_cpp = float(shells[index_i][0] * inp.apix)
        high_cpp = float(shells[index_i][1] * inp.apix)
        bandpass_filter = bandpass_filter_hyptan(
            low=low_cpp,
            high=high_cpp,
            falloff=inp.falloff,
            image_shape=padded_image_shape,
            rfft=True,
            fftshift=False,
            device=device,
        )

        for b in range(n_batch):
            # Band-limit real half-maps and each surrogate; crop off FFT zero-pad.
            sample1_filtered, sample2_filtered, padded_image_shape_b, fft_crop_b = (
                _bandpass_halfmaps(fft_pairs, bandpass_filter, b)
            )

            permutated_sample2_filtered = []
            for ind_rand in range(inp.n_random_maps):
                permutated_sample2_filtered.append(
                    utils.apply_bandpass_and_invert(
                        permutation_maps_fft_all[b][ind_rand],
                        bandpass_filter,
                        padded_image_shape_b,
                        fft_crop_b,
                    )
                )
            # API wants a list for permuted map 1; only entry is unpermuted reference.
            permutated_sample1_filtered = [sample1_filtered]

            # Local window mask, permutation null, grid scan -> p-values (float16).
            loc_res_map[b, index_i] = (
                utils_correlations.compute_local_resolution_pvalues(
                    sample1_filtered,
                    sample2_filtered,
                    permutated_sample2_filtered,
                    permutated_sample1_filtered,
                    window_size,
                    corrected_box_size,
                    max_radius,
                    step_size_dim,
                    inp.n_random_maps,
                    inp.reference_dist_size,
                    device,
                    inp.batch_size,
                )
            )

    return loc_res_map


def estimate_local_resolution(
    apix: float,
    windows_radii: list[float],
    resolutions: list[float],
    batch_half_map1: torch.Tensor,
    batch_half_map2: torch.Tensor,
    step_size: int = 3,
    gpu_id: int | None = None,
    skip_statistics: bool = False,
    n_random_maps: int = 0,
    reference_dist_size: int = 10000,
    do_phase_permutation: bool = True,
    batch_size: int = 4096,
    shell_size: float = 0.05,
    falloff: float = 1.5,
) -> torch.Tensor:
    """Compute per-shell p-value map by local correlation, no resolution thresholding.

    Arguments are validated with :class:`ComputeResolutionInput` then passed to
    :func:`compute_resolution`. The compute device is ``cpu`` when ``gpu_id`` is
    ``None``, otherwise ``cuda:{gpu_id}`` (CUDA must be available if ``gpu_id`` is
    set).

    Parameters
    ----------
    apix
        Voxel size in Å per pixel.
    windows_radii
        Window radii in voxels, one entry per shell; paired with ``resolutions``.
    resolutions
        Target resolutions in Å, one per ``windows_radii`` entry.
    batch_half_map1, batch_half_map2
        Half-maps with shape ``(B, Z, Y, X)``. Use ``Z = 1`` for 2D (single slice).
        Both tensors must match in shape; they are moved to the compute device
        inside :func:`compute_resolution`.
    step_size
        Stride in voxels between correlation samples on the output grid.
    gpu_id
        If ``None``, run on CPU. If an int, use that CUDA device index (must exist).
    skip_statistics
        Return correlation values directly, skipping p-value determination entirely.
        The correlation values will not be adequate for resolution estimation and
        should not be compared across map sizes and datasets. The benefit is
        differentiability.
    n_random_maps
        Number of random surrogate / permuted maps for the null distribution.
    reference_dist_size
        Number of random window pairs used to build the permutation null.
    do_phase_permutation
        If ``True``, build surrogates by permuting Fourier phases of half-map 2
        (``torch_fourier_filter.phase_randomize.phase_permutation``); if ``False``,
        permute real-space voxels before FFT.
    batch_size
        How many grid positions to process together in the local correlation scan
        (GPU memory vs throughput).
    shell_size
        Relative shell width for bandpass (passed to shell construction in 1/Å).
    falloff
        Hypertangent bandpass falloff (``torch_fourier_filter``).

    Returns
    -------
    torch.Tensor
        P-value map per batch element and shell. Shape
        ``(n_batch, len(windows_radii), *output_shape)``, ``float16``. Spatial size
        is reduced versus the input when ``step_size > 1``.
    """
    # Pydantic validates all arguments before any heavy compute runs.
    inp = ComputeResolutionInput(
        apix=apix,
        windows_radii=windows_radii,
        resolutions=resolutions,
        batch_half_map1=batch_half_map1,
        batch_half_map2=batch_half_map2,
        step_size=step_size,
        gpu_id=gpu_id,
        n_random_maps=n_random_maps,
        reference_dist_size=reference_dist_size,
        do_phase_permutation=do_phase_permutation,
        batch_size=batch_size,
        shell_size=shell_size,
        falloff=falloff,
    )

    if skip_statistics:
        return compute_correlation(inp)
    return compute_resolution(inp)
