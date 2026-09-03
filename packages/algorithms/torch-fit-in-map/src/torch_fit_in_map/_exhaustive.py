"""Exhaustive SO(3) search with FFT-based optimal translation per rotation."""

from __future__ import annotations

import heapq
import math

import torch
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_3d import Ry, Rz
from torch_grid_utils import coordinate_grid
from torch_image_interpolation import sample_image_3d
from torch_so3 import get_symmetry_ranges, get_uniform_euler_angles
from tqdm import tqdm

from ._config import ExhaustiveSearchConfig
from ._preprocess import _normalise_volume
from ._result import AlignmentResult


def _parse_symmetry(sym: str) -> tuple[str, int]:
    """Parse a point-group string into ``(group, order)``.

    Accepts strings like ``"C4"``, ``"D2"``, ``"T"``.

    Raises ``ValueError`` for unrecognised strings.
    """
    sym = sym.strip().upper()
    if sym in {"T", "O", "I"}:
        return sym, 1
    if len(sym) < 2 or sym[0] not in {"C", "D"}:
        raise ValueError(
            f"Unrecognised symmetry '{sym}'.  "
            "Expected 'C<n>', 'D<n>', 'T', 'O', or 'I' (e.g. 'C1', 'C4', 'D2')."
        )
    try:
        order = int(sym[1:])
    except ValueError:
        raise ValueError(
            f"Unrecognised symmetry '{sym}'.  "
            "Expected 'C<n>' or 'D<n>' where <n> is a positive integer."
        ) from None
    if order < 1:
        raise ValueError(f"Symmetry order must be >= 1, got {order}.")
    return sym[0], order


def _euler_zyz_to_4x4_zyx(
    euler_angles: torch.Tensor,
) -> torch.Tensor:
    """Convert ZYZ Euler angles to 4x4 affine matrices in zyx convention.

    Parameters
    ----------
    euler_angles : torch.Tensor
        ``(N, 3)`` angles in degrees with columns ``(phi, theta, psi)``.

    Returns
    -------
    matrices : torch.Tensor
        ``(N, 4, 4)`` rotation matrices for zyx homogeneous coordinates.
    """
    phi = euler_angles[:, 0]
    theta = euler_angles[:, 1]
    psi = euler_angles[:, 2]
    # ZYZ intrinsic: R = Rz(phi) @ Ry(theta) @ Rz(psi) in zyx convention
    return Rz(phi, zyx=True) @ Ry(theta, zyx=True) @ Rz(psi, zyx=True)


def _batch_rotate_volume(
    volume: torch.Tensor,
    centred_matrices: torch.Tensor,
    coord_grid_flat: torch.Tensor,
    vol_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Apply a batch of centred affine matrices to *volume*.

    Parameters
    ----------
    volume : torch.Tensor
        ``(d, h, w)`` input volume.
    centred_matrices : torch.Tensor
        ``(B, 4, 4)`` affine matrices in zyx convention.
    coord_grid_flat : torch.Tensor
        ``(d*h*w, 4)`` flattened homogeneous coordinate grid.
    vol_shape : tuple[int, int, int]
        ``(d, h, w)`` for reshaping output.

    Returns
    -------
    rotated : torch.Tensor
        ``(B, d, h, w)`` rotated volumes.
    """
    d, h, w = vol_shape
    # (B, 4, 4) @ (4, d*h*w) -> (B, 4, d*h*w)
    coords = torch.einsum("bij,jk->bik", centred_matrices, coord_grid_flat.T)
    coords = coords[:, :3, :].permute(0, 2, 1)  # (B, d*h*w, 3) zyx
    coords = coords.view(centred_matrices.shape[0], d, h, w, 3)  # (B, d, h, w, 3)
    return sample_image_3d(volume, coords, interpolation="trilinear")  # (B, d, h, w)


def _argmax_to_shift(
    flat_idx: torch.Tensor, shape: tuple[int, int, int]
) -> torch.Tensor:
    """Convert CC-map flat argmax index to a centred zyx shift vector."""
    idx = torch.unravel_index(flat_idx, shape)
    shifts = []
    for k in range(3):
        i = idx[k].item()
        s = shape[k]
        shifts.append(float(i if i <= s // 2 else i - s))
    return torch.tensor(shifts, dtype=torch.float32)


def _exhaustive_topk(
    reference: torch.Tensor,
    mobile: torch.Tensor,
    config: ExhaustiveSearchConfig,
    mask: torch.Tensor | None,
    verbose: bool,
) -> list[AlignmentResult]:
    """Run the exhaustive search and return the top-k results sorted best-first.

    Internal helper; use :func:`exhaustive_search` for the public API.
    """
    devices = config.devices
    if devices is None:
        devices = [str(reference.device)]

    d, h, w = reference.shape[-3:]
    n_start = config.n_start

    # Common orientation grid generation
    sym_group, sym_order = _parse_symmetry(config.symmetry)
    sym_ranges = get_symmetry_ranges(symmetry_group=sym_group, symmetry_order=sym_order)
    euler_angles = get_uniform_euler_angles(
        psi_step=config.angular_step_degrees,
        theta_step=config.angular_step_degrees,
        base_grid_method=config.angular_sampling_method,
        phi_min=sym_ranges.phi_min,
        phi_max=sym_ranges.phi_max,
        theta_min=sym_ranges.theta_min,
        theta_max=sym_ranges.theta_max,
        psi_min=sym_ranges.psi_min,
        psi_max=sym_ranges.psi_max,
    )  # Keep on CPU for splitting
    R_4x4_all = _euler_zyz_to_4x4_zyx(euler_angles)
    n_rotations = R_4x4_all.shape[0]

    centre = torch.tensor(
        [(d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0], dtype=torch.float32
    )
    R3_all = R_4x4_all[:, :3, :3]
    t_centred_all = centre.unsqueeze(0) - torch.einsum("nij,j->ni", R3_all, centre)
    M_rot_all = R_4x4_all.clone()
    M_rot_all[:, :3, 3] = t_centred_all

    from concurrent.futures import ThreadPoolExecutor

    def _worker(
        device_str: str,
        M_rot_chunk: torch.Tensor,
        R3_chunk: torch.Tensor,
        pbar_shared=None,
    ):
        dev = torch.device(device_str)
        ref_norm_dev = _normalise_volume(reference.float(), mask).to(dev)
        mob_norm_dev = _normalise_volume(mobile.float(), mask).to(dev)
        mask_dev = mask.to(dev) if mask is not None else None

        if mask_dev is not None:
            ref_norm_dev = ref_norm_dev * mask_dev
            mob_norm_dev = mob_norm_dev * mask_dev

        ref_rfft_dev = torch.fft.rfftn(ref_norm_dev, dim=(-3, -2, -1))
        grid_dev = coordinate_grid(image_shape=(d, h, w), device=dev)
        grid_flat_dev = homogenise_coordinates(grid_dev).view(-1, 4)

        M_rot_chunk = M_rot_chunk.to(dev)
        R3_chunk = R3_chunk.to(dev)

        heap: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
        _ctr = 0
        batch_size = config.rotation_batch_size

        for batch_start in range(0, M_rot_chunk.shape[0], batch_size):
            batch_M = M_rot_chunk[batch_start : batch_start + batch_size]
            batch_R3 = R3_chunk[batch_start : batch_start + batch_size]

            rotated_batch = _batch_rotate_volume(
                mob_norm_dev, batch_M, grid_flat_dev, (d, h, w)
            )
            if mask_dev is not None:
                rotated_batch = rotated_batch * mask_dev.unsqueeze(0)

            mob_rfft_batch = torch.fft.rfftn(rotated_batch, dim=(-3, -2, -1))
            cc_batch = torch.fft.irfftn(
                ref_rfft_dev.unsqueeze(0) * torch.conj(mob_rfft_batch),
                s=(d, h, w),
                dim=(-3, -2, -1),
            )

            b = cc_batch.shape[0]
            cc_flat = cc_batch.view(b, -1)
            peak_vals, peak_idxs = cc_flat.max(dim=-1)

            for i in range(b):
                score = float(peak_vals[i].item())
                if len(heap) < n_start or score > heap[0][0]:
                    t_i = _argmax_to_shift(peak_idxs[i], (d, h, w)).to(dev)
                    entry = (score, _ctr, batch_R3[i].clone(), t_i)
                    _ctr += 1
                    if len(heap) < n_start:
                        heapq.heappush(heap, entry)
                    else:
                        heapq.heapreplace(heap, entry)
            if pbar_shared is not None:
                pbar_shared.update(1)

        # Move tensors back to reference device or CPU before returning
        return [
            (s, c, r.to(reference.device), t.to(reference.device))
            for s, c, r, t in heap
        ]

    n_batches = math.ceil(n_rotations / config.rotation_batch_size)
    pbar = tqdm(
        total=n_batches,
        desc=f"Exhaustive search ({len(devices)} GPUs)"
        if len(devices) > 1
        else "Exhaustive search",
        unit="batch",
        dynamic_ncols=True,
        disable=not verbose,
    )

    # Split rotations across devices
    chunks_M = torch.chunk(M_rot_all, len(devices))
    chunks_R3 = torch.chunk(R3_all, len(devices))

    all_heaps = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [
            executor.submit(_worker, devices[i], chunks_M[i], chunks_R3[i], pbar)
            for i in range(len(devices))
        ]
        for f in futures:
            all_heaps.append(f.result())

    pbar.close()

    # Merge results from all heaps
    final_heap: list[tuple[float, int, torch.Tensor, torch.Tensor]] = []
    _ctr = 0
    for h_i in all_heaps:
        for score, _, r, t in h_i:
            entry = (score, _ctr, r, t)
            _ctr += 1
            if len(final_heap) < n_start:
                heapq.heappush(final_heap, entry)
            elif score > final_heap[0][0]:
                heapq.heapreplace(final_heap, entry)

    # Sort by score descending
    final_heap.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, _c, R3_i, t_i in final_heap:
        t_ang = (
            t_i * config.pixel_size_angstroms if config.pixel_size_angstroms else None
        )
        results.append(
            AlignmentResult(
                rotation_matrix=R3_i,
                translation_pixels=t_i,
                score=score,
                translation_angstroms=t_ang,
            )
        )
    return results


def exhaustive_search(
    reference: torch.Tensor,
    mobile: torch.Tensor,
    config: ExhaustiveSearchConfig | None = None,
    mask: torch.Tensor | None = None,
    verbose: bool = True,
) -> AlignmentResult:
    """Exhaustive SO(3) search with per-rotation FFT translation optimisation.

    For each candidate orientation the mobile volume is rotated and the
    optimal translation is found as the ``argmax`` of the 3-D normalised
    cross-correlation map, computed via a single FFT pair.  Both volumes must
    already share the same voxel size; call
    :func:`~torch_fit_in_map._preprocess.normalise_voxel_sizes` first if
    needed.

    Parameters
    ----------
    reference : torch.Tensor
        ``(d, h, w)`` reference volume.
    mobile : torch.Tensor
        ``(d, h, w)`` mobile volume (same voxel size as *reference*).
    config : ExhaustiveSearchConfig or None
        Search parameters.  Defaults are used when ``None``.
    mask : torch.Tensor or None
        Optional ``(d, h, w)`` soft mask in ``[0, 1]`` applied to both
        volumes before scoring.
    verbose : bool
        Whether to print progress during the search.

    Returns
    -------
    AlignmentResult
        Best rotation matrix (3x3, zyx), translation in pixels (3,), and NCC
        peak score.
    """
    if config is None:
        config = ExhaustiveSearchConfig()
    return _exhaustive_topk(reference, mobile, config, mask, verbose)[0]
