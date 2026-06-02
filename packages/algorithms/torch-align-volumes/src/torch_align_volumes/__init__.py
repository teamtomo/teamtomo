"""Rigid-body volume alignment for cryo-EM in PyTorch."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-align-volumes")
except PackageNotFoundError:
    __version__ = "uninstalled"

import torch

from ._config import (
    ExhaustiveSearchConfig,
    GradientRefinementConfig,
    ProjectionAlignmentConfig,
)
from ._exhaustive import _exhaustive_topk, exhaustive_search
from ._preprocess import crop_or_pad_to_shape
from ._gradient import gradient_refine
from ._io import align_map_to_pdb_from_files, align_volumes_from_files
from ._projection import projection_align
from ._result import AlignmentResult
from ._simulate import DEFAULT_SIMULATOR, DensitySimulator

# Sentinel for "use default gradient config" (avoids mutable default argument)
_GRADIENT_UNSET: object = object()


def align_volumes(
    reference: torch.Tensor,
    mobile: torch.Tensor,
    exhaustive_config: ExhaustiveSearchConfig | None = None,
    gradient_config: GradientRefinementConfig | None | object = _GRADIENT_UNSET,
    mask: torch.Tensor | None = None,
    pixel_size_angstroms: float | None = None,
    verbose: bool = True,
) -> AlignmentResult:
    """Align *mobile* onto *reference* via exhaustive SO(3) search + refinement.

    Both volumes must share the same voxel size.  Call
    :func:`~torch_align_volumes._preprocess.normalise_voxel_sizes` first if
    they differ, or use :func:`align_volumes_from_files` which handles this
    automatically.

    Parameters
    ----------
    reference : torch.Tensor
        ``(d, h, w)`` reference volume.
    mobile : torch.Tensor
        ``(d, h, w)`` mobile volume.
    exhaustive_config : ExhaustiveSearchConfig or None
        Parameters for the exhaustive SO(3) grid search.
    gradient_config : GradientRefinementConfig or None
        Parameters for gradient-based local refinement.  Pass ``None`` to
        return the exhaustive result without further refinement.
    mask : torch.Tensor or None
        Optional ``(d, h, w)`` soft mask in ``[0, 1]``.
    pixel_size_angstroms : float or None
        When given, ``AlignmentResult.translation_angstroms`` is populated.

    Returns
    -------
    AlignmentResult
        Rotation matrix (3×3, zyx), translation in pixels (3,), NCC score, and
        optionally translation in Angstroms.
    """
    if exhaustive_config is None:
        exhaustive_config = ExhaustiveSearchConfig(
            pixel_size_angstroms=pixel_size_angstroms
        )

    # Resolve sentinel: default is to run gradient refinement
    resolved_gradient: GradientRefinementConfig | None
    if gradient_config is _GRADIENT_UNSET:
        resolved_gradient = GradientRefinementConfig()
    else:
        resolved_gradient = gradient_config  # type: ignore[assignment]

    candidates = _exhaustive_topk(
        reference, mobile, config=exhaustive_config, mask=mask, verbose=verbose
    )
    result = candidates[0]

    if resolved_gradient is not None:
        if resolved_gradient.pixel_size_angstroms is None and pixel_size_angstroms is not None:
            resolved_gradient = resolved_gradient.model_copy(
                update={"pixel_size_angstroms": pixel_size_angstroms}
            )

        devices = resolved_gradient.devices
        if devices is None:
            devices = [str(reference.device)]

        n_start = len(candidates)
        from concurrent.futures import ThreadPoolExecutor

        def _refine_worker(i, candidate, device_str):
            dev = torch.device(device_str)
            if verbose and len(devices) == 1:
                from tqdm import tqdm as _tqdm
                _tqdm.write(f"Gradient refinement: start {i + 1}/{n_start}")

            # Ensure volumes are on the target device for this worker
            ref_dev = reference.to(dev)
            mob_dev = mobile.to(dev)
            mask_dev = mask.to(dev) if mask is not None else None

            r = gradient_refine(
                ref_dev,
                mob_dev,
                initial_rotation=candidate.rotation_matrix.to(dev),
                initial_translation=candidate.translation_pixels.to(dev),
                config=resolved_gradient,
                mask=mask_dev,
                verbose=verbose and len(devices) == 1,  # Only show pbar if sequential
            )
            # Move result back to reference device
            r.rotation_matrix = r.rotation_matrix.to(reference.device)
            r.translation_pixels = r.translation_pixels.to(reference.device)
            if r.translation_angstroms is not None:
                r.translation_angstroms = r.translation_angstroms.to(reference.device)
            return r

        refined: list = []
        if verbose and n_start > 1:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=n_start, desc="Refining poses", unit="pose", dynamic_ncols=True)
        else:
            pbar = None

        if len(devices) > 1:
            if verbose:
                from tqdm import tqdm as _tqdm
                _tqdm.write(f"Parallel gradient refinement on {len(devices)} devices...")

            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                futures = [
                    executor.submit(_refine_worker, i, candidate, devices[i % len(devices)])
                    for i, candidate in enumerate(candidates)
                ]
                for f in futures:
                    refined.append(f.result())
                    if pbar is not None:
                        pbar.update(1)
        else:
            for i, candidate in enumerate(candidates):
                refined.append(_refine_worker(i, candidate, devices[0]))
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        result = max(refined, key=lambda r: r.score)

    if pixel_size_angstroms is not None and result.translation_angstroms is None:
        result.translation_angstroms = result.translation_pixels * pixel_size_angstroms

    return result


def apply_alignment(
    mobile: torch.Tensor,
    result: AlignmentResult,
    interpolation: str = "trilinear",
) -> torch.Tensor:
    """Apply an :class:`AlignmentResult` transform to produce the aligned mobile volume.

    Parameters
    ----------
    mobile : torch.Tensor
        ``(d, h, w)`` mobile volume.
    result : AlignmentResult
        Alignment result from :func:`align_volumes` or :func:`exhaustive_search`.
    interpolation : {"trilinear", "nearest"}
        Interpolation mode.

    Returns
    -------
    aligned : torch.Tensor
        ``(d, h, w)`` mobile volume transformed to match the reference frame.
    """
    from torch_transform_image import affine_transform_image_3d

    device = mobile.device
    d, h, w = mobile.shape[-3:]
    R_3x3 = result.rotation_matrix.to(device=device, dtype=torch.float32)
    t = result.translation_pixels.to(device=device, dtype=torch.float32)

    centre = torch.tensor(
        [(d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0],
        dtype=torch.float32,
        device=device,
    )
    # Centred rotation 4x4: [R, c - R@c; 0, 1]
    M_rot = torch.eye(4, dtype=torch.float32, device=device)
    M_rot[:3, :3] = R_3x3
    M_rot[:3, 3] = centre - R_3x3 @ centre

    # Translation 4x4: [I, -t; 0, 1]
    T_t = torch.eye(4, dtype=torch.float32, device=device)
    T_t[:3, 3] = -t

    # Combined: rotate around centre, then shift by t
    M_combined = M_rot @ T_t
    return affine_transform_image_3d(
        mobile.float(), M_combined, interpolation=interpolation, zyx_matrices=True  # type: ignore[arg-type]
    )


def align_map_to_pdb(
    density_map: torch.Tensor,
    pdb_path: str,
    pixel_size_angstroms: float,
    box_size: int,
    simulator: DensitySimulator | None = None,
    save_simulated: bool = False,
    exhaustive_config: ExhaustiveSearchConfig | None = None,
    gradient_config: GradientRefinementConfig | None = None,
    mask: torch.Tensor | None = None,
) -> AlignmentResult:
    """Simulate a density from a PDB and align it to *density_map*.

    Parameters
    ----------
    density_map : torch.Tensor
        ``(d, h, w)`` experimental density map.
    pdb_path : str
        Path to the atomic model (PDB or mmCIF).
    pixel_size_angstroms : float
        Voxel size for the simulated density (Angstroms).
    box_size : int
        Cubic box size for the simulated density (voxels).
    simulator : DensitySimulator or None
        Density simulator.  See :class:`~torch_align_volumes.DensitySimulator`.
        When ``None``, a placeholder is used that raises ``NotImplementedError``
        with guidance until ``torch-calculate-electrostatic-potential`` is
        available.
    save_simulated : bool
        Store the simulated density in ``AlignmentResult.simulated_volume``.
    exhaustive_config : ExhaustiveSearchConfig or None
        Search parameters.
    gradient_config : GradientRefinementConfig or None
        Refinement parameters.
    mask : torch.Tensor or None
        Optional ``(d, h, w)`` soft mask.

    Returns
    -------
    AlignmentResult
    """
    from pathlib import Path

    from ._preprocess import normalise_voxel_sizes

    if simulator is None:
        simulator = DEFAULT_SIMULATOR

    device = density_map.device
    simulated = simulator.simulate(
        pdb_path=Path(pdb_path),
        pixel_size=pixel_size_angstroms,
        box_size=box_size,
        device=device,
    )

    density_map, simulated, common_px = normalise_voxel_sizes(
        density_map, simulated, pixel_size_angstroms, pixel_size_angstroms
    )

    result = align_volumes(
        density_map,
        simulated,
        exhaustive_config=exhaustive_config,
        gradient_config=gradient_config,
        mask=mask,
        pixel_size_angstroms=common_px,
    )

    if save_simulated:
        result.simulated_volume = simulated.cpu()

    return result


__all__ = [
    "AlignmentResult",
    "DensitySimulator",
    "ExhaustiveSearchConfig",
    "GradientRefinementConfig",
    "ProjectionAlignmentConfig",
    "align_map_to_pdb",
    "align_map_to_pdb_from_files",
    "align_volumes",
    "align_volumes_from_files",
    "apply_alignment",
    "crop_or_pad_to_shape",
    "exhaustive_search",
    "gradient_refine",
    "projection_align",
]
