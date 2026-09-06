"""Rigid-body volume alignment for cryo-EM in PyTorch."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Literal

try:
    __version__ = version("torch-fit-in-map")
except PackageNotFoundError:
    __version__ = "uninstalled"

import torch
from torch_transform_image import affine_transform_image_3d
from tqdm import tqdm

from ._atoms import apply_alignment_to_structure
from ._config import (
    ExhaustiveSearchConfig,
    GradientRefinementConfig,
    PotentialSimulatorConfig,
    ProjectionAlignmentConfig,
)
from ._exhaustive import _exhaustive_topk, exhaustive_search
from ._gradient import gradient_refine
from ._preprocess import crop_or_pad_to_shape, normalise_voxel_sizes
from ._projection import projection_align
from ._result import AlignmentResult
from ._simulate import (
    DEFAULT_POTENTIAL_SIMULATOR,
    DEFAULT_SIMULATOR,
    DensitySimulator,
    PotentialSimulator,
)

if TYPE_CHECKING:
    import pandas as pd

# Sentinel for "use default gradient config" (avoids mutable default argument)
_GRADIENT_UNSET: object = object()


def fit_map_in_map(
    mobile_map: torch.Tensor,
    reference_map: torch.Tensor,
    exhaustive_config: ExhaustiveSearchConfig | None = None,
    gradient_config: GradientRefinementConfig | None | object = _GRADIENT_UNSET,
    mask: torch.Tensor | None = None,
    pixel_size_angstroms: float | None = None,
    verbose: bool = True,
) -> AlignmentResult:
    """Fit *mobile_map* into *reference_map* via exhaustive SO(3) search + refinement.

    Both volumes must share the same voxel size.  Call
    :func:`normalise_voxel_sizes` first if they differ (the
    ``torch-fit-in-map-cli`` file wrappers handle this automatically).

    Parameters
    ----------
    mobile_map : torch.Tensor
        ``(d, h, w)`` mobile volume to be fitted.
    reference_map : torch.Tensor
        ``(d, h, w)`` reference volume.
    exhaustive_config : ExhaustiveSearchConfig or None
        Parameters for the exhaustive SO(3) grid search.
    gradient_config : GradientRefinementConfig or None
        Parameters for gradient-based local refinement.  Pass ``None`` to
        return the exhaustive result without further refinement.
    mask : torch.Tensor or None
        Optional ``(d, h, w)`` soft mask in ``[0, 1]``.
    pixel_size_angstroms : float or None
        When given, ``AlignmentResult.translation_angstroms`` is populated.
    verbose : bool
        Show search and refinement progress.

    Returns
    -------
    AlignmentResult
        Rotation matrix (3 x 3, zyx), translation in pixels (3,), NCC score, and
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
        reference_map, mobile_map, config=exhaustive_config, mask=mask, verbose=verbose
    )
    result = candidates[0]

    if resolved_gradient is not None:
        if (
            resolved_gradient.pixel_size_angstroms is None
            and pixel_size_angstroms is not None
        ):
            resolved_gradient = resolved_gradient.model_copy(
                update={"pixel_size_angstroms": pixel_size_angstroms}
            )

        devices = resolved_gradient.devices
        if devices is None:
            devices = [str(reference_map.device)]

        n_start = len(candidates)

        def _refine_worker(
            i: int, candidate: AlignmentResult, device_str: str
        ) -> AlignmentResult:
            dev = torch.device(device_str)
            if verbose and len(devices) == 1:
                tqdm.write(f"Gradient refinement: start {i + 1}/{n_start}")

            ref_dev = reference_map.to(dev)
            mob_dev = mobile_map.to(dev)
            mask_dev = mask.to(dev) if mask is not None else None

            r = gradient_refine(
                ref_dev,
                mob_dev,
                initial_rotation=candidate.rotation_matrix.to(dev),
                initial_translation=candidate.translation_pixels.to(dev),
                config=resolved_gradient,
                mask=mask_dev,
                verbose=verbose and len(devices) == 1,
            )
            r.rotation_matrix = r.rotation_matrix.to(reference_map.device)
            r.translation_pixels = r.translation_pixels.to(reference_map.device)
            if r.translation_angstroms is not None:
                r.translation_angstroms = r.translation_angstroms.to(
                    reference_map.device
                )
            return r

        refined: list[AlignmentResult] = []
        if verbose and n_start > 1:
            pbar = tqdm(
                total=n_start, desc="Refining poses", unit="pose", dynamic_ncols=True
            )
        else:
            pbar = None

        if len(devices) > 1:
            if verbose:
                tqdm.write(f"Parallel gradient refinement on {len(devices)} devices...")

            with ThreadPoolExecutor(max_workers=len(devices)) as executor:
                futures = [
                    executor.submit(
                        _refine_worker, i, candidate, devices[i % len(devices)]
                    )
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
    interpolation: Literal["nearest", "trilinear"] = "trilinear",
) -> torch.Tensor:
    """Apply an :class:`AlignmentResult` transform to produce the aligned mobile volume.

    Parameters
    ----------
    mobile : torch.Tensor
        ``(d, h, w)`` mobile volume.
    result : AlignmentResult
        Alignment result from :func:`fit_map_in_map` or :func:`exhaustive_search`.
    interpolation : {"trilinear", "nearest"}
        Interpolation mode.

    Returns
    -------
    aligned : torch.Tensor
        ``(d, h, w)`` mobile volume transformed to match the reference frame.
    """
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
        mobile.float(),
        M_combined,
        interpolation=interpolation,
        zyx_matrices=True,
    )


def fit_map_in_structure(
    mobile_map: torch.Tensor,
    reference_atoms: pd.DataFrame,
    pixel_size_angstroms: float,
    box_size: int,
    simulator: PotentialSimulator | None = None,
    simulator_config: PotentialSimulatorConfig | None = None,
    save_simulated: bool = False,
    exhaustive_config: ExhaustiveSearchConfig | None = None,
    gradient_config: GradientRefinementConfig | None = None,
    mask: torch.Tensor | None = None,
    verbose: bool = True,
) -> AlignmentResult:
    """Fit *mobile_map* into the coordinate frame defined by an atomic model.

    The atoms are simulated as a potential map, which serves as the **reference**.
    The experimental map *mobile_map* is the **mobile** being fitted.
    For the inverse (fit a structure into a map), use
    :func:`fit_structure_in_map`.

    Parameters
    ----------
    mobile_map : torch.Tensor
        ``(d, h, w)`` experimental map to be fitted.
    reference_atoms : pandas.DataFrame
        Atom table (columns ``x``, ``y``, ``z``, ``element``) that serves as
        reference. Obtain one with ``mmdf.read("model.cif")``, for example.
    pixel_size_angstroms : float
        Voxel size for the simulated reference potential (Angstroms).
    box_size : int
        Cubic box size for the simulated reference potential (voxels).
    simulator : PotentialSimulator or None
        Potential simulator.  See :class:`~torch_fit_in_map.PotentialSimulator`.
        When ``None``, the default electrostatic-potential simulator is used.
    simulator_config : PotentialSimulatorConfig or None
        Options for the default simulator (scattering factors, sublattice radius,
        etc.).  Ignored when a custom ``simulator`` is supplied.
    save_simulated : bool
        Store the simulated reference potential in
        ``AlignmentResult.simulated_potential``.
    exhaustive_config : ExhaustiveSearchConfig or None
        Search parameters.
    gradient_config : GradientRefinementConfig or None
        Refinement parameters.
    mask : torch.Tensor or None
        Optional ``(d, h, w)`` soft mask.
    verbose : bool
        Show search and refinement progress.

    Returns
    -------
    AlignmentResult
    """
    if simulator is None:
        simulator = DEFAULT_POTENTIAL_SIMULATOR

    device = mobile_map.device
    simulated = simulator.simulate(
        atoms=reference_atoms,
        pixel_size=pixel_size_angstroms,
        box_size=box_size,
        device=device,
        config=simulator_config,
    )

    mobile_map, simulated, common_px = normalise_voxel_sizes(
        simulated, mobile_map, pixel_size_angstroms, pixel_size_angstroms
    )

    result = fit_map_in_map(
        mobile_map,
        simulated,
        exhaustive_config=exhaustive_config,
        gradient_config=gradient_config,
        mask=mask,
        pixel_size_angstroms=common_px,
        verbose=verbose,
    )

    if save_simulated:
        result.simulated_potential = simulated.cpu()

    return result


def fit_structure_in_map(
    mobile_atoms: pd.DataFrame,
    reference_map: torch.Tensor,
    pixel_size_angstroms: float,
    box_size: int,
    simulator: PotentialSimulator | None = None,
    simulator_config: PotentialSimulatorConfig | None = None,
    save_simulated: bool = False,
    exhaustive_config: ExhaustiveSearchConfig | None = None,
    gradient_config: GradientRefinementConfig | None = None,
    mask: torch.Tensor | None = None,
    verbose: bool = True,
) -> AlignmentResult:
    """Fit an atomic model into a map.

    The atoms are simulated as a potential map, which is the **mobile** being
    fitted into *reference_map*.  For the inverse (fit a map into a
    structure frame), use :func:`fit_map_in_structure`.

    Parameters
    ----------
    mobile_atoms : pandas.DataFrame
        Atom table (columns ``x``, ``y``, ``z``, ``element``) to be fitted.
        Obtain one with ``mmdf.read("model.cif")``, for example.
    reference_map : torch.Tensor
        ``(d, h, w)`` experimental map that serves as reference.
    pixel_size_angstroms : float
        Voxel size for the simulated potential (Angstroms).
    box_size : int
        Cubic box size for the simulated potential (voxels).
    simulator : PotentialSimulator or None
        Potential simulator.  See :class:`~torch_fit_in_map.PotentialSimulator`.
    simulator_config : PotentialSimulatorConfig or None
        Options for the default simulator.  Ignored when a custom ``simulator``
        is supplied.
    save_simulated : bool
        Store the simulated potential in ``AlignmentResult.simulated_potential``.
    exhaustive_config : ExhaustiveSearchConfig or None
        Search parameters.
    gradient_config : GradientRefinementConfig or None
        Refinement parameters.
    mask : torch.Tensor or None
        Optional ``(d, h, w)`` soft mask.
    verbose : bool
        Show search and refinement progress.

    Returns
    -------
    AlignmentResult
    """
    if simulator is None:
        simulator = DEFAULT_POTENTIAL_SIMULATOR

    device = reference_map.device
    simulated = simulator.simulate(
        atoms=mobile_atoms,
        pixel_size=pixel_size_angstroms,
        box_size=box_size,
        device=device,
        config=simulator_config,
    )

    reference_map, simulated, common_px = normalise_voxel_sizes(
        reference_map, simulated, pixel_size_angstroms, pixel_size_angstroms
    )
    simulated = crop_or_pad_to_shape(simulated, tuple(reference_map.shape[-3:]))  # type: ignore[arg-type]

    result = fit_map_in_map(
        simulated,
        reference_map,
        exhaustive_config=exhaustive_config,
        gradient_config=gradient_config,
        mask=mask,
        pixel_size_angstroms=common_px,
        verbose=verbose,
    )

    if save_simulated:
        result.simulated_potential = simulated.cpu()

    return result


__all__ = [
    "AlignmentResult",
    "DEFAULT_POTENTIAL_SIMULATOR",
    "DEFAULT_SIMULATOR",
    "DensitySimulator",
    "ExhaustiveSearchConfig",
    "GradientRefinementConfig",
    "PotentialSimulator",
    "PotentialSimulatorConfig",
    "ProjectionAlignmentConfig",
    "apply_alignment",
    "apply_alignment_to_structure",
    "crop_or_pad_to_shape",
    "exhaustive_search",
    "fit_map_in_map",
    "fit_map_in_structure",
    "fit_structure_in_map",
    "gradient_refine",
    "normalise_voxel_sizes",
    "projection_align",
]
