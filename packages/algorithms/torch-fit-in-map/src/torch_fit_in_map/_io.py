"""File-path convenience wrappers for MRC and PDB inputs."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from ._config import ExhaustiveSearchConfig, GradientRefinementConfig
from ._preprocess import crop_or_pad_to_shape, normalise_voxel_sizes
from ._result import AlignmentResult
from ._simulate import DEFAULT_SIMULATOR, DensitySimulator


def _load_mrc(path: str | os.PathLike[str]) -> tuple[torch.Tensor, float]:
    """Load an MRC file and return ``(data_tensor, pixel_size_angstroms)``."""
    import mrcfile  # type: ignore[import]

    with mrcfile.open(str(path), mode="r") as mrc:
        data = torch.from_numpy(mrc.data.copy()).float()
        px = float(mrc.voxel_size.x)
    if px == 0.0:
        px = 1.0  # fallback if header has no pixel size
    return data, px


def _read_mrc_header(
    path: str | os.PathLike[str],
) -> tuple[tuple[int, int, int], float, tuple[float, float, float]]:
    """Return ``(shape_dhw, pixel_size_angstroms, origin_xyz_angstroms)`` from an MRC header.

    Origin is read from the MRC2014 ``origin`` field.  If that field is all
    zeros, the older ``nxstart/nystart/nzstart`` convention is used as a
    fallback (``n*start * voxel_size``).
    """
    import mrcfile  # type: ignore[import]

    with mrcfile.open(str(path), mode="r") as mrc:
        shape: tuple[int, int, int] = tuple(mrc.data.shape)  # type: ignore[assignment]
        px = float(mrc.voxel_size.x) or 1.0
        ox = float(mrc.header.origin.x)
        oy = float(mrc.header.origin.y)
        oz = float(mrc.header.origin.z)
        if ox == 0.0 and oy == 0.0 and oz == 0.0:
            ox = float(mrc.header.nxstart) * px
            oy = float(mrc.header.nystart) * px
            oz = float(mrc.header.nzstart) * px
    return shape, px, (ox, oy, oz)


def _save_mrc(
    path: str | os.PathLike[str],
    data: torch.Tensor,
    pixel_size: float = 1.0,
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Save a float32 tensor as an MRC file.

    Parameters
    ----------
    path : str or Path
        Output path.
    data : torch.Tensor
        ``(d, h, w)`` volume in ZYX order.
    pixel_size : float
        Voxel size in Angstroms.
    origin_xyz : tuple[float, float, float]
        XYZ origin of the map in Angstroms (position of voxel [0,0,0] in world space).
    """
    import mrcfile  # type: ignore[import]
    import numpy as np

    arr = data.cpu().numpy().astype(np.float32)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(arr)
        mrc.voxel_size = pixel_size
        mrc.header.origin.x = origin_xyz[0]
        mrc.header.origin.y = origin_xyz[1]
        mrc.header.origin.z = origin_xyz[2]


def _pdb_centroid_xyz(path: str | os.PathLike[str]) -> tuple[float, float, float]:
    """Return the geometric centroid ``(x, y, z)`` in Angstroms of all atoms in a PDB/mmCIF."""
    import gemmi  # type: ignore[import]

    structure = gemmi.read_structure(str(path))
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    xs.append(atom.pos.x)
                    ys.append(atom.pos.y)
                    zs.append(atom.pos.z)
    if not xs:
        raise ValueError(f"No atoms found in {path}")
    n = len(xs)
    return (sum(xs) / n, sum(ys) / n, sum(zs) / n)


def transform_atomic_model(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    rotation_matrix_zyx: torch.Tensor,
    translation_pixels_zyx: torch.Tensor,
    pixel_size: float,
    box_shape: tuple[int, int, int],
    sim_box_size: int | None = None,
    ref_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Apply an alignment transform to an atomic model and write the result.

    The coordinate pipeline mirrors what the simulator and ``crop_or_pad_to_shape``
    did, then applies the alignment and converts back to Angstroms:

    1. Atom Å → simulation voxels:
       ``p_sim = (atom_Å_zyx − centroid_Å_zyx + box_centre_sim_Å) / pixel_size``
    2. Simulation voxels → cropped-box voxels (accounts for ``crop_or_pad_to_shape``):
       ``p_mob = p_sim − crop_start_zyx``
    3. Apply alignment (rotation around box centre, then translate):
       ``p_ref = R⁻¹ @ (p_mob − c) + c + t``
    4. Reference voxels → Å (adds MRC origin):
       ``atom_ref_Å = p_ref × pixel_size + origin_Å``

    Parameters
    ----------
    input_path : str or Path
        Input atomic model (PDB or mmCIF).
    output_path : str or Path
        Destination path.  Extension controls format (``.pdb`` or ``.cif``).
    rotation_matrix_zyx : torch.Tensor
        ``(3, 3)`` rotation matrix in zyx convention from :class:`AlignmentResult`.
    translation_pixels_zyx : torch.Tensor
        ``(3,)`` translation in zyx pixels from :class:`AlignmentResult`.
    pixel_size : float
        Voxel size of the reference map in Angstroms.
    box_shape : tuple[int, int, int]
        ``(d, h, w)`` shape of the reference map.
    sim_box_size : int or None
        Cubic box size used during simulation.  Defaults to ``max(box_shape)``.
    ref_origin_xyz : tuple[float, float, float]
        XYZ origin of the reference map in Angstroms (from MRC header).
    """
    import gemmi  # type: ignore[import]
    import numpy as np

    d, h, w = box_shape
    if sim_box_size is None:
        sim_box_size = max(d, h, w)

    # Rotation centre in reference (cropped) voxel space
    c_ref_zyx = np.array([(d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0])

    # Simulation box centre in Å (atoms were centred here during simulation)
    box_centre_sim_A = (sim_box_size - 1) / 2.0 * pixel_size

    # Crop offsets: simulation was cubic (sim_box_size³), then cropped to box_shape
    crop_start_zyx = np.array([
        max(0, (sim_box_size - d) // 2),
        max(0, (sim_box_size - h) // 2),
        max(0, (sim_box_size - w) // 2),
    ], dtype=float)

    # Reference map origin in ZYX order
    origin_zyx = np.array([ref_origin_xyz[2], ref_origin_xyz[1], ref_origin_xyz[0]])

    R_inv = rotation_matrix_zyx.cpu().numpy().T  # (3, 3), R orthogonal → R⁻¹ = Rᵀ
    t = translation_pixels_zyx.cpu().numpy()     # (3,) zyx

    # Compute centroid over all atoms (same centroid used in _ESPSimulator)
    structure = gemmi.read_structure(str(input_path))
    all_coords_zyx = np.array([
        [a.pos.z, a.pos.y, a.pos.x]
        for model in structure for chain in model
        for residue in chain for a in residue
    ])
    centroid_zyx = all_coords_zyx.mean(axis=0)  # (3,) zyx, Å

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    pos = atom.pos
                    p_A_zyx = np.array([pos.z, pos.y, pos.x])

                    # Step 1: atom Å → simulation voxel space
                    p_sim_vox = (p_A_zyx - centroid_zyx + box_centre_sim_A) / pixel_size

                    # Step 2: simulation voxels → cropped-box voxels
                    p_mob_vox = p_sim_vox - crop_start_zyx

                    # Step 3: apply alignment
                    p_ref_vox = R_inv @ (p_mob_vox - c_ref_zyx) + c_ref_zyx + t

                    # Step 4: reference voxels → Å (add map origin)
                    p_ref_A_zyx = p_ref_vox * pixel_size + origin_zyx
                    atom.pos = gemmi.Position(
                        float(p_ref_A_zyx[2]),
                        float(p_ref_A_zyx[1]),
                        float(p_ref_A_zyx[0]),
                    )

    out = Path(output_path)
    if out.suffix.lower() in {".cif", ".mmcif"}:
        structure.make_mmcif_document().write_file(str(out))
    else:
        structure.write_pdb(str(out))


def fit_map_in_map_from_files(
    mobile_path: str | os.PathLike[str],
    reference_path: str | os.PathLike[str],
    exhaustive_config: ExhaustiveSearchConfig | None = None,
    gradient_config: GradientRefinementConfig | None = None,
    mask_path: str | os.PathLike[str] | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> AlignmentResult:
    """Load MRC files and fit the mobile volume into the reference.

    Voxel sizes are read from MRC headers; if they differ, the mobile is
    automatically resampled to match the reference via Fourier rescaling.

    Parameters
    ----------
    mobile_path : str or Path
        Path to the mobile MRC map.
    reference_path : str or Path
        Path to the reference MRC map.
    exhaustive_config : ExhaustiveSearchConfig or None
        Parameters for the exhaustive SO(3) search.
    gradient_config : GradientRefinementConfig or None
        Parameters for local gradient refinement.  Pass ``None`` to skip.
    mask_path : str or Path or None
        Optional MRC mask (float values in ``[0, 1]``).
    device : torch.device or None
        Target device.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    AlignmentResult
    """
    from . import fit_map_in_map

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref, ref_px = _load_mrc(reference_path)
    mob, mob_px = _load_mrc(mobile_path)
    ref, mob, common_px = normalise_voxel_sizes(ref, mob, ref_px, mob_px)

    ref = ref.to(device)
    mob = mob.to(device)

    mask: torch.Tensor | None = None
    if mask_path is not None:
        mask, _ = _load_mrc(mask_path)
        mask = mask.to(device)

    if exhaustive_config is None:
        exhaustive_config = ExhaustiveSearchConfig(pixel_size_angstroms=common_px)
    if gradient_config is None and exhaustive_config.pixel_size_angstroms:
        gradient_config = GradientRefinementConfig(
            pixel_size_angstroms=exhaustive_config.pixel_size_angstroms
        )

    return fit_map_in_map(
        mob,
        ref,
        exhaustive_config=exhaustive_config,
        gradient_config=gradient_config,
        mask=mask,
        pixel_size_angstroms=common_px,
        verbose=verbose,
    )


def fit_pdb_in_map_from_files(
    mobile_pdb_path: str | os.PathLike[str],
    reference_map_path: str | os.PathLike[str],
    pixel_size_angstroms: float | None = None,
    box_size: int | None = None,
    *,
    desired_resolution_angstroms: float | None = None,
    save_simulated: bool = False,
    simulated_output_path: str | os.PathLike[str] | None = None,
    simulator: DensitySimulator | None = None,
    exhaustive_config: ExhaustiveSearchConfig | None = None,
    gradient_config: GradientRefinementConfig | None = None,
    mask_path: str | os.PathLike[str] | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> AlignmentResult:
    """Simulate a density from a PDB and fit it into the target density map.

    The PDB-derived simulation is the **mobile**; the density map is the **reference**.

    Parameters
    ----------
    mobile_pdb_path : str or Path
        Path to the atomic model (PDB or mmCIF) to be fitted.
    reference_map_path : str or Path
        Path to the experimental MRC density map.
    pixel_size_angstroms : float or None
        Voxel size for the simulated density in Angstroms.  When ``None``
        (default), the pixel size is read from the MRC header of
        *reference_map_path* so that the simulated density is directly
        comparable to the experimental map.
    box_size : int or None
        Cubic box size for the simulated density in voxels.  When ``None``
        (default), the largest dimension of the reference map is used.
    desired_resolution_angstroms : float or None
        If given, low-pass filter the simulated density to this resolution
        before alignment.  Must be >= 2 × *pixel_size_angstroms* (Nyquist).
    save_simulated : bool
        If ``True``, store the simulated density in
        ``AlignmentResult.simulated_volume``.
    simulated_output_path : str or Path or None
        If given, the simulated MRC is saved to this path.
    simulator : DensitySimulator or None
        Density simulator.  Defaults to
        :data:`~torch_fit_in_map._simulate.DEFAULT_SIMULATOR` which raises
        ``NotImplementedError`` with guidance until
        ``torch-calculate-electrostatic-potential`` is available.
    exhaustive_config : ExhaustiveSearchConfig or None
        Search parameters.
    gradient_config : GradientRefinementConfig or None
        Refinement parameters.
    mask_path : str or Path or None
        Optional MRC mask path.
    device : torch.device or None
        Target device.

    Returns
    -------
    AlignmentResult
        Includes ``simulated_volume`` when *save_simulated* is ``True``.
    """
    from . import fit_map_in_map

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if simulator is None:
        simulator = DEFAULT_SIMULATOR

    density_map, map_px = _load_mrc(reference_map_path)
    density_map = density_map.to(device)

    if pixel_size_angstroms is None:
        pixel_size_angstroms = map_px
    if box_size is None:
        box_size = max(density_map.shape[-3:])

    if desired_resolution_angstroms is not None and desired_resolution_angstroms < 2.0 * pixel_size_angstroms:
        raise ValueError(
            f"desired_resolution_angstroms ({desired_resolution_angstroms} Å) must be "
            f">= 2 × pixel_size ({2.0 * pixel_size_angstroms} Å)."
        )

    simulated = simulator.simulate(
        pdb_path=Path(mobile_pdb_path),
        pixel_size=pixel_size_angstroms,
        box_size=box_size,
        device=device,
    )

    if desired_resolution_angstroms is not None:
        from torch_fourier_filter.bandpass import low_pass_filter

        cutoff = pixel_size_angstroms / desired_resolution_angstroms
        lp = low_pass_filter(
            cutoff=cutoff,
            falloff=0.02,
            image_shape=simulated.shape,  # type: ignore[arg-type]
            rfft=True,
            fftshift=False,
            device=device,
        )
        ft = torch.fft.rfftn(simulated, norm="ortho")
        simulated = torch.fft.irfftn(ft * lp, s=simulated.shape, norm="ortho")

    density_map, simulated, common_px = normalise_voxel_sizes(
        density_map, simulated, map_px, pixel_size_angstroms
    )
    # After Fourier rescaling the simulated box may have a different shape than the
    # reference map (different box_size, non-cubic reference, rounding in rescale).
    # Crop/pad to match so that FFT cross-correlation is well-defined.
    simulated = crop_or_pad_to_shape(simulated, tuple(density_map.shape[-3:]))  # type: ignore[arg-type]

    mask: torch.Tensor | None = None
    if mask_path is not None:
        mask, _ = _load_mrc(mask_path)
        mask = mask.to(device)

    if exhaustive_config is None:
        exhaustive_config = ExhaustiveSearchConfig(pixel_size_angstroms=common_px)

    result = fit_map_in_map(
        simulated,
        density_map,
        exhaustive_config=exhaustive_config,
        gradient_config=gradient_config,
        mask=mask,
        pixel_size_angstroms=common_px,
        verbose=verbose,
    )

    if save_simulated:
        result.simulated_volume = simulated.cpu()

    if simulated_output_path is not None:
        _save_mrc(simulated_output_path, simulated, pixel_size=pixel_size_angstroms)

    return result
