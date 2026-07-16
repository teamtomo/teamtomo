"""Apply an alignment transform to a table of atomic coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from ._result import AlignmentResult


def transform_atoms(
    atoms: pd.DataFrame,
    result: AlignmentResult,
    pixel_size: float,
    box_shape: tuple[int, int, int],
    sim_box_size: int | None = None,
    ref_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> pd.DataFrame:
    """Apply an :class:`AlignmentResult` transform to a table of atoms.

    The coordinate pipeline mirrors what the density simulator and
    :func:`~torch_fit_in_map.crop_or_pad_to_shape` do during
    :func:`~torch_fit_in_map.fit_map_in_pdb`, then applies the alignment and
    converts back to Angstroms:

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
    atoms : pandas.DataFrame
        Atom table with columns ``x``, ``y``, ``z`` (Angstroms).  Any additional
        columns are preserved unchanged.
    result : AlignmentResult
        Alignment result (rotation in zyx, translation in zyx pixels).
    pixel_size : float
        Voxel size of the reference map in Angstroms.
    box_shape : tuple[int, int, int]
        ``(d, h, w)`` shape of the reference map.
    sim_box_size : int or None
        Cubic box size used during simulation.  Defaults to ``max(box_shape)``.
    ref_origin_xyz : tuple[float, float, float]
        XYZ origin of the reference map in Angstroms (from MRC header).

    Returns
    -------
    pandas.DataFrame
        A copy of *atoms* with the ``x``, ``y``, ``z`` columns transformed into
        the reference frame.
    """
    import numpy as np

    missing = {"x", "y", "z"} - set(atoms.columns)
    if missing:
        raise ValueError(
            f"atoms DataFrame is missing required column(s): {sorted(missing)}"
        )

    d, h, w = box_shape
    if sim_box_size is None:
        sim_box_size = max(d, h, w)

    # Rotation centre in reference (cropped) voxel space
    c_ref_zyx = np.array([(d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0])

    # Simulation box centre in Å (atoms were centred here during simulation)
    box_centre_sim_A = (sim_box_size - 1) / 2.0 * pixel_size

    # Crop offsets: simulation was cubic (sim_box_size³), then cropped to box_shape
    crop_start_zyx = np.array(
        [
            max(0, (sim_box_size - d) // 2),
            max(0, (sim_box_size - h) // 2),
            max(0, (sim_box_size - w) // 2),
        ],
        dtype=float,
    )

    # Reference map origin in ZYX order
    origin_zyx = np.array([ref_origin_xyz[2], ref_origin_xyz[1], ref_origin_xyz[0]])

    R_inv = result.rotation_matrix.detach().cpu().numpy().T  # R orthogonal → R⁻¹ = Rᵀ
    t = result.translation_pixels.detach().cpu().numpy()     # (3,) zyx

    # Atoms in ZYX Å; centroid over all atoms (same centroid used by the simulator)
    coords_zyx = atoms[["z", "y", "x"]].to_numpy(dtype=float)  # (N, 3) zyx
    centroid_zyx = coords_zyx.mean(axis=0)                     # (3,) zyx, Å

    # Step 1: atom Å → simulation voxel space
    p_sim_vox = (coords_zyx - centroid_zyx + box_centre_sim_A) / pixel_size
    # Step 2: simulation voxels → cropped-box voxels
    p_mob_vox = p_sim_vox - crop_start_zyx
    # Step 3: apply alignment (rotation around box centre, then translate)
    p_ref_vox = (p_mob_vox - c_ref_zyx) @ R_inv.T + c_ref_zyx + t
    # Step 4: reference voxels → Å (add map origin)
    p_ref_A_zyx = p_ref_vox * pixel_size + origin_zyx  # (N, 3) zyx

    out = atoms.copy()
    out["z"] = p_ref_A_zyx[:, 0]
    out["y"] = p_ref_A_zyx[:, 1]
    out["x"] = p_ref_A_zyx[:, 2]
    return out
