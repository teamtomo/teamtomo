"""Apply an alignment transform to a table of atomic coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._geometry import (
    coords_xyz_to_simulation_voxels,
    crop_start_xyz,
)

if TYPE_CHECKING:
    import pandas as pd

    from ._result import AlignmentResult


def apply_alignment_to_structure(
    atoms: pd.DataFrame,
    result: AlignmentResult,
    pixel_size: float,
    box_shape: tuple[int, int, int],
    sim_box_size: int | None = None,
    ref_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> pd.DataFrame:
    """Apply an :class:`AlignmentResult` transform to a table of atoms.

    The coordinate pipeline mirrors what the potential simulator and
    :func:`~torch_fit_in_map.crop_or_pad_to_shape` do during
    :func:`~torch_fit_in_map.fit_map_in_structure`, then applies the alignment and
    converts back to Angstroms:

    1. Atom Å → simulation voxels:
       ``p_sim = (atom_Å_zyx - centroid_Å_zyx + box_centre_sim_Å) / pixel_size``
    2. Simulation voxels → cropped-box voxels (accounts for ``crop_or_pad_to_shape``):
       ``p_mob = p_sim - crop_start_zyx``
    3. Apply alignment (rotation around box centre, then translate):
       ``p_ref = R⁻¹ @ (p_mob - c) + c + t``
    4. Reference voxels → Å (adds MRC origin):
       ``atom_ref_Å = p_ref * pixel_size + origin_Å``

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
    import torch
    from torch_structure_manipulation import (
        apply_rotation_to_coords,
        apply_translation_to_coords,
        df_to_atomxyz,
    )

    missing = {"x", "y", "z"} - set(atoms.columns)
    if missing:
        raise ValueError(
            f"atoms DataFrame is missing required column(s): {sorted(missing)}"
        )

    d, h, w = box_shape
    if sim_box_size is None:
        sim_box_size = max(d, h, w)

    # Rotation centre in reference (cropped) voxel space
    c_ref_zyx = torch.tensor([(d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0])

    coords_xyz = df_to_atomxyz(atoms)
    centroid_xyz = coords_xyz.mean(dim=0)

    # Step 1: atom Å → simulation voxel space
    p_sim_vox_xyz = coords_xyz_to_simulation_voxels(
        coords_xyz, centroid_xyz, sim_box_size, pixel_size
    )
    # Step 2: simulation voxels → cropped-box voxels
    p_mob_vox_xyz = p_sim_vox_xyz - crop_start_xyz(sim_box_size, box_shape)

    # Convert the alignment's pull-convention ZYX matrix into the conventional
    # column-vector XYZ rotation expected by the canonical structure kernel.
    rotation_xyz = result.rotation_matrix.detach().cpu().T.flip((0, 1))
    translation_xyz = result.translation_pixels.detach().cpu().flip(0)
    p_ref_vox_xyz = apply_rotation_to_coords(
        p_mob_vox_xyz,
        rotation_xyz,
        center_point=tuple(c_ref_zyx.flip(0).tolist()),
        zyx=False,
    )
    p_ref_vox_xyz = apply_translation_to_coords(p_ref_vox_xyz, translation_xyz)
    # Step 4: reference voxels → Å (add map origin)
    p_ref_A_xyz = p_ref_vox_xyz * pixel_size + torch.tensor(ref_origin_xyz)

    out = atoms.copy()
    out.loc[:, ["x", "y", "z"]] = p_ref_A_xyz.numpy()
    return out
