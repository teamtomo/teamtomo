"""Integration tests: download EMD-39549 + 8YRQ, perturb, recover within 0.5 Å / 0.5°.

Convention note
---------------
``apply_alignment(vol, AlignmentResult(R, t))`` uses a pull transform:
    output[q] = vol[R @ (q - centre - t) + centre]

So ``fit_map_in_map(apply_alignment(ref, (R_p, t_p)), ref)`` returns
    R_pred  ≈  R_p^T        (inverse rotation)
    t_pred  ≈  -R_p @ t_p  (inverse translation)
"""

from __future__ import annotations

import gzip
import math
import shutil
import urllib.request
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

# ── download targets ──────────────────────────────────────────────────────────
_EMD_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-39549/map/emd_39549.map.gz"
)
_PDB_URL = "https://files.rcsb.org/download/8YRQ.pdb"

# Cap box size for test speed (voxels per side)
_MAX_BOX = 64


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_mrc(path: Path) -> tuple[torch.Tensor, float]:
    import mrcfile  # type: ignore[import]

    with mrcfile.open(str(path), mode="r") as mrc:
        data = torch.from_numpy(mrc.data.copy()).float()
        px = float(mrc.voxel_size.x) or 1.0
    return data, px


def _rotation_z_zyx(angle_deg: float) -> torch.Tensor:
    """Rotation around the Z axis expressed in ZYX matrix convention.

    In ZYX coordinates [z, y, x], a rotation by *angle_deg* around Z leaves
    the z-component unchanged and rotates y and x:

        [[1,   0,    0],
         [0,   cos,  sin],
         [0,  -sin,  cos]]
    """
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]],
        dtype=torch.float32,
    )


def _rotation_error_deg(R_a: torch.Tensor, R_b: torch.Tensor) -> float:
    """Geodesic angle in degrees between two rotation matrices."""
    cos_val = ((R_a.T @ R_b).trace() - 1.0) / 2.0
    return math.degrees(math.acos(cos_val.clamp(-1.0, 1.0).item()))


class _GaussianCaSimulator:
    """CA-atom Gaussian density simulator (not physically accurate; for tests only)."""

    sigma_A: float = 4.0  # Gaussian width in Ångström

    def simulate(
        self,
        atoms: pd.DataFrame,
        pixel_size: float,
        box_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        ca = atoms[atoms["atom"] == "CA"]
        ca_zyx = ca[["z", "y", "x"]].to_numpy(dtype=np.float32)
        if len(ca_zyx) == 0:
            raise ValueError("No CA atoms in the provided DataFrame")

        centroid = ca_zyx.mean(0)
        box_centre_A = (box_size - 1) / 2.0 * pixel_size
        # Atom positions in voxel coordinates, centred in the box
        vox = torch.tensor(
            (ca_zyx - centroid + box_centre_A) / pixel_size, dtype=torch.float32
        )  # (N_CA, 3)

        sigma_vox = self.sigma_A / pixel_size
        grid = torch.arange(box_size, dtype=torch.float32)
        zz, yy, xx = torch.meshgrid(grid, grid, grid, indexing="ij")
        coords = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)  # (B³, 3)

        # Accumulate Gaussians in chunks to keep memory bounded
        density = torch.zeros(coords.shape[0])
        chunk = 16
        for i in range(0, len(vox), chunk):
            batch = vox[i : i + chunk]  # (c, 3)
            diff = coords.unsqueeze(0) - batch.unsqueeze(1)  # (c, B³, 3)
            d2 = diff.pow(2).sum(-1)  # (c, B³)
            density += torch.exp(-d2 / (2 * sigma_vox**2)).sum(0)

        density = density.reshape(box_size, box_size, box_size)
        return density.to(device or torch.device("cpu"))


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def emdb_map(tmp_path_factory):
    """Download EMD-39549 once per test module into /tmp."""
    tmp = tmp_path_factory.mktemp("emdb")
    gz_path = tmp / "emd_39549.map.gz"
    map_path = tmp / "emd_39549.map"
    urllib.request.urlretrieve(_EMD_URL, gz_path)
    with gzip.open(gz_path, "rb") as f_in, open(map_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return map_path


@pytest.fixture(scope="module")
def pdb_8yrq(tmp_path_factory):
    """Download 8YRQ once per test module into /tmp."""
    tmp = tmp_path_factory.mktemp("pdb")
    pdb_path = tmp / "8yrq.pdb"
    urllib.request.urlretrieve(_PDB_URL, pdb_path)
    return pdb_path


# ── tests ─────────────────────────────────────────────────────────────────────


def _run_map_alignment_recovery(
    emdb_map: Path,
    angle_deg: float,
    shift_A: float,
    angular_step_degrees: float = 5.0,
    n_iterations: int = 200,
) -> None:
    """Shared logic for map-to-map alignment recovery tests.

    Perturbs EMD-39549 with ``angle_deg`` (Z-axis rotation) and ``shift_A``
    (Å along Z), runs ``fit_map_in_map``, and asserts recovery within 0.5°/0.5Å.

    Under the pull convention ``apply_alignment(ref, (R_p, t_p))`` produces
    a mobile whose true inverse is R_p^T / -R_p@t_p.
    """
    from torch_fit_in_map import (
        AlignmentResult,
        ExhaustiveSearchConfig,
        GradientRefinementConfig,
        apply_alignment,
        crop_or_pad_to_shape,
        fit_map_in_map,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref, pixel_size = _load_mrc(emdb_map)

    if max(ref.shape) > _MAX_BOX:
        target = tuple(min(s, _MAX_BOX) for s in ref.shape[-3:])
        ref = crop_or_pad_to_shape(ref, target)  # type: ignore[arg-type]

    ref = ref.to(device)

    R_perturb = _rotation_z_zyx(angle_deg).to(device)
    t_px = shift_A / pixel_size
    t_perturb = torch.tensor([t_px, 0.0, 0.0], dtype=torch.float32, device=device)

    mobile = apply_alignment(ref, AlignmentResult(R_perturb, t_perturb, score=1.0))

    result = fit_map_in_map(
        mobile,
        ref,
        exhaustive_config=ExhaustiveSearchConfig(
            angular_step_degrees=angular_step_degrees,
            pixel_size_angstroms=pixel_size,
        ),
        gradient_config=GradientRefinementConfig(
            n_iterations=n_iterations,
            pixel_size_angstroms=pixel_size,
        ),
        pixel_size_angstroms=pixel_size,
        verbose=False,
    )

    R_expected = R_perturb.T
    t_expected = -(R_perturb @ t_perturb)

    R_pred = result.rotation_matrix.cpu()
    t_pred = result.translation_pixels.cpu()

    angle_err = _rotation_error_deg(R_pred, R_expected.cpu())
    t_err_A = (t_pred - t_expected.cpu()).norm().item() * pixel_size

    assert angle_err < 0.5, f"Rotation error {angle_err:.3f}° > 0.5°"
    assert t_err_A < 0.5, f"Translation error {t_err_A:.3f} Å > 0.5 Å"


def test_map_alignment_translation_only(emdb_map):
    """CI-fast: pure translation (4 Å), 90° angular step (~24 orientations)."""
    _run_map_alignment_recovery(
        emdb_map,
        angle_deg=0.0,
        shift_A=4.0,
        angular_step_degrees=90.0,
        n_iterations=50,
    )


def test_map_alignment_90deg_rotation(emdb_map):
    """CI-fast: 90° Z-rotation + 2 Å shift, 90° angular step (rotation on grid)."""
    _run_map_alignment_recovery(
        emdb_map,
        angle_deg=90.0,
        shift_A=2.0,
        angular_step_degrees=90.0,
        n_iterations=50,
    )


@pytest.mark.slow
def test_map_alignment_recovery(emdb_map):
    """Perturb EMD-39549 with a known rigid transform and recover within 0.5 Å / 0.5°.

    Perturbation: 5° rotation around Z, 2 Å translation along Z.
    """
    _run_map_alignment_recovery(emdb_map, angle_deg=5.0, shift_A=2.0)


@pytest.mark.slow
def test_map_alignment_large_rotation(emdb_map):
    """Recover a large rotation (25°) with a moderate shift (4 Å) within 0.5°/0.5 Å."""
    _run_map_alignment_recovery(emdb_map, angle_deg=25.0, shift_A=4.0)


@pytest.mark.slow
def test_map_alignment_large_shift(emdb_map):
    """Recover a small rotation (8°) with a large shift (12 Å) within 0.5°/0.5 Å."""
    _run_map_alignment_recovery(emdb_map, angle_deg=8.0, shift_A=12.0)


@pytest.mark.slow
def test_structure_in_map_recovery(pdb_8yrq):
    """Simulate 8YRQ and recover placement within 0.5 Å / 0.5°.

    Uses ``fit_structure_in_map`` with a custom Gaussian CA-atom simulator. The
    structure is re-simulated internally (as the mobile) and fitted into a
    pre-perturbed reference density.  Because the *reference* is perturbed
    (not the mobile), the expected result is the forward transform
    (R_perturb, t_perturb), not its inverse.
    """
    import mmdf

    from torch_fit_in_map import (
        AlignmentResult,
        ExhaustiveSearchConfig,
        GradientRefinementConfig,
        apply_alignment,
        fit_structure_in_map,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sim = _GaussianCaSimulator()
    pixel_size = 2.0  # Å — coarse enough for speed, fine enough for 0.5 Å tolerance
    box_size = _MAX_BOX

    atoms = mmdf.read(str(pdb_8yrq))

    # Base density (unperturbed)
    ref_density = sim.simulate(atoms, pixel_size, box_size, device=device)

    # ── perturbation ──────────────────────────────────────────────────────────
    R_perturb = _rotation_z_zyx(5.0).to(device)
    t_px = 2.0 / pixel_size  # 2 Å in voxels
    t_perturb = torch.tensor([t_px, 0.0, 0.0], dtype=torch.float32, device=device)

    # Perturb the reference density; fitting will re-simulate the structure
    # (≈ ref_density) and fit it into this perturbed map.
    perturbed_ref = apply_alignment(
        ref_density, AlignmentResult(R_perturb, t_perturb, score=1.0)
    )

    # ── alignment ─────────────────────────────────────────────────────────────
    result = fit_structure_in_map(
        mobile_atoms=atoms,
        reference_map=perturbed_ref,
        pixel_size_angstroms=pixel_size,
        box_size=box_size,
        simulator=sim,
        exhaustive_config=ExhaustiveSearchConfig(
            angular_step_degrees=5.0,
            pixel_size_angstroms=pixel_size,
        ),
        gradient_config=GradientRefinementConfig(
            n_iterations=200,
            pixel_size_angstroms=pixel_size,
        ),
        verbose=False,
    )

    # ── expected transform ───────────────────────────────────────────────────
    # Here the *reference* is perturbed (not the mobile), so we expect the
    # forward transform: the simulated PDB (mobile) must be rotated/shifted
    # by (R_perturb, t_perturb) to match the displaced density.
    R_expected = R_perturb
    t_expected = t_perturb

    # ── verify ────────────────────────────────────────────────────────────────
    R_pred = result.rotation_matrix.cpu()
    t_pred = result.translation_pixels.cpu()

    angle_err = _rotation_error_deg(R_pred, R_expected.cpu())
    t_err_A = (t_pred - t_expected.cpu()).norm().item() * pixel_size

    assert angle_err < 0.5, f"Rotation error {angle_err:.3f}° > 0.5°"
    assert t_err_A < 0.5, f"Translation error {t_err_A:.3f} Å > 0.5 Å"
