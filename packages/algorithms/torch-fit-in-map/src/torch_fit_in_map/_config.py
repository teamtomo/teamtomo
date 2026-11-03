"""Configuration dataclasses for alignment algorithms."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from teamtomo_basemodel import BaseModelTeamTomo


class PotentialSimulatorConfig(BaseModelTeamTomo):
    """Options for the default electrostatic-potential simulator.

    Attributes
    ----------
    scattering_factors : {"peng_elemental", "peng_bonded"}
        Peng parameter model passed to `potential_from_structure_3d`. Default
        is `"peng_elemental"`. With `"peng_bonded"`, existing
        `bonded_environments` and `molecule_type` columns are used if present;
        otherwise the atoms are annotated first.
    annotate_bonding : bool
        When `True` (or when `scattering_factors` is `peng_bonded`), build
        the structure with
        `AtomicStructure.from_annotated_dataframe`
        so bonded environments are available. Requires `chain`, `residue_id`,
        `residue`, and `atom` columns in addition to coordinates. Default is
        False.
    include_hydrogens : bool
        Passed to `from_annotated_dataframe` when bonding annotation runs.
        Default is True.
    sublattice_radius : float or None
        Per-atom stencil radius in Angstroms. `None` (default) uses
        `default_sublattice_radius`.
    per_voxel_averaging : bool
        Average the potential over each voxel instead of sampling its centre.
        Default is True.
    bonded_fallback : {"elemental", "error"}
        Behaviour for unsupported bonded providers or environment keys.
        Default is `"elemental"`.
    batch_size : int
        Number of atoms evaluated per chunk during potential calculation.
        Must be at least 1. Default is 4096.
    """

    scattering_factors: Literal["peng_elemental", "peng_bonded"] = "peng_elemental"
    annotate_bonding: bool = False
    include_hydrogens: bool = True
    sublattice_radius: float | None = None
    per_voxel_averaging: bool = True
    bonded_fallback: Literal["elemental", "error"] = "elemental"
    batch_size: int = Field(default=4096, ge=1)


class ExhaustiveSearchConfig(BaseModelTeamTomo):
    """Configuration for the exhaustive SO(3) grid search.

    Attributes
    ----------
    angular_step_degrees : float
        Step size in degrees for both `theta` and `psi` axes of the SO(3) grid.
        Smaller values give finer orientation sampling at the cost of runtime.
        Must be positive. Default is 15.0.
    angular_sampling_method : {"uniform", "healpix", "cartesian"}
        Base grid method passed to `get_uniform_euler_angles`. Default is
        `"uniform"`.
    rotation_batch_size : int
        Number of rotation matrices processed per GPU batch.  Reduce if you
        run out of VRAM; increase for faster throughput. Default is 16.
    symmetry : str
        Point-group symmetry of the *reference* map, e.g. `"C1"`, `"C4"`,
        `"D2"`, `"T"`, `"O"`, `"I"`.  The SO(3) search is restricted to
        the corresponding asymmetric unit, reducing runtime by the symmetry
        order.  Default `"C1"` performs a full SO(3) search.
    n_start : int
        Number of top-scoring poses from the exhaustive search to keep. When
        used through `fit_map_in_map`, each is refined independently and the
        best-scoring refined pose is returned, at the cost of `n_start` times
        the gradient-refinement time. `exhaustive_search` returns only the
        best pose. Must be at least 1. Default is 1.
    pixel_size_angstroms : float or None
        If provided, `AlignmentResult.translation_angstroms` is populated.
        Default is None.
    devices : list of str or None
        Devices to split the rotation search across, e.g.
        `["cuda:0", "cuda:1"]`. If None (default), the device of the reference
        volume is used.
    """

    angular_step_degrees: float = Field(default=15.0, gt=0.0)
    angular_sampling_method: Literal["uniform", "healpix", "cartesian"] = "uniform"
    rotation_batch_size: int = Field(default=16, ge=1)
    symmetry: str = "C1"
    n_start: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of top poses from the exhaustive search to refine independently. "
            "The best-scoring refined pose is returned.  n_start=1 gives the same "
            "behaviour as before; higher values improve robustness at the cost of "
            "n_start x gradient-refinement time."
        ),
    )
    pixel_size_angstroms: float | None = None
    devices: list[str] | None = Field(
        default=None,
        description=(
            "List of devices to use (e.g. ['cuda:0', 'cuda:1']). If None, uses "
            "the device of the input tensors."
        ),
    )


class GradientRefinementConfig(BaseModelTeamTomo):
    """Configuration for gradient-based local refinement.

    Attributes
    ----------
    optimizer : {"lbfgs", "adam"}
        Optimizer to use.  "lbfgs" is generally faster and more precise for
        low-dimensional problems like this.  "adam" is more robust to noise.
        Default is `"lbfgs"`.
    n_iterations : int
        Number of optimisation steps.  For "adam", the number of optimizer
        steps. For "lbfgs", passed as `max_iter` to `torch.optim.LBFGS`, which
        runs up to that many iterations in a single step. Must be at least 1.
        Default is 100.
    learning_rate : float
        Learning rate (step size). Must be positive. Default is 1.0; with
        "adam", a value of exactly 1.0 is replaced by 1e-2.
    loss : {"ncc", "mse"}
        Similarity metric minimised during optimisation. Default is `"ncc"`.
    pixel_size_angstroms : float or None
        If provided, `AlignmentResult.translation_angstroms` is populated.
        Default is None.
    devices : list of str or None
        Devices used by `fit_map_in_map` to refine multiple starting poses in
        parallel (poses are assigned round-robin). If None (default), the
        device of the reference map is used. Not used by `gradient_refine`
        itself, which runs on the device of its inputs.
    """

    optimizer: Literal["lbfgs", "adam"] = "lbfgs"
    n_iterations: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=1.0, gt=0.0)
    loss: Literal["ncc", "mse"] = "ncc"
    pixel_size_angstroms: float | None = None
    devices: list[str] | None = Field(
        default=None,
        description="List of devices to use for parallel refinement.",
    )


class ProjectionAlignmentConfig(BaseModelTeamTomo):
    """Configuration for the projection-based alignment.

    Attributes
    ----------
    angular_step_degrees : float
        Angular sampling step in degrees for the mobile orientation search.
        Must be positive. Default is 15.0.
    fftfreq_max : float or None
        High-frequency cutoff (cycles/pixel) applied to the projections.
        Passed to `project_3d_to_2d`. Default is None.
    pixel_size_angstroms : float or None
        If provided, `AlignmentResult.translation_angstroms` is populated.
        Default is None.
    """

    angular_step_degrees: float = Field(default=15.0, gt=0.0)
    fftfreq_max: float | None = None
    pixel_size_angstroms: float | None = None
