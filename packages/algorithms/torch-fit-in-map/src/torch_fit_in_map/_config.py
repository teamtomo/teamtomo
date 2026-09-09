"""Configuration dataclasses for alignment algorithms."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from teamtomo_basemodel import BaseModelTeamTomo


class PotentialSimulatorConfig(BaseModelTeamTomo):
    """Options for the default electrostatic-potential simulator.

    Parameters
    ----------
    scattering_factors : {"peng_elemental", "peng_bonded"}
        Peng parameter model passed to ``potential_from_structure_3d``.
    annotate_bonding : bool
        When ``True`` (or when ``scattering_factors`` is ``peng_bonded``), build
        the structure with :meth:`~torch_structure_manipulation.AtomicStructure.from_annotated_dataframe`
        so bonded environments are available. Requires ``chain``, ``residue_id``,
        ``residue``, and ``atom`` columns in addition to coordinates.
    include_hydrogens : bool
        Passed to ``from_annotated_dataframe`` when bonding annotation runs.
    sublattice_radius : float or None
        Per-atom stencil radius in Angstroms. ``None`` uses
        :func:`~torch_calculate_electrostatic_potential.default_sublattice_radius`.
    per_voxel_averaging : bool
        Average the potential over each voxel instead of sampling its centre.
    bonded_fallback : {"elemental", "error"}
        Behaviour for unsupported bonded providers or environment keys.
    batch_size : int
        Number of atoms evaluated per chunk during potential calculation.
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

    Parameters
    ----------
    angular_step_degrees : float
        Step size for both ``theta`` and ``psi`` axes of the SO(3) grid.
        Smaller values give finer orientation sampling at the cost of runtime.
    angular_sampling_method : {"uniform", "healpix", "cartesian"}
        Base grid method passed to ``get_uniform_euler_angles``.
    rotation_batch_size : int
        Number of rotation matrices processed per GPU batch.  Reduce if you
        run out of VRAM; increase for faster throughput.
    symmetry : str
        Point-group symmetry of the *reference* map, e.g. ``"C1"``, ``"C4"``,
        ``"D2"``, ``"T"``, ``"O"``, ``"I"``.  The SO(3) search is restricted to
        the corresponding asymmetric unit, reducing runtime by the symmetry
        order.  Default ``"C1"`` performs a full SO(3) search.
    pixel_size_angstroms : float or None
        If provided, ``AlignmentResult.translation_angstroms`` is populated.
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

    Parameters
    ----------
    optimizer : {"lbfgs", "adam"}
        Optimizer to use.  "lbfgs" is generally faster and more precise for
        low-dimensional problems like this.  "adam" is more robust to noise.
    n_iterations : int
        Number of optimisation steps.  For "lbfgs", this corresponds to the
        maximum number of iterations within the line search.
    learning_rate : float
        Learning rate (step size).  Default is 1.0 for "lbfgs" and 1e-2 for "adam".
    loss : {"ncc", "mse"}
        Similarity metric minimised during optimisation.
    pixel_size_angstroms : float or None
        If provided, ``AlignmentResult.translation_angstroms`` is populated.
    devices : list of str or None
        List of devices to use for parallel refinement of multiple poses.
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

    Parameters
    ----------
    angular_step_degrees : float
        Angular sampling step for the mobile orientation search.
    fftfreq_max : float or None
        High-frequency cutoff (cycles/pixel) applied to the projections.
        Passed to ``project_3d_to_2d``.
    pixel_size_angstroms : float or None
        If provided, ``AlignmentResult.translation_angstroms`` is populated.
    """

    angular_step_degrees: float = Field(default=15.0, gt=0.0)
    fftfreq_max: float | None = None
    pixel_size_angstroms: float | None = None
