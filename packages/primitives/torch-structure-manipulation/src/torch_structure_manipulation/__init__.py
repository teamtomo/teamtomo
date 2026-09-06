"""Atomic structure data, bonding annotations, and structure transforms."""

from importlib.metadata import PackageNotFoundError, version

from .atomic_structure import AtomicStructure
from .bonding import (
    annotate_bonding_environments,
    classify_structure_composition,
    get_scattering_provider_keys,
)
from .structure_transforms import (
    apply_rotation,
    apply_rotation_to_coords,
    apply_translation,
    apply_translation_to_coords,
    ball_query_atoms,
    calculate_center_from_tensors,
    center_structure,
    center_structure_from_coords,
    create_rotation_matrix_from_euler,
    df_to_atomxyz,
    df_to_atomzyx,
    find_atoms_in_ball,
    get_nucleic_acid_residues,
    get_protein_residues,
    remove_sidechains,
    separate_protein_rna,
)

try:
    __version__ = version("torch-structure-manipulation")
except PackageNotFoundError:
    __version__ = "uninstalled"

__all__ = [
    "AtomicStructure",
    "annotate_bonding_environments",
    "apply_rotation",
    "apply_rotation_to_coords",
    "apply_translation",
    "apply_translation_to_coords",
    "ball_query_atoms",
    "calculate_center_from_tensors",
    "center_structure",
    "center_structure_from_coords",
    "classify_structure_composition",
    "create_rotation_matrix_from_euler",
    "df_to_atomxyz",
    "df_to_atomzyx",
    "find_atoms_in_ball",
    "get_nucleic_acid_residues",
    "get_protein_residues",
    "get_scattering_provider_keys",
    "remove_sidechains",
    "separate_protein_rna",
]
