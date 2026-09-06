"""Tensor-backed atomic structure data."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import gemmi
import torch

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class AtomicStructure:
    """Lightweight numerical representation of an atomic structure.

    Positions use array-friendly ``(z, y, x)`` order. Numerical fields may have
    arbitrary broadcast-compatible batch dimensions before their atom dimension.
    Text metadata is shared across batches and held in immutable tuples.

    Bonding metadata (``bonded_environments``, ``molecule_types``) is not batched:
    one tuple is shared by every batch member. That matches ensemble-of-poses use
    cases (same chemistry, different coordinates) but not batched structures with
    different chemistry. See :meth:`from_annotated_dataframe` and the bonded-factor
    notes on :func:`torch_calculate_electrostatic_potential.potential_from_structure_3d`.
    """

    positions_zyx: torch.Tensor
    atomic_numbers: torch.Tensor
    elements: tuple[str, ...]
    atom_names: tuple[str, ...]
    b_factors: torch.Tensor
    occupancies: torch.Tensor
    bonded_environments: tuple[str, ...] | None = None
    molecule_types: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate atom dimensions and batch broadcasting."""
        if self.positions_zyx.ndim < 2 or self.positions_zyx.shape[-1] != 3:
            raise ValueError("positions_zyx must have shape (..., n_atoms, 3)")
        n_atoms = self.positions_zyx.shape[-2]
        if self.atomic_numbers.ndim < 1 or self.atomic_numbers.shape[-1] != n_atoms:
            raise ValueError(
                "atomic_numbers must have shape (..., n_atoms) with the same "
                "number of atoms as positions_zyx"
            )
        if len(self.elements) != n_atoms or len(self.atom_names) != n_atoms:
            raise ValueError("text metadata must have one value per atom")
        numerical_fields = {
            "b_factors": self.b_factors,
            "occupancies": self.occupancies,
        }
        for name, value in numerical_fields.items():
            if value.ndim > 0 and value.shape[-1] != n_atoms:
                raise ValueError(f"{name} must be scalar or have shape (..., n_atoms)")
        batch_shapes = [
            self.positions_zyx.shape[:-2],
            self.atomic_numbers.shape[:-1],
            *[
                value.shape[:-1] if value.ndim > 0 else ()
                for value in numerical_fields.values()
            ],
        ]
        try:
            torch.broadcast_shapes(*batch_shapes)  # type: ignore[no-untyped-call]
        except RuntimeError as error:
            raise ValueError(
                "numerical AtomicStructure fields have incompatible batch shapes"
            ) from error
        if (
            self.bonded_environments is not None
            and len(self.bonded_environments) != n_atoms
        ):
            raise ValueError("bonded_environments must have one value per atom")
        if self.molecule_types is not None and len(self.molecule_types) != n_atoms:
            raise ValueError("molecule_types must have one value per atom")

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> AtomicStructure:
        """Construct from an mmdf-compatible DataFrame.

        Required columns are ``x``, ``y``, ``z``, and ``element``. Atom names
        come from ``atom`` when present. ``b_isotropic`` and ``occupancy``
        default to zero and one, respectively. Optional ``bonded_environments``
        and ``molecule_type`` columns are preserved when present.
        """
        required = {"x", "y", "z", "element"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"missing required structure columns: {missing}")

        elements = tuple(str(value).strip().upper() for value in df["element"])
        if "atomic_number" in df:
            atomic_number_values = [int(value) for value in df["atomic_number"]]
        else:
            atomic_number_values = [
                gemmi.Element(element).atomic_number for element in elements
            ]
        unknown = sorted(
            element
            for element, atomic_number in zip(
                elements, atomic_number_values, strict=True
            )
            if atomic_number == 0
        )
        if unknown:
            raise ValueError(f"unknown element symbols: {unknown}")

        positions = torch.as_tensor(
            df.loc[:, ["z", "y", "x"]].to_numpy(copy=True),
            dtype=dtype,
            device=device,
        )
        atomic_numbers = torch.tensor(
            atomic_number_values,
            dtype=torch.int64,
            device=device,
        )
        b_values = df["b_isotropic"] if "b_isotropic" in df else [0.0] * len(df)
        occupancy_values = df["occupancy"] if "occupancy" in df else [1.0] * len(df)
        atom_names = (
            tuple(str(value).strip() for value in df["atom"])
            if "atom" in df
            else ("",) * len(df)
        )
        bonded = (
            tuple(str(value) for value in df["bonded_environments"])
            if "bonded_environments" in df
            else None
        )
        molecule_types = (
            tuple(str(value) for value in df["molecule_type"])
            if "molecule_type" in df
            else None
        )
        return cls(
            positions_zyx=positions,
            atomic_numbers=atomic_numbers,
            elements=elements,
            atom_names=atom_names,
            b_factors=torch.as_tensor(b_values, dtype=dtype, device=device),
            occupancies=torch.as_tensor(occupancy_values, dtype=dtype, device=device),
            bonded_environments=bonded,
            molecule_types=molecule_types,
        )

    @classmethod
    def from_annotated_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        include_hydrogens: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> AtomicStructure:
        """Annotate bonding metadata, then construct from the result.

        This is the usual entry point for Peng bonded scattering factors: it
        calls :func:`~torch_structure_manipulation.annotate_bonding_environments`
        to add ``bonded_environments`` and ``molecule_type`` columns, then
        delegates to :meth:`from_dataframe`.

        The input must include ``chain``, ``residue_id``, ``residue``, ``atom``,
        and ``element`` in addition to the coordinate columns required by
        :meth:`from_dataframe`.
        """
        from .bonding import annotate_bonding_environments

        annotated = annotate_bonding_environments(df, include_hydrogens=include_hydrogens)
        return cls.from_dataframe(annotated, device=device, dtype=dtype)

    @property
    def num_atoms(self) -> int:
        """Number of atoms in each structure."""
        return self.positions_zyx.shape[-2]

    @property
    def batch_shape(self) -> torch.Size:
        """Broadcasted batch shape of all numerical fields."""
        batch_shapes = [
            self.positions_zyx.shape[:-2],
            self.atomic_numbers.shape[:-1],
            self.b_factors.shape[:-1] if self.b_factors.ndim > 0 else (),
            self.occupancies.shape[:-1] if self.occupancies.ndim > 0 else (),
        ]
        return torch.Size(
            torch.broadcast_shapes(*batch_shapes)  # type: ignore[no-untyped-call]
        )

    @property
    def device(self) -> torch.device:
        """Device containing the atomic positions."""
        return self.positions_zyx.device

    def with_positions(self, positions_zyx: torch.Tensor) -> AtomicStructure:
        """Return a copy with replacement, broadcast-compatible positions."""
        if positions_zyx.ndim < 2 or positions_zyx.shape[-2:] != (self.num_atoms, 3):
            raise ValueError(
                "replacement positions must have shape (..., n_atoms, 3) with the "
                "same number of atoms"
            )
        return replace(self, positions_zyx=positions_zyx)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> AtomicStructure:
        """Return a copy with numerical tensors moved to a device and dtype."""
        floating_dtype = self.positions_zyx.dtype if dtype is None else dtype
        return replace(
            self,
            positions_zyx=self.positions_zyx.to(device=device, dtype=floating_dtype),
            atomic_numbers=self.atomic_numbers.to(device=device),
            b_factors=self.b_factors.to(device=device, dtype=floating_dtype),
            occupancies=self.occupancies.to(device=device, dtype=floating_dtype),
        )
