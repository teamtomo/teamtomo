"""Compatibility adapter from the historical AtomStack API to AtomicStructure."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gemmi
import torch
from torch_structure_manipulation import AtomicStructure

from .potential import (
    calculate_scattering_potential_2d,
    calculate_scattering_potential_3d,
)
from .utils.peng_model import get_peng_scattering_parameters

if TYPE_CHECKING:
    from .grid import GridConfig


class AtomStack:
    """Expose the historical atom-stack API over an ``AtomicStructure``."""

    def __init__(
        self,
        atom_pos_zyx: torch.Tensor,
        atomic_numbers: torch.Tensor,
        atom_bfactors: torch.Tensor | float = 0.0,
        atom_occupancies: torch.Tensor | float = 1.0,
        atom_names: list[str] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize from coordinates and atomic numbers.

        Parameters
        ----------
        atom_pos_zyx : torch.Tensor
            Atom coordinates in Angstroms, zyx order, shape `(..., N, 3)`.
        atomic_numbers : torch.Tensor
            Integer atomic numbers, shape `(N,)`. Element symbols are derived
            from these.
        atom_bfactors : torch.Tensor | float, optional
            Atomic B-factors in Angstroms squared, shape `(N,)` or a scalar
            applied to every atom. Default is 0.0.
        atom_occupancies : torch.Tensor | float, optional
            Unitless atomic occupancies, shape `(N,)` or a scalar applied to
            every atom. Default is 1.0.
        atom_names : list[str] | None, optional
            Optional per-atom names, stored as the structure's atom names and
            returned by `atom_names`. If None, empty names are stored and
            `atom_names` returns None.
        device : torch.device | str, optional
            Device on which all atom data is stored. Inputs are moved to this
            device. Default is `"cpu"`.
        """
        resolved_device = torch.device(device)
        positions = atom_pos_zyx.to(resolved_device)
        numbers = torch.as_tensor(
            atomic_numbers, dtype=torch.int64, device=resolved_device
        )
        b_factors = torch.as_tensor(
            atom_bfactors, dtype=torch.float32, device=resolved_device
        )
        occupancies = torch.as_tensor(
            atom_occupancies, dtype=torch.float32, device=resolved_device
        )

        num_atoms = positions.shape[-2] if positions.ndim >= 2 else 0
        names = ("",) * num_atoms if atom_names is None else tuple(atom_names)
        elements = tuple(
            gemmi.Element(int(number)).name for number in numbers.detach().cpu()
        )
        self._structure = AtomicStructure(
            positions_zyx=positions,
            atomic_numbers=numbers,
            elements=elements,
            atom_names=names,
            b_factors=b_factors,
            occupancies=occupancies,
        )
        self._has_atom_names = atom_names is not None
        self._atom_params: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def structure(self) -> AtomicStructure:
        """The wrapped atomic structure."""
        return self._structure

    @property
    def device(self) -> torch.device:
        """Device on which the atom data is stored."""
        return self._structure.device

    @property
    def atom_names(self) -> list[str] | None:
        """Optional legacy list of atom names."""
        if not self._has_atom_names:
            return None
        return list(self._structure.atom_names)

    @property
    def atomic_numbers(self) -> torch.Tensor:
        """Integer atomic numbers with shape ``(N,)``."""
        return self._structure.atomic_numbers

    @property
    def atom_pos_zyx(self) -> torch.Tensor:
        """Atom coordinates in Angstroms with shape ``(..., N, 3)``."""
        return self._structure.positions_zyx

    @property
    def atom_bfactors(self) -> torch.Tensor:
        """Atomic B-factors in Angstroms squared."""
        return self._structure.b_factors

    @property
    def atom_occupancies(self) -> torch.Tensor:
        """Unitless atomic occupancies."""
        return self._structure.occupancies

    @property
    def atom_params_a(self) -> torch.Tensor:
        """Peng scattering amplitude parameters with shape ``(N, 5)``."""
        return self._peng_parameters[0]

    @property
    def atom_params_b(self) -> torch.Tensor:
        """Peng scattering width parameters with shape ``(N, 5)``."""
        return self._peng_parameters[1]

    @property
    def _peng_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._atom_params is None:
            self._atom_params = get_peng_scattering_parameters(
                self.atomic_numbers, device=self.device
            )
        return self._atom_params

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the stack."""
        return self._structure.num_atoms

    def __repr__(self) -> str:
        """Obtain the historical concise string representation."""
        batch_shape = tuple(self.atom_pos_zyx.shape[:-2])
        return (
            f"AtomStack(num_atoms={self.num_atoms}, "
            f"batch_shape={batch_shape}, "
            f"device={self.device})"
        )

    @classmethod
    def from_coords_and_names(
        cls,
        atom_pos_zyx: torch.Tensor,
        atom_names: list[str],
        atom_bfactors: torch.Tensor | float = 0.0,
        atom_occupancies: torch.Tensor | float = 1.0,
        device: torch.device | str = "cpu",
    ) -> AtomStack:
        """Construct an AtomStack from coordinates and element names.

        Parameters
        ----------
        atom_pos_zyx : torch.Tensor
            Atom coordinates in Angstroms, zyx order, shape `(..., N, 3)`.
        atom_names : list[str]
            Element symbols, one per atom (e.g. `"C"`, `"Fe"`), parsed with
            `gemmi.Element` to obtain atomic numbers. Also kept as the stack's
            `atom_names`.
        atom_bfactors : torch.Tensor | float, optional
            Atomic B-factors in Angstroms squared, shape `(N,)` or a scalar
            applied to every atom. Default is 0.0.
        atom_occupancies : torch.Tensor | float, optional
            Unitless atomic occupancies, shape `(N,)` or a scalar applied to
            every atom. Default is 1.0.
        device : torch.device | str, optional
            Device on which all atom data is stored. Inputs are moved to this
            device. Default is `"cpu"`.

        Returns
        -------
        AtomStack
            The new atom stack.
        """
        atomic_numbers = torch.tensor(
            [gemmi.Element(name).atomic_number for name in atom_names],
            dtype=torch.int64,
        )
        return cls(
            atom_pos_zyx,
            atomic_numbers,
            atom_bfactors,
            atom_occupancies,
            atom_names,
            device,
        )

    @classmethod
    def from_coords_and_atomic_numbers(
        cls,
        atom_pos_zyx: torch.Tensor,
        atomic_numbers: torch.Tensor,
        atom_bfactors: torch.Tensor | float = 0.0,
        atom_occupancies: torch.Tensor | float = 1.0,
        device: torch.device | str = "cpu",
    ) -> AtomStack:
        """Construct an AtomStack from coordinates and atomic numbers.

        Parameters
        ----------
        atom_pos_zyx : torch.Tensor
            Atom coordinates in Angstroms, zyx order, shape `(..., N, 3)`.
        atomic_numbers : torch.Tensor
            Integer atomic numbers, shape `(N,)`.
        atom_bfactors : torch.Tensor | float, optional
            Atomic B-factors in Angstroms squared, shape `(N,)` or a scalar
            applied to every atom. Default is 0.0.
        atom_occupancies : torch.Tensor | float, optional
            Unitless atomic occupancies, shape `(N,)` or a scalar applied to
            every atom. Default is 1.0.
        device : torch.device | str, optional
            Device on which all atom data is stored. Inputs are moved to this
            device. Default is `"cpu"`.

        Returns
        -------
        AtomStack
            The new atom stack, with `atom_names` set to None.
        """
        return cls(
            atom_pos_zyx, atomic_numbers, atom_bfactors, atom_occupancies, None, device
        )

    def to_scattering_potential_3d(
        self, grid_config: GridConfig, **kwargs: Any
    ) -> torch.Tensor:
        """Compute a 3D potential using the current physical kernel.

        Uses elemental Peng 1996 scattering parameters and the stack's
        B-factors and occupancies.

        Parameters
        ----------
        grid_config : GridConfig
            Three-dimensional output grid.
        **kwargs : Any
            Forwarded to `calculate_scattering_potential_3d`
            (`per_voxel_averaging`, `batch_size`, `verbose`).

        Returns
        -------
        torch.Tensor
            Potential volume(s) in volts, shape `(..., *grid_config.grid_shape)`.
        """
        return calculate_scattering_potential_3d(
            atom_pos_zyx=self.atom_pos_zyx,
            atom_bfactors=self.atom_bfactors,
            atom_params_a=self.atom_params_a,
            atom_params_b=self.atom_params_b,
            grid_config=grid_config,
            atom_occupancies=self.atom_occupancies,
            **kwargs,
        )

    def to_scattering_potential_2d(
        self, grid_config: GridConfig, **kwargs: Any
    ) -> torch.Tensor:
        """Compute a projected 2D potential using the current physical kernel.

        The z coordinate is dropped and the yx coordinates are projected onto
        the grid using elemental Peng 1996 scattering parameters and the
        stack's B-factors and occupancies.

        Parameters
        ----------
        grid_config : GridConfig
            Two-dimensional output grid in yx order.
        **kwargs : Any
            Forwarded to `calculate_scattering_potential_2d`
            (`per_voxel_averaging`, `batch_size`, `verbose`).

        Returns
        -------
        torch.Tensor
            Projected potential image(s) in V Angstrom, shape
            `(..., *grid_config.grid_shape)`.
        """
        return calculate_scattering_potential_2d(
            atom_pos_yx=self.atom_pos_zyx[..., 1:],
            atom_bfactors=self.atom_bfactors,
            atom_params_a=self.atom_params_a,
            atom_params_b=self.atom_params_b,
            grid_config=grid_config,
            atom_occupancies=self.atom_occupancies,
            **kwargs,
        )
