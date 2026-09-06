"""PotentialSimulator protocol and default ESP implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from ._config import PotentialSimulatorConfig
from ._geometry import center_positions_in_simulation_box, simulation_box_center_zyx

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class PotentialSimulator(Protocol):
    """Protocol for simulating an electrostatic-potential map from a table of atoms.

    Atoms are supplied as a :class:`pandas.DataFrame` with (at least) the
    columns ``x``, ``y``, ``z`` (Cartesian coordinates in Angstroms) and
    ``element`` (atomic symbol, e.g. ``"C"``).  This is the column convention
    produced by :func:`mmdf.read`, so callers can obtain a compatible frame with
    ``mmdf.read("model.pdb")`` (file reading itself lives in the CLI wrapper,
    ``torch-fit-in-map-cli``).

    The default implementation uses ``torch-calculate-electrostatic-potential``.

    A custom simulator can be injected by implementing this protocol and passing
    it to :func:`fit_map_in_structure` or :func:`fit_structure_in_map`.

    Example
    -------
    .. code-block:: python

        import mmdf


        class MySimulator:
            def simulate(self, atoms, pixel_size, box_size, device=None, config=None):
                ...


        atoms = mmdf.read("model.pdb")
        result = fit_structure_in_map(atoms, potential_map, 1.5, 128, simulator=MySimulator())
    """

    def simulate(
        self,
        atoms: pd.DataFrame,
        pixel_size: float,
        box_size: int,
        device: torch.device | None = None,
        config: PotentialSimulatorConfig | None = None,
    ) -> torch.Tensor:
        """Simulate a potential map from a table of atoms.

        Parameters
        ----------
        atoms : pandas.DataFrame
            Atom table with columns ``x``, ``y``, ``z`` (Angstroms) and
            ``element`` (atomic symbol).
        pixel_size : float
            Voxel size in Angstroms.
        box_size : int
            Cubic box dimension in voxels.
        device : torch.device or None
            Target device for the output tensor.
        config : PotentialSimulatorConfig or None
            Simulator options. ``None`` uses the default configuration.

        Returns
        -------
        potential : torch.Tensor
            ``(box_size, box_size, box_size)`` float32 potential map in volts.
        """
        ...


# Deprecated alias; prefer PotentialSimulator.
DensitySimulator = PotentialSimulator


class _ESPSimulator:
    """Default simulator using the workspace electrostatic-potential APIs."""

    def simulate(
        self,
        atoms: pd.DataFrame,
        pixel_size: float,
        box_size: int,
        device: torch.device | None = None,
        config: PotentialSimulatorConfig | None = None,
    ) -> torch.Tensor:
        from torch_calculate_electrostatic_potential import (
            GridConfig,
            default_sublattice_radius,
            potential_from_structure_3d,
        )
        from torch_structure_manipulation import AtomicStructure

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config is None:
            config = PotentialSimulatorConfig()

        missing = {"x", "y", "z", "element"} - set(atoms.columns)
        if missing:
            raise ValueError(
                f"atoms DataFrame is missing required column(s): {sorted(missing)}"
            )
        if len(atoms) == 0:
            raise ValueError("atoms DataFrame is empty")
        if pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        if box_size <= 0:
            raise ValueError("box_size must be positive")

        use_bonded = config.scattering_factors == "peng_bonded"
        has_bonding_metadata = (
            "bonded_environments" in atoms.columns and "molecule_type" in atoms.columns
        )
        if use_bonded and has_bonding_metadata:
            structure = AtomicStructure.from_dataframe(atoms, device=device)
        elif use_bonded or config.annotate_bonding:
            structure = AtomicStructure.from_annotated_dataframe(
                atoms,
                include_hydrogens=config.include_hydrogens,
                device=device,
            )
        else:
            structure = AtomicStructure.from_dataframe(atoms, device=device)

        centered_positions = center_positions_in_simulation_box(
            structure.positions_zyx, box_size, pixel_size
        )
        structure = structure.with_positions(centered_positions)

        box_center_zyx = simulation_box_center_zyx(box_size, pixel_size, device=device)
        radius = (
            default_sublattice_radius(pixel_size)
            if config.sublattice_radius is None
            else config.sublattice_radius
        )
        grid_config = GridConfig.from_grid_shape_and_voxel_size(
            grid_shape=(box_size, box_size, box_size),
            voxel_size=(pixel_size, pixel_size, pixel_size),
            center_zyx=box_center_zyx,
            sublattice_radius=radius,
            device=device,
        )
        potential_zyx: torch.Tensor = potential_from_structure_3d(
            structure,
            grid_config,
            scattering_factors=config.scattering_factors,
            bonded_fallback=config.bonded_fallback,
            per_voxel_averaging=config.per_voxel_averaging,
            batch_size=config.batch_size,
            verbose=False,
        )
        return potential_zyx.float().contiguous()


DEFAULT_POTENTIAL_SIMULATOR: PotentialSimulator = _ESPSimulator()

# Deprecated alias; prefer DEFAULT_POTENTIAL_SIMULATOR.
DEFAULT_SIMULATOR = DEFAULT_POTENTIAL_SIMULATOR
