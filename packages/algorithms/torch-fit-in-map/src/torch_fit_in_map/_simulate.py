"""DensitySimulator protocol and default ESP implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class DensitySimulator(Protocol):
    """Protocol for simulating a density map from a table of atoms.

    Atoms are supplied as a :class:`pandas.DataFrame` with (at least) the
    columns ``x``, ``y``, ``z`` (Cartesian coordinates in Angstroms) and
    ``element`` (atomic symbol, e.g. ``"C"``).  This is the column convention
    produced by :func:`mmdf.read`, so callers can obtain a compatible frame with
    ``mmdf.read("model.pdb")`` (file reading itself lives in the CLI wrapper,
    ``torch-fit-in-map-cli``).

    The default implementation uses ``torch-calculate-electrostatic-potential``
    (``espcalculator``) which must be installed separately::

        pip install git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git

    A custom simulator can be injected by implementing this protocol and passing
    it to :func:`fit_map_in_pdb` or :func:`fit_pdb_in_map`.

    Example
    -------
    .. code-block:: python

        import mmdf

        class MySimulator:
            def simulate(self, atoms, pixel_size, box_size, device=None):
                ...

        atoms = mmdf.read("model.pdb")
        result = fit_pdb_in_map(atoms, density, 1.5, 128,
                                simulator=MySimulator())
    """

    def simulate(
        self,
        atoms: pd.DataFrame,
        pixel_size: float,
        box_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Simulate a density from a table of atoms.

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

        Returns
        -------
        density : torch.Tensor
            ``(box_size, box_size, box_size)`` float32 density map.
        """
        ...


class _ESPSimulator:
    """Default simulator using ``espcalculator`` (electrostatic potential).

    Atom coordinates and element symbols are taken from the ``atoms`` DataFrame,
    centered in the box, and passed to ``espcalculator.calculate_esp``.  Install
    the optional dependency with::

        pip install git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git
    """

    def simulate(
        self,
        atoms: pd.DataFrame,
        pixel_size: float,
        box_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        try:
            from espcalculator import (  # type: ignore[import]
                AtomStack,
                Lattice,
                calculate_esp,
            )
        except ImportError as exc:
            raise ImportError(
                "The default DensitySimulator requires 'espcalculator'.\n"
                "Install it with:\n"
                "  pip install git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git\n"
                "or pass a custom simulator= to fit_map_in_pdb / fit_pdb_in_map."
            ) from exc

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- read atoms from the DataFrame ---
        missing = {"x", "y", "z", "element"} - set(atoms.columns)
        if missing:
            raise ValueError(
                f"atoms DataFrame is missing required column(s): {sorted(missing)}"
            )
        if len(atoms) == 0:
            raise ValueError("atoms DataFrame is empty")

        names: list[str] = atoms["element"].astype(str).tolist()
        coords_t = torch.tensor(
            atoms[["x", "y", "z"]].to_numpy(),
            dtype=torch.float32,
            device=device,
        )  # (N, 3)

        # --- center atoms at box centre ---
        box_centre_a = (box_size - 1) / 2.0 * pixel_size
        centroid = coords_t.mean(dim=0)
        coords_t = coords_t - centroid + box_centre_a  # (N, 3)

        # AtomStack expects (B=1, N, 3)
        atom_stack = AtomStack.from_coords_and_names(
            atom_coordinates=coords_t.unsqueeze(0),
            atom_names=names,
            device=device,
        )
        atom_stack.fill_constant_bfactor(8.0 * torch.pi**2 * 0.5**2)

        # --- lattice: grid covers [0, (box_size-1)*pixel_size]^3 ---
        extent = (box_size - 1) * pixel_size
        lattice = Lattice.from_grid_dimensions_and_voxel_sizes(
            grid_dimensions=(box_size, box_size, box_size),
            voxel_sizes_in_A=(pixel_size, pixel_size, pixel_size),
            left_bottom_point_in_A=(0.0, 0.0, 0.0),
            right_upper_point_in_A=(extent, extent, extent),
            sublattice_radius_in_A=max(5.0, 3.0 * pixel_size),
            device=device,
        )

        density_xyz: torch.Tensor = calculate_esp(
            atom_stack=atom_stack,
            lattice=lattice,
            B=64,
            per_voxel_averaging=True,
            verbose=False,
        )
        # espcalculator returns (Dx, Dy, Dz) in XYZ order (axis 0 = X).
        # Transpose to ZYX so the output matches MRC convention and our pipeline.
        return density_xyz.permute(2, 1, 0).float().contiguous()


DEFAULT_SIMULATOR: DensitySimulator = _ESPSimulator()  # type: ignore[assignment]
