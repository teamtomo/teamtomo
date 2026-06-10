"""DensitySimulator protocol and default ESP implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class DensitySimulator(Protocol):
    """Protocol for simulating a density map from an atomic model.

    The default implementation uses ``torch-calculate-electrostatic-potential``
    (``espcalculator``) which must be installed separately::

        pip install git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git

    A custom simulator can be injected by implementing this protocol and passing
    it to :func:`fit_map_in_pdb`, :func:`fit_pdb_in_map`, or their
    ``_from_files`` variants.

    Example
    -------
    .. code-block:: python

        class MySimulator:
            def simulate(self, pdb_path, pixel_size, box_size, device=None):
                ...

        result = fit_pdb_in_map("model.pdb", density, 1.5, 128,
                                simulator=MySimulator())
    """

    def simulate(
        self,
        pdb_path: str | Path,
        pixel_size: float,
        box_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Simulate a density from an atomic model.

        Parameters
        ----------
        pdb_path : str or Path
            Path to the PDB or mmCIF file.
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
    """Default simulator using ``espcalculator`` (torch-calculate-electrostatic-potential).

    Atoms are loaded with ``gemmi``, centered in the box, and passed to
    ``espcalculator.calculate_esp``.  Install the optional dependency with::

        pip install git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git
    """

    def simulate(
        self,
        pdb_path: str | Path,
        pixel_size: float,
        box_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        try:
            import gemmi  # type: ignore[import]
            from espcalculator import AtomStack, Lattice, calculate_esp  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The default DensitySimulator requires 'espcalculator'.\n"
                "Install it with:\n"
                "  pip install git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git\n"
                "or pass a custom simulator= to fit_map_in_pdb / fit_pdb_in_map / their _from_files variants."
            ) from exc

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- load atoms with gemmi ---
        structure = gemmi.read_structure(str(pdb_path))
        coords: list[list[float]] = []
        names: list[str] = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        pos = atom.pos
                        coords.append([pos.x, pos.y, pos.z])
                        names.append(atom.element.name)

        if not coords:
            raise ValueError(f"No atoms found in {pdb_path}")

        coords_t = torch.tensor(coords, dtype=torch.float32, device=device)  # (N, 3)

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
