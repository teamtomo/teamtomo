"""Hierarchical S2 grid implementation based on healpy/healpix."""

import platform
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

if platform.system() != "Windows":
    import healpy as hp


def _check_healpy_available() -> None:
    """Raise a clear error if running on Windows, where healpy is unavailable."""
    if platform.system() == "Windows":
        raise ImportError("healpy cannot be installed on Windows systems.")


@dataclass(frozen=True)
class GridLevel:
    """A single HEALPix NESTED resolution within a HierarchicalS2Grid."""

    nside: int
    depth: int

    @property
    def npix(self) -> int:
        """Total number of pixels at this level."""
        _check_healpy_available()
        return hp.nside2npix(self.nside)

    @property
    def s2_step_deg(self) -> float:
        """Approximate angular spacing between neighboring pixels, in degrees."""
        return 58.6 / self.nside

    def angle_from_index(self, ipix: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Get (theta, phi) in radians for pixel index/indices at this level."""
        _check_healpy_available()
        return hp.pix2ang(self.nside, np.asarray(ipix), nest=True)

    def index_from_angle(self, theta: npt.ArrayLike, phi: npt.ArrayLike) -> np.ndarray:
        """Get the pixel index/indices closest to (theta, phi) angles at this level."""
        _check_healpy_available()
        return hp.ang2pix(self.nside, np.asarray(theta), np.asarray(phi), nest=True)

    def all_indices(self) -> np.ndarray:
        """Every pixel index at this level."""
        return np.arange(self.npix)

    def all_angles(self) -> tuple[np.ndarray, np.ndarray]:
        """(theta, phi) for every pixel at this level."""
        return self.angle_from_index(self.all_indices())


class HierarchicalS2Grid:
    """A stack of HEALPix NESTED-ordering resolutions for coarse-to-fine S2 search.

    Attributes
    ----------
    nside_finest: int
        The HEALPix nside of the finest resolution level.
    n_levels: int
        The total number of resolution levels in the grid.
    levels: list[GridLevel]
        List of GridLevel objects, from coarsest (index 0) to finest (index n_levels-1).
    """

    nside_finest: int
    n_levels: int
    levels: list[GridLevel]

    def __init__(self, nside_finest: int, n_levels: int):
        """Initialize with validation checks."""
        _check_healpy_available()
        if nside_finest & (nside_finest - 1) != 0:
            raise ValueError("nside_finest must be a power of 2")
        if nside_finest % (2 ** (n_levels - 1)) != 0:
            raise ValueError("nside_finest must be divisible by 2**(n_levels-1)")

        self.nside_finest = nside_finest
        self.n_levels = n_levels

        # levels[0] = coarsest ... levels[-1] = finest
        self.levels = [
            GridLevel(nside=nside_finest // (2**depth), depth=depth)
            for depth in range(n_levels - 1, -1, -1)
        ]

    @classmethod
    def from_target_step_deg(
        cls, target_step_deg: float, n_levels: int
    ) -> "HierarchicalS2Grid":
        """Create a grid whose finest level is closest to a target step, in degrees."""
        closest_nside = round(58.6 / target_step_deg)
        if closest_nside < 1:
            raise ValueError("Target step size too large; resulting nside < 1.")
        closest_nside = 2 ** int(np.round(np.log2(closest_nside)))  # nearest power of 2

        return cls(nside_finest=closest_nside, n_levels=n_levels)

    def __repr__(self) -> str:
        """String representation showing the grid's levels."""
        nsides = [level.nside for level in self.levels]
        return f"HierarchicalS2Grid(n_levels={self.n_levels}, nsides={nsides})"

    @property
    def finest_level(self) -> int:
        """Index of the finest level (always `n_levels - 1`)."""
        return self.n_levels - 1

    @property
    def coarsest_level(self) -> int:
        """Index of the coarsest level (always `0`).

        Returns
        -------
        int
            Index of the coarsest level in the grid (0).
        """
        return 0

    def get_level(self, level: int) -> GridLevel:
        """Get the grid level object for a given level index.

        Parameters
        ----------
        level: int
            Level index to retrieve (0 = coarsest, `n_levels - 1` = finest).

        Returns
        -------
        GridLevel
            The corresponding GridLevel object.

        Raises
        ------
        ValueError
            If the level index is out of bounds.
        """
        if level < 0 or level >= self.n_levels:
            raise ValueError(f"Level must be between 0 and {self.n_levels - 1}.")

        return self.levels[level]

    def get_level_orientations(
        self, level: int, degrees: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get (theta, phi) for every pixel at a given level.

        Parameters
        ----------
        level: int
            Level to query.
        degrees: bool
            If True, output angles are in degrees. Otherwise returned in radians.
            By default True.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (theta, phi) angles for every pixel at the specified level.
        """
        theta, phi = self.get_level(level).all_angles()
        if degrees:
            return np.degrees(theta), np.degrees(phi)
        return theta, phi

    def angle_from_index(
        self, ipix: npt.ArrayLike, level: int, degrees: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get (theta, phi) angles for pixel index/indices at a given level.

        Parameters
        ----------
        ipix: ArrayLike
            Pixel index/indices at `level`'s resolution.
        level: int
            Level to query.
        degrees: bool
            If True, output angles are in degrees. Otherwise returned in radians.
            By default True.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (theta, phi) angles for the specified pixel index/indices.
        """
        theta, phi = self.get_level(level).angle_from_index(ipix)
        if degrees:
            return np.degrees(theta), np.degrees(phi)
        return theta, phi

    def index_from_angle(
        self, theta: npt.ArrayLike, phi: npt.ArrayLike, level: int, degrees: bool = True
    ) -> np.ndarray:
        """Get the pixel index/indices closest to (theta, phi) at a given level.

        Parameters
        ----------
        theta: ArrayLike
            Polar angle(s).
        phi: ArrayLike
            Azimuthal angle(s).
        level: int
            Level to query.
        degrees: bool
            If True, input angles are in degrees. Otherwise assumed to be in radians.
            By default True.

        Returns
        -------
        np.ndarray
            Pixel index/indices at the specified level.
        """
        if degrees:
            theta = np.radians(theta)
            phi = np.radians(phi)
        return self.get_level(level).index_from_angle(theta, phi)

    def convert_index(
        self, ipix: npt.ArrayLike, from_level: int, to_level: int
    ) -> np.ndarray:
        """Translate pixel index/indices from one level's resolution to another's.

        Parameters
        ----------
        ipix: ArrayLike
            Pixel index/indices at `from_level`'s resolution.
        from_level: int
            Level that `ipix` is defined at.
        to_level: int
            Level to convert pixel indices to.

        Returns
        -------
        np.ndarray
            Pixel indices at `to_level`'s resolution.
        """
        source = self.get_level(from_level)
        target = self.get_level(to_level)
        ipix = np.atleast_1d(np.asarray(ipix, dtype=np.int64))

        depth_diff = int(np.log2(source.nside) - np.log2(target.nside))
        if depth_diff == 0:
            return ipix
        if depth_diff > 0:  # target is coarser: many-to-one
            return ipix >> (2 * depth_diff)

        # target is finer: one-to-many, each pixel expands to 4**d children
        n_finer = -depth_diff
        children = np.arange(4**n_finer, dtype=np.int64)
        return ((ipix << (2 * n_finer))[:, None] + children[None, :]).reshape(-1)

    def neighbor_indices(
        self, ipix: npt.ArrayLike, level: int, rings: int = 1
    ) -> np.ndarray:
        """Get pixel indices within `rings`-hops of `ipix`, at the same level.

        Parameters
        ----------
        ipix: ArrayLike
            Pixel index/indices at `level`'s resolution to find neighbors of.
        level: int
            Level that `ipix` is defined at.
        rings: int
            Number of neighbor-hops to include (0 = just `ipix`, 1 = immediate
            neighbors, etc.). Used to control local search window.

        Returns
        -------
        np.ndarray
            Unique pixel indices at `level`'s resolution that are within `rings`
            hops of `ipix`.
        """
        nside = self.get_level(level).nside
        seed = np.atleast_1d(np.asarray(ipix, dtype=np.int64))

        visited = set(seed.tolist())
        searching = set(seed.tolist())
        for _ in range(rings):
            if not searching:
                break
            neighbors = hp.get_all_neighbours(
                nside, np.fromiter(searching, dtype=np.int64), nest=True
            )
            neighbors = neighbors[neighbors >= 0]  # filter out invalid neighbors (-1)
            new_neighbors = set(neighbors.tolist()) - visited
            visited |= new_neighbors
            searching = new_neighbors

        return np.array(sorted(visited), dtype=np.int64)

    def sector_child_angles(
        self, coarse_level: int, fine_level: int, degrees: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """(theta, phi) for every `fine_level` pixel, grouped by `coarse_level` parent.

        Parameters
        ----------
        coarse_level: int
            Level whose pixels define the sectors to group by.
        fine_level: int
            Level to sample child angles from within each sector.
        degrees: bool
            If True, output angles are in degrees. Otherwise returned in radians.
            By default True.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(theta, phi)``, each of shape ``(n_sectors, k)``.
        """
        coarse = self.get_level(coarse_level)
        fine = self.get_level(fine_level)
        n_sectors = coarse.npix
        k = fine.npix // n_sectors

        fine_ipix = self.convert_index(
            coarse.all_indices(), from_level=coarse_level, to_level=fine_level
        ).reshape(n_sectors, k)

        theta, phi = self.angle_from_index(
            fine_ipix.reshape(-1), level=fine_level, degrees=degrees
        )
        return theta.reshape(n_sectors, k), phi.reshape(n_sectors, k)

    def sector_bounds_mask(
        self,
        coarse_level: int,
        fine_level: int,
        theta_min: float = 0.0,
        theta_max: float = 180.0,
        phi_min: float = 0.0,
        phi_max: float = 360.0,
        degrees: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Which `coarse_level` sectors have a `fine_level` child within bounds.

        Notes
        -----
        A sector is kept if any fine-level pixels fall within the given bounds rather
        than requiring all fine-level pixels to be within the bounds.

        Parameters
        ----------
        coarse_level: int
            Level whose pixels define the sectors to test.
        fine_level: int
            Level to sample child angles from within each sector.
        theta_min, theta_max, phi_min, phi_max: float
            Bounds on the child angles, in degrees if `degrees` else radians.
        degrees: bool
            If True, the bounds and returned angles are in degrees. Otherwise both are
            in radians. By default True.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(keep_mask, theta, phi)``: a boolean mask of shape ``(n_sectors,)`` and
            the grouped child angles of shape ``(n_sectors, k)`` used to compute it.
        """
        theta, phi = self.sector_child_angles(coarse_level, fine_level, degrees=degrees)
        keep_mask = (
            (theta >= theta_min)
            & (theta <= theta_max)
            & (phi >= phi_min)
            & (phi <= phi_max)
        ).any(axis=1)
        return keep_mask, theta, phi

    def followup(
        self,
        ipix: npt.ArrayLike,
        source_level: int,
        target_level: int,
        rings: int = 1,
    ) -> np.ndarray:
        """Find follow-up pixels at `target_level`, given `ipix` at `source_level`.

        Parameters
        ----------
        ipix: ArrayLike
            Pixel index/indices at `source_level`'s resolution to follow up on.
        source_level: int
            Level that `ipix` is defined at.
        target_level: int
            Level to return follow-up pixel indices at.
        rings: int
            Number of neighbor-hops around `ipix` (at `source_level`) to include.

        Returns
        -------
        np.ndarray
            Unique pixel indices at `target_level`'s resolution.
        """
        neighborhood = self.neighbor_indices(ipix, level=source_level, rings=rings)
        return np.unique(self.convert_index(neighborhood, source_level, target_level))
