"""Tilt series geometry and point projection for cryo-ET."""

from collections.abc import Callable
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_3d import Rx, Ry, Rz, T

LocalShiftFn = Callable[[torch.Tensor], torch.Tensor]


def _writable(data):
    if isinstance(data, np.ndarray) and not data.flags.writeable:
        data = data.copy()
    return data


def _as_tensor(data, device: torch.device | str) -> torch.Tensor:
    return torch.as_tensor(_writable(data), device=device).float()


class TiltSeries:
    """Tilt series alignment geometry and 3D -> 2D point projection.

    Holds alignment parameters, all in Angstroms, plus metadata describing
    where matching raw tilt images live (`image_path`, `image_indices`,
    `pixel_spacing`). It never reads or holds image pixel data itself -
    see `torch_tilt_series.io.load_tilt_series_images` and
    `torch_tilt_series.preprocessing` for that. `project_points` maps 3D
    points (Angstroms) to 2D detector positions (Angstroms); converting
    to/from pixel coordinates happens in `torch_reconstruct_tomogram`.

    Coordinate spaces:
    - sample space: canonical 3D space representing the sample before stage
      rotation (`x_tilts` is defined here). Volume deformations (e.g. local
      warping) are modelled relative to this space -> see `self.local_shifts`.
    - levelled sample space: sample space plus a fixed, data-derived
      correction (e.g. a leveling rotation), via `sample2levelled`.
    - tomogram space: arbitrary 3D reconstruction/visualization volume; may
      be reoriented relative to levelled sample space via `levelled2tomo`.
      Points passed to `project_points` are given in this space.
    - microscope space: fixed 3D system, tilt axis pinned along y.
    - detector space: 2D, rotated xy plane aligned to the detector's
      row/col pixel axes.

    named transforms
        sample2levelled : sample -> levelled sample   (`self.sample2levelled`,
                                                         default identity)
        levelled2sample : levelled sample -> sample   (`self.levelled2sample`,
                                                         inverse of
                                                         sample2levelled)
        levelled2tomo   : levelled sample -> tomogram (`self.levelled2tomo`,
                                                         default identity)
        tomo2levelled   : tomogram -> levelled sample (`self.tomo2levelled`,
                                                         inverse of
                                                         levelled2tomo)
        sample2tomo     : sample -> tomogram          (`self.sample2tomo`,
                                                         = levelled2tomo @
                                                         sample2levelled)
        tomo2sample     : tomogram -> sample          (`self.tomo2sample`,
                                                         inverse of
                                                         sample2tomo)
        sample2scope    : sample -> microscope        (`self.sample2scope`,
                                                         per tilt)
        scope2sample    : microscope -> sample        (`self.scope2sample`,
                                                         inverse of
                                                         sample2scope)
        scope2detector  : microscope -> detector      (`self.scope2detector`,
                                                         per tilt)
        detector2scope  : detector -> microscope      (`self.detector2scope`,
                                                         inverse of
                                                         scope2detector; see
                                                         caveat in its
                                                         docstring)
        projection_matrices = scope2detector @ sample2scope

    `project_points` composes `tomo2sample` with `projection_matrices` to go
    tomogram space -> detector space (Angstroms) in one call.

    Parameters
    ----------
    tilt_angles : torch.Tensor
        `(n_tilts,)` stage tilt angles in degrees (rotation about the
        microscope y axis). Array-likes such as numpy arrays are accepted and
        converted to float32 tensors on `device`.
    tilt_axis_angle : torch.Tensor
        In-plane rotation of the tilt axis on the detector, in degrees
        (rotation about the optical/z axis). Either a scalar or `(n_tilts,)`,
        one value per tilt.
    sample_translations : torch.Tensor
        `(n_tilts, 2)` per-tilt 2D shifts in `(y, x)` order, in Angstroms,
        applied in detector space after rotation and projection.
    x_tilts : torch.Tensor | float
        X-axis tilt of the specimen (IMOD XAXISTILT / XTILTFILE convention,
        sign flipped by the etomo loader), in degrees. A scalar or
        `(n_tilts,)`. Applied to sample points before the stage tilt.
        Default 0.0.
    sample2levelled : torch.Tensor | None
        `(4, 4)` homogeneous zyxw transform from sample space to levelled
        sample space. Defaults to identity.
    levelled2tomo : torch.Tensor | None
        `(4, 4)` homogeneous zyxw transform from levelled sample space to
        tomogram space. Defaults to identity, i.e. tomogram space equals
        levelled sample space.
    local_shifts : Callable[[torch.Tensor], torch.Tensor] | None
        Optional sample-space deformation model. Called in `project_points`
        with sample-space points `(n_points, 3)` (zyx, Angstroms) and must
        return an additive correction of the same shape, in Angstroms.
    local_shifts_2d : Callable[[torch.Tensor], torch.Tensor] | None
        Optional per-tilt 2D correction. Called in `project_points` with the
        projected points `(n_points, n_tilts, 2)` (yx), or
        `(n_points, n_tilts, 4)` (zyxw) when `output_zyxw=True`, in
        Angstroms, and must return an additive correction of the same shape.
    image_path : Path | str | None
        Path to the MRC stack holding the raw tilt images. Only used by
        `load_tilt_series_images`; no image data is read on construction.
    image_indices : torch.Tensor | np.ndarray | None
        `(n_tilts,)` 0-based indices into the stack at `image_path`, selecting
        and ordering the images that match `tilt_angles`. Stored as an int64
        tensor on the CPU. If None, all images in the stack are used in order.
    pixel_spacing : float | None
        Pixel size of the raw images at `image_path`, in Angstroms. Accessing
        the `pixel_spacing` property raises `ValueError` while it is unset.
    device : torch.device | str
        Device on which the geometry tensors are stored. Default `"cpu"`.
    """

    def __init__(
        self,
        tilt_angles: torch.Tensor,
        tilt_axis_angle: torch.Tensor,
        sample_translations: torch.Tensor,
        x_tilts: torch.Tensor | float = 0.0,
        sample2levelled: torch.Tensor | None = None,
        levelled2tomo: torch.Tensor | None = None,
        local_shifts: LocalShiftFn | None = None,
        local_shifts_2d: LocalShiftFn | None = None,
        image_path: Path | str | None = None,
        image_indices: torch.Tensor | np.ndarray | None = None,
        pixel_spacing: float | None = None,
        device: torch.device | str = "cpu",
    ):
        self.tilt_angles = _as_tensor(tilt_angles, device)
        self.tilt_axis_angle = _as_tensor(tilt_axis_angle, device)
        # Sample translations, in Angstroms, (y, x) per tilt.
        self.sample_translations = _as_tensor(sample_translations, device)
        # X-axis tilt (IMOD XAXISTILT / XTILTFILE), scalar or per-tilt, in degrees.
        self.x_tilts = _as_tensor(x_tilts, device)
        # sample -> levelled-sample correction. Defaults
        # to identity.
        self.sample2levelled = _as_tensor(
            sample2levelled if sample2levelled is not None else torch.eye(4), device
        )
        # Arbitrary levelled-sample -> tomogram transform.
        # Defaults to identity, i.e. tomogram space == levelled sample space.
        self.levelled2tomo = _as_tensor(
            levelled2tomo if levelled2tomo is not None else torch.eye(4), device
        )
        # Sample-space deformation model: called with sample-space points
        # (n_points, 3) in Angstroms, must return an Angstrom-space
        # correction of the same shape.
        self.local_shifts = local_shifts
        self.local_shifts_2d = local_shifts_2d
        self.image_path = Path(image_path) if image_path is not None else None
        self.image_indices = (
            torch.as_tensor(_writable(image_indices)).long()
            if image_indices is not None
            else None
        )
        self._pixel_spacing = pixel_spacing
        self.device = device

    @property
    def pixel_spacing(self) -> float:
        """Pixel size of the raw images at image_path, in Angstroms."""
        if self._pixel_spacing is None:
            raise ValueError(
                "pixel_spacing is not set -> construct the TiltSeries via a "
                "torch_tilt_series loader (e.g. from_aretomo_output, "
                "from_etomo_directory), or set it yourself."
            )
        return self._pixel_spacing

    @pixel_spacing.setter
    def pixel_spacing(self, value: float | None) -> None:
        self._pixel_spacing = value

    @property
    def levelled2sample(self) -> torch.Tensor:
        """Inverse of sample2levelled: levelled sample space -> sample space."""
        return torch.linalg.inv(self.sample2levelled)

    @property
    def tomo2levelled(self) -> torch.Tensor:
        """Inverse of levelled2tomo: tomogram space -> levelled sample space."""
        return torch.linalg.inv(self.levelled2tomo)

    @property
    def sample2tomo(self) -> torch.Tensor:
        """Composition of sample2levelled and levelled2tomo: sample -> tomogram."""
        return self.levelled2tomo @ self.sample2levelled

    @property
    def tomo2sample(self) -> torch.Tensor:
        """Inverse of sample2tomo: tomogram space -> sample space."""
        return torch.linalg.inv(self.sample2tomo)

    @property
    def sample2scope(self) -> torch.Tensor:
        """Rotation from sample space to microscope space, per tilt."""
        # X-axis tilt is an intrinsic property of the specimen, so it is applied
        # to sample points first (innermost), before the per-view stage tilt.
        rx = Rx(self.x_tilts, zyx=True, device=self.device)
        r0 = Ry(self.tilt_angles, zyx=True, device=self.device)
        return r0 @ rx

    @property
    def scope2sample(self) -> torch.Tensor:
        """Inverse of sample2scope: microscope space -> sample space, per tilt."""
        return torch.linalg.inv(self.sample2scope)

    @property
    def scope2detector(self) -> torch.Tensor:
        """Transform from microscope space to detector space, per tilt.

        Aligns the microscope's fixed y/x axes to the detector's row/col
        pixel axes (in-plane rotation by tilt_axis_angle about the
        optical/Z axis), then adds the per-view 2D shift, in Angstroms.

        Kept as a (n_tilts, 4, 4) matrix, matching `sample2scope`, even
        though only the y/x output rows are physically meaningful once
        applied to a scope-space point (z is an identity passthrough here).
        Callers wanting genuine 2D coordinates take rows `[..., [1, 2], :]`,
        as `project_points` does.
        """
        shifts_3d = F.pad(self.sample_translations, (1, 0), value=0)
        r1 = Rz(self.tilt_axis_angle, zyx=True, device=self.device)
        t2 = T(shifts_3d, device=self.device)
        return t2 @ r1

    @property
    def detector2scope(self) -> torch.Tensor:
        """Inverse of scope2detector: detector space -> microscope space, per tilt.

        Caveat: this inverts the full affine map on a homogeneous zyxw point
        where z has NOT been dropped. Projecting scope space -> detector
        space discards the z (depth/optical-axis) coordinate, and that
        information is genuinely unrecoverable from a real 2D detector
        position alone. This property exists for API symmetry; it is not a
        way to reconstruct 3D scope-space points from actual 2D detector
        coordinates without an assumed/known z (e.g. a defocus plane).
        """
        return torch.linalg.inv(self.scope2detector)

    @property
    def projection_matrices(self) -> torch.Tensor:
        """Matrices that project points from sample space to detector space.

        projection_matrices = scope2detector @ sample2scope
        (T(shift) @ Rz(tilt_axis_angle) @ Ry(tilt_angle) @ Rx(x_tilt))

        Does not include the sample2tomo/tomo2sample step:project_points
        applies that separately, first.
        """
        return self.scope2detector @ self.sample2scope

    def to(self, device: torch.device | str) -> None:
        """Move the geometry tensors of the tilt series to `device`, in place.

        Moves `tilt_angles`, `tilt_axis_angle`, `sample_translations`,
        `x_tilts`, `sample2levelled` and `levelled2tomo`, and updates
        `self.device`. `image_indices` stays on the CPU and the
        `local_shifts`/`local_shifts_2d` callables are left untouched.

        Parameters
        ----------
        device : torch.device | str
            Target device.
        """
        self.device = device
        self.tilt_angles = self.tilt_angles.to(device)
        self.tilt_axis_angle = self.tilt_axis_angle.to(device)
        self.sample_translations = self.sample_translations.to(device)
        self.x_tilts = self.x_tilts.to(device)
        self.sample2levelled = self.sample2levelled.to(device)
        self.levelled2tomo = self.levelled2tomo.to(device)

    def project_points(
        self,
        points_zyx: torch.Tensor,
        output_zyxw: bool = False,
    ) -> torch.Tensor:
        """Project 3D points to 2D detector coordinates, both in Angstroms.

        - points are 3D zyx coordinates, in Angstroms, positions relative to
          the center of tomogram space (see `sample2tomo`)
        - projected 2D points are in Angstroms, relative to the center of
          the detector
        - if `self.local_shifts` is set, it is called with the sample-space
          points (n_points, 3), in Angstroms, after tomo2sample:
          tilt-independent, applied once, before projection (e.g. for local
          sample deformation/warping)
        - if `self.local_shifts_2d` is set, it is called with the projected
          points (n_points, n_tilts, 2 or 4), in Angstroms, after
          projection: per-tilt, applied once per tilt (e.g. for per-tilt
          image alignment refinement)
        - output_zyxw, if True, skips dropping the z row: returns
          (n_points, n_tilts, 4) zyxw instead of (n_points,
          n_tilts, 2) yx. z here is scope-space depth (Rz leaves it
          untouched), so this is enough to invert exactly back to sample
          space via `scope2sample @ detector2scope`.

        Parameters
        ----------
        points_zyx : torch.Tensor
            `(n_points, 3)` zyx coordinates in tomogram space, in Angstroms,
            relative to the tomogram centre. Converted to float32 on
            `self.device`.
        output_zyxw : bool
            If False (default), return only the detector yx coordinates. If
            True, return homogeneous zyxw coordinates including scope-space
            depth.

        Returns
        -------
        projected : torch.Tensor
            `(n_points, n_tilts, 2)` yx detector coordinates in Angstroms,
            relative to the detector centre, or `(n_points, n_tilts, 4)` zyxw
            if `output_zyxw` is True.
        """
        points_zyx = torch.as_tensor(_writable(points_zyx), device=self.device).float()

        # tomogram space -> sample space (identity by default: no-op)
        points_zyxw = homogenise_coordinates(points_zyx)  # (n_points, 4)
        points_zyxw = points_zyxw @ self.tomo2sample.T  # unbatched (4, 4), not per-tilt
        points_zyx = points_zyxw[..., :3]

        if self.local_shifts is not None:
            points_zyx = points_zyx + self.local_shifts(points_zyx)

        # Apply projection matrices
        M = self.projection_matrices
        if not output_zyxw:
            M = M[..., [1, 2], :]
        points_zyxw = homogenise_coordinates(points_zyx)
        projected = M @ einops.rearrange(
            points_zyxw, "nparticles zyxw -> nparticles 1 zyxw 1"
        )
        projected = einops.rearrange(
            projected, "nparticles ntilts c 1 -> nparticles ntilts c"
        )
        if self.local_shifts_2d is not None:
            projected = projected + self.local_shifts_2d(projected)
        return projected  # (points, tilts, yx) or (points, tilts, zyxw)
