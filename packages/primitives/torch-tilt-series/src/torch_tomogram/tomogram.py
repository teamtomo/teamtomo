"""Tomogram reconstruction in pytorch."""

import einops
import torch
import torch.nn.functional as F
from torch_affine_utils import homogenise_coordinates
from torch_affine_utils.transforms_3d import Ry, Rz, T
from torch_fourier_slice import backproject_2d_to_3d
from torch_grid_utils import dft_center
from torch_subpixel_crop import subpixel_crop_2d


class Tomogram:
    """Tomogram class that enables reconstruction position querying."""

    def __init__(
        self,
        tilt_angles: torch.Tensor,
        tilt_axis_angle: torch.Tensor,
        sample_translations: torch.Tensor,
        images: torch.Tensor,  # (b, h, w)
        device: torch.device | str = "cpu",
    ):
        self.images = torch.as_tensor(images, device=device).float()
        self.tilt_angles = torch.as_tensor(tilt_angles, device=device).float()
        self.tilt_axis_angle = torch.as_tensor(tilt_axis_angle, device=device).float()
        self.sample_translations = torch.as_tensor(
            sample_translations, device=device
        ).float()
        self.device = device
        self._pad_factor = 2.0

    @property
    def projection_matrices(self) -> torch.Tensor:
        """Matrices that project points from 3D -> 2D."""
        shifts_3d = F.pad(self.sample_translations, (1, 0), value=0)
        r0 = Ry(self.tilt_angles, zyx=True, device=self.device)
        r1 = Rz(self.tilt_axis_angle, zyx=True, device=self.device)
        t2 = T(shifts_3d, device=self.device)
        return t2 @ r1 @ r0

    def to(self, device: torch.device | str) -> None:
        """Move all objects of the tomogram to the device."""
        self.device = device
        self.images = self.images.to(device)
        self.tilt_angles = self.tilt_angles.to(device)
        self.tilt_axis_angle = self.tilt_axis_angle.to(device)
        self.sample_translations = self.sample_translations.to(device)

    def project_points(self, points_zyx: torch.Tensor) -> torch.Tensor:
        """Project points from 3D to 2D.

        - points are 3D zyx coordinates
        - points are positions relative to center of tomogram
        - projected 2D points are relative to center of 2D image
        """
        points_zyx = torch.as_tensor(points_zyx, device=self.device).float()
        M_yx = self.projection_matrices[..., [1, 2], :]  # (ntilts, 2, 4)
        points_zyxw = homogenise_coordinates(points_zyx)
        projected_yx = M_yx @ einops.rearrange(
            points_zyxw, "nparticles zyxw -> nparticles 1 zyxw 1"
        )
        projected_yx = einops.rearrange(
            projected_yx, "nparticles ntilts yx 1 -> nparticles ntilts yx"
        )
        return projected_yx  # (points, tilts, yx)

    def extract_particle_tilt_series(
        self, points_zyx: torch.Tensor, sidelength: int
    ) -> torch.Tensor:
        """Extract a subtilt-series at a 3D location in the sample."""
        projected_yx = self.project_points(points_zyx)
        projected_yx += dft_center(
            self.images.shape[-2:], rfft=False, fftshift=True, device=self.device
        )
        images = subpixel_crop_2d(
            image=self.images,
            positions=projected_yx,
            sidelength=sidelength,
        )
        return images

    def reconstruct_subvolume(
        self, point_zyx: torch.Tensor, sidelength: int
    ) -> torch.Tensor:
        """Reconstruct a subvolume at a 3D location in the sample."""
        point_zyx = torch.as_tensor(point_zyx, device=self.device).float()
        point_zyx = point_zyx.reshape((-1, 3))
        rotation_matrices = self.projection_matrices[:, :3, :3]
        rotation_matrices = torch.linalg.pinv(rotation_matrices)
        sidelength_padded = int(self._pad_factor * sidelength)
        particle_tilt_series = self.extract_particle_tilt_series(
            point_zyx, sidelength=sidelength_padded
        )
        volume = backproject_2d_to_3d(
            images=particle_tilt_series[0],
            rotation_matrices=rotation_matrices,
            zyx_matrices=True,
            pad_factor=1.0,  # we already incorporate padding in subtilts
            fftfreq_max=0.5,
        )
        p = (sidelength_padded - sidelength) // 2
        volume = F.pad(volume, [-p] * 6)  # remove padding
        return volume

    def reconstruct_tomogram(
        self, volume_shape: tuple[int, int, int], sidelength: int
    ) -> torch.Tensor:
        """Reconstruct the full tomogram by tiling the positions in 3D."""
        d, h, w = volume_shape
        r = sidelength // 2

        # setup grid points
        z = torch.arange(start=r, end=d + r, step=sidelength) - d // 2
        y = torch.arange(start=r, end=h + r, step=sidelength) - h // 2
        x = torch.arange(start=r, end=w + r, step=sidelength) - w // 2

        # allocate whole volume
        tomogram = torch.zeros(
            size=volume_shape, dtype=torch.float32, device=self.device
        )

        for _z in z:
            for _y in y:
                for _x in x:
                    zyx = torch.tensor([_z, _y, _x]).float()
                    subvolume = self.reconstruct_subvolume(zyx, sidelength=sidelength)
                    _d, _h, _w = zyx + torch.tensor(volume_shape) // 2
                    _d, _h, _w = int(_d), int(_h), int(_w)
                    d_min, d_max = _d - r, min(_d + r, d)
                    h_min, h_max = _h - r, min(_h + r, h)
                    w_min, w_max = _w - r, min(_w + r, w)
                    d_max_sub = d_max - d_min
                    h_max_sub = h_max - h_min
                    w_max_sub = w_max - w_min
                    tomogram[d_min:d_max, h_min:h_max, w_min:w_max] = subvolume[
                        :d_max_sub, :h_max_sub, :w_max_sub
                    ]
        return tomogram
