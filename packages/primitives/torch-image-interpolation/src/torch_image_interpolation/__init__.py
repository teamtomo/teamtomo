"""Sample from and insert into 1D/2D/3D images at arbitrary coordinates."""

from .image_interpolation_1d import insert_into_image_1d, sample_image_1d
from .image_interpolation_2d import insert_into_image_2d, sample_image_2d
from .image_interpolation_3d import insert_into_image_3d, sample_image_3d

__all__ = [
    "insert_into_image_1d",
    "insert_into_image_2d",
    "insert_into_image_3d",
    "sample_image_1d",
    "sample_image_2d",
    "sample_image_3d",
]
