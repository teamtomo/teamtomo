"""Sample from and insert into 1D/2D/3D images at arbitrary coordinates."""

from .backend import get_backend, mojo_available, set_backend, use_backend
from .image_interpolation_1d import insert_into_image_1d, sample_image_1d
from .image_interpolation_2d import insert_into_image_2d, sample_image_2d
from .image_interpolation_3d import insert_into_image_3d, sample_image_3d

__all__ = [
    "get_backend",
    "insert_into_image_1d",
    "insert_into_image_2d",
    "insert_into_image_3d",
    "mojo_available",
    "sample_image_1d",
    "sample_image_2d",
    "sample_image_3d",
    "set_backend",
    "use_backend",
]
