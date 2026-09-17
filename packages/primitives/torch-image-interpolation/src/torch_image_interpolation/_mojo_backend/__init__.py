"""Mojo-accelerated implementation of the sampling / insertion primitives.

The kernels live in ``torch_image_interpolation/_mojo/`` and are compiled on
first use via ``mojo.importer`` (see :mod:`._kernels`). The public functions in
:mod:`torch_image_interpolation` dispatch here when the selected backend (see
:mod:`torch_image_interpolation.backend`) allows it and the inputs are supported
(see :func:`.should_use_mojo`).
"""

from ._api import insert_into_image, sample_image, should_use_mojo
from ._kernels import kernels_available

__all__ = [
    "insert_into_image",
    "kernels_available",
    "sample_image",
    "should_use_mojo",
]
