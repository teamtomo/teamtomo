"""Torch reference implementation of the experimental Mojo operators.

The same functions as :mod:`torch_fourier_slice.experimental`, with the same
signatures and results, computed with plain torch ops instead of the Mojo
kernels. Gradients come from autograd rather than the dedicated backward kernels.

:mod:`torch_fourier_slice.experimental` falls back to these functions when the
Mojo kernels can't be built or loaded. They also serve as the reference the Mojo
kernels are tested against.

So far this covers the central slices (rfft layer) and the real-space projection
and backprojection built on them; the central lines still need the Mojo kernels.
"""

from .backproject import backproject_2d_to_3d, backproject_2d_to_3d_multichannel
from .extraction import (
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multichannel,
)
from .insertion import (
    insert_central_slices_rfft_3d,
    insert_central_slices_rfft_3d_multichannel,
)
from .project import project_3d_to_2d, project_3d_to_2d_multichannel

__all__ = [
    "backproject_2d_to_3d",
    "backproject_2d_to_3d_multichannel",
    "extract_central_slices_rfft_3d",
    "extract_central_slices_rfft_3d_multichannel",
    "insert_central_slices_rfft_3d",
    "insert_central_slices_rfft_3d_multichannel",
    "project_3d_to_2d",
    "project_3d_to_2d_multichannel",
]
