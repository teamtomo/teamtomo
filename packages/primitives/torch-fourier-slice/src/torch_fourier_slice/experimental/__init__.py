"""Experimental Mojo-backed kernels for torch-fourier-slice.

This namespace hosts the same Fourier-slice operators as ``torch_fourier_slice``,
with their compute kernels written in Mojo (1.0.0b2) and exposed to Python via
Mojo's Python interop. APIs
here are experimental and may change without notice.

Importing this package eagerly compiles + loads every Mojo kernel module (via
``mojo.importer``); use :func:`mojo_kernels_available` to check whether that
succeeded on this system.

The Python API has two levels, mirroring ``torch_fourier_slice``:

- rfft layer (rfft in / rfft out): :func:`extract_central_slices_rfft_3d` (+
  ``_multivolume``) forward, and :func:`insert_central_slices_rfft_3d` (+
  ``_multivolume``) adjoint. One Mojo kernel underneath; differentiable w.r.t. the
  volume/slices, rotations, 2D / 3D shifts, and (insertion) weights.
- real-space layer: :func:`project_3d_to_2d` / :func:`backproject_2d_to_3d_forw`.
"""

# importing _kernels triggers the eager compile + load of all Mojo modules
from ._kernels import mojo_kernels_available
from .backproject import backproject_2d_to_3d_forw
from .project import project_3d_to_2d
from .slice_extraction import (
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multivolume,
)
from .slice_insertion import (
    insert_central_slices_rfft_3d,
    insert_central_slices_rfft_3d_multivolume,
)

__all__ = [
    "backproject_2d_to_3d_forw",
    "extract_central_slices_rfft_3d",
    "extract_central_slices_rfft_3d_multivolume",
    "insert_central_slices_rfft_3d",
    "insert_central_slices_rfft_3d_multivolume",
    "mojo_kernels_available",
    "project_3d_to_2d",
]
