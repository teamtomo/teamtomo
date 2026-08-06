"""Experimental Mojo-backed kernels for torch-fourier-slice.

The same Fourier-slice operators as ``torch_fourier_slice``, with the compute
kernels written in Mojo (1.0.0b2) and exposed to Python via Mojo's Python
interop. APIs here are experimental and may change without notice.

Importing this package eagerly compiles + loads every Mojo kernel module (via
``mojo.importer``); use :func:`mojo_kernels_available` to check whether that
succeeded on this system.

Each operator follows its input tensor's device: a CPU tensor runs the
multithreaded CPU kernel, an ``mps`` / ``cuda`` tensor runs the GPU kernel, and
the output comes back on that device.

**Real-space layer** -- real volumes and images in, real volumes and images out,
mirroring ``torch_fourier_slice``'s own :func:`~torch_fourier_slice.project_3d_to_2d`
/ :func:`~torch_fourier_slice.backproject_2d_to_3d`. These handle the padding,
the FFTs and the gridding correction for you:

- :func:`project_3d_to_2d` / :func:`backproject_2d_to_3d` (+ ``_multivolume``).

**Fourier layer** -- rfft in, rfft out, in **rfft layout with DC at the origin**
(see the layout note in ``README.md`` for the one-``fftshift`` bridge to the
canonical layout). Every operator is an extraction (gather) or its adjoint
insertion (scatter), each in a single- and a ``_multivolume`` rank form:

- **central slices**, posed by a rotation matrix: 3D volume <-> 2D slices, via
  :func:`extract_central_slices_rfft_3d` / :func:`insert_central_slices_rfft_3d`.
- **central lines**, posed by a direction vector: 3D volume <-> 1D lines, via
  :func:`extract_central_line_rfft_3d` / :func:`insert_central_line_rfft_3d`;
  and 2D image <-> 1D lines, via :func:`extract_central_line_rfft_2d` /
  :func:`insert_central_line_rfft_2d`.

All are differentiable w.r.t. their volume/slice/line data, their pose
(rotations or directions), their 2D / 3D shifts, and -- for the insertions --
their weights.
"""

# importing _kernels triggers the eager compile + load of all Mojo modules
from ._kernels import mojo_kernels_available
from .backproject import (
    backproject_2d_to_3d,
    backproject_2d_to_3d_multivolume,
)
from .line_extraction import (
    extract_central_line_rfft_3d,
    extract_central_line_rfft_3d_multivolume,
)
from .line_extraction_2d import (
    extract_central_line_rfft_2d,
    extract_central_line_rfft_2d_multivolume,
)
from .line_insertion import (
    insert_central_line_rfft_3d,
    insert_central_line_rfft_3d_multivolume,
)
from .line_insertion_2d import (
    insert_central_line_rfft_2d,
    insert_central_line_rfft_2d_multivolume,
)
from .project import project_3d_to_2d, project_3d_to_2d_multivolume
from .slice_extraction import (
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multivolume,
)
from .slice_insertion import (
    insert_central_slices_rfft_3d,
    insert_central_slices_rfft_3d_multivolume,
)

__all__ = [
    "backproject_2d_to_3d",
    "backproject_2d_to_3d_multivolume",
    "extract_central_line_rfft_2d",
    "extract_central_line_rfft_2d_multivolume",
    "extract_central_line_rfft_3d",
    "extract_central_line_rfft_3d_multivolume",
    "extract_central_slices_rfft_3d",
    "extract_central_slices_rfft_3d_multivolume",
    "insert_central_line_rfft_2d",
    "insert_central_line_rfft_2d_multivolume",
    "insert_central_line_rfft_3d",
    "insert_central_line_rfft_3d_multivolume",
    "insert_central_slices_rfft_3d",
    "insert_central_slices_rfft_3d_multivolume",
    "mojo_kernels_available",
    "project_3d_to_2d",
    "project_3d_to_2d_multivolume",
]
