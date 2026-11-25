"""Loading and compilation of the Mojo extension module.

The Mojo kernels live in a single module (`_mojo/fourier_slice_kernels.mojo`)
exposed to Python via Mojo's documented ``mojo.importer`` hook: importing
``fourier_slice_kernels`` compiles it with ``mojo build --emit shared-lib``
(cached in ``_mojo/__mojocache__/``) and loads it.

The module is compiled + loaded **eagerly** when this module is first imported
(i.e. when ``torch_fourier_slice.experimental`` is imported). If the optional
``mojo`` package is missing or the build fails, the error is stored and surfaced
on use; :func:`mojo_kernels_available` reports the status so callers can degrade
gracefully.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

_MOJO_DIR = pathlib.Path(__file__).parent / "_mojo"

_MODULE: object | None = None
_LOAD_ERROR: Exception | None = None


def _load():
    import mojo.importer  # noqa: F401  (installs the .mojo import hook)

    if str(_MOJO_DIR) not in sys.path:
        sys.path.insert(0, str(_MOJO_DIR))
    # Mojo's PythonModuleBuilder.add_type creates the bound type without a
    # __module__, which CPython >=3.12 flags with a DeprecationWarning at load.
    # Suppress it locally so it can't be escalated to an error (e.g. by a test
    # runner's warnings-as-errors filter) and abort the whole extension import.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*has no __module__ attribute",
            category=DeprecationWarning,
        )
        import fourier_slice_kernels

    # A build without a GPU target has no GPU entry points (see PyInit in
    # fourier_slice_kernels.mojo). The importer caches builds by source hash
    # only, so one made on a machine without a GPU (e.g. a cluster login node
    # sharing this environment) can be loaded where a GPU is present.
    if not hasattr(fourier_slice_kernels, "DeviceSession") and _gpu_present():
        raise ImportError(
            "the cached Mojo kernels were built on a machine without a GPU; "
            f"delete {_MOJO_DIR / '__mojocache__'} and import again to rebuild "
            "them for this GPU"
        )
    return fourier_slice_kernels


def _gpu_present() -> bool:
    import torch

    return torch.cuda.is_available() or torch.backends.mps.is_available()


try:  # eager compile + load on first import of this module
    _MODULE = _load()
except Exception as exc:
    _LOAD_ERROR = exc


def kernels():
    """Return the loaded Mojo ``fourier_slice_kernels`` module."""
    if _LOAD_ERROR is not None:
        raise ImportError(
            "experimental Mojo kernels failed to load. Install the optional "
            "'mojo' extra: pip install 'torch-fourier-slice[mojo]'"
        ) from _LOAD_ERROR
    return _MODULE


def mojo_kernels_available() -> bool:
    """Return True if the Mojo kernels compiled and loaded on this system."""
    return _LOAD_ERROR is None


_SESSION: object | None = None


def device_session():
    """Return the process-wide cached GPU ``DeviceSession``.

    All GPU entry points share this one ``DeviceContext`` (Metal command queue).
    Constructing a context per call leaked Metal queues and crashed long
    training loops after ~1000 steps; one cached session avoids that.
    """
    global _SESSION
    if _SESSION is None:
        _SESSION = kernels().DeviceSession()
    return _SESSION
