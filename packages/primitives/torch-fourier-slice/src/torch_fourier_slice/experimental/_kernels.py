"""Loading and compilation of the Mojo extension module.

The Mojo kernels live in a single module (`_mojo/projectors.mojo`) exposed to
Python via Mojo's documented ``mojo.importer`` hook: importing ``projectors``
compiles it with ``mojo build --emit shared-lib`` (cached in
``_mojo/__mojocache__/``) and loads it.

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
        import projectors

    return projectors


try:  # eager compile + load on first import of this module
    _MODULE = _load()
except Exception as exc:
    _LOAD_ERROR = exc


def kernels():
    """Return the loaded Mojo ``projectors`` module."""
    if _LOAD_ERROR is not None:
        raise ImportError(
            "experimental Mojo kernels failed to load. Install the optional "
            "'mojo' package: pip install 'mojo==1.0.0b2' --prerelease allow"
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
