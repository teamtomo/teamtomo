"""Lazy compilation + loading of the Mojo extension modules.

Two extension modules are built from ``torch_image_interpolation/_mojo/``:

* ``image_interpolation_cpu`` -- CPU kernels (``parallelize`` over samples).
* ``image_interpolation_gpu`` -- GPU kernels (one thread per sample) plus the
  ``DeviceSession`` holding the process-wide ``DeviceContext``.

They are separate on purpose: the GPU module needs a working GPU compiler
(e.g. Apple's Metal Toolchain), and a failure there must not take the CPU
acceleration down with it. Each module is compiled on its first use (cached in
``_mojo/__mojocache__/``) rather than at package import, since this package is
a low-level dependency of many others and must import cleanly without Mojo.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import warnings
from typing import Literal

_MOJO_DIR = pathlib.Path(__file__).resolve().parent.parent / "_mojo"
_MODULE_NAMES = {"cpu": "image_interpolation_cpu", "gpu": "image_interpolation_gpu"}

_modules: dict[str, object] = {}
_errors: dict[str, Exception] = {}


def _load(kind: Literal["cpu", "gpu"]) -> object:
    name = _MODULE_NAMES[kind]
    if name in _modules:
        return _modules[name]
    if name in _errors:
        raise ImportError(
            f"Mojo kernels '{name}' failed to load earlier in this process"
        ) from _errors[name]
    try:
        import mojo.importer  # noqa: F401  (installs the .mojo import hook)

        if str(_MOJO_DIR) not in sys.path:
            sys.path.insert(0, str(_MOJO_DIR))
        # Mojo's PythonModuleBuilder.add_type creates types without __module__,
        # which CPython >= 3.12 flags with a DeprecationWarning at load time.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*has no __module__ attribute",
                category=DeprecationWarning,
            )
            module = importlib.import_module(name)
    except Exception as exc:  # compile failure, missing `mojo`, ...
        _errors[name] = exc
        raise ImportError(
            f"Mojo kernels '{name}' failed to compile / load. Install the optional "
            "'mojo' extra (pip install 'torch-image-interpolation[mojo]') and, for "
            "GPU kernels, a working GPU compiler toolchain."
        ) from exc
    _modules[name] = module
    return module


def cpu_kernels() -> object:
    """The loaded ``image_interpolation_cpu`` module (compiles on first call)."""
    return _load("cpu")


def gpu_kernels() -> object:
    """The loaded ``image_interpolation_gpu`` module (compiles on first call)."""
    return _load("gpu")


def kernels_available(kind: Literal["cpu", "gpu"] = "cpu") -> bool:
    """Return True if the ``kind`` Mojo kernels compile and load on this system.

    The first call for each ``kind`` triggers the (cached) compile.
    """
    try:
        _load(kind)
    except ImportError:
        return False
    return True


def load_error(kind: Literal["cpu", "gpu"]) -> Exception | None:
    """The exception that made ``kind`` unavailable, if any."""
    return _errors.get(_MODULE_NAMES[kind])


_SESSION: object | None = None


def device_session() -> object:
    """Process-wide cached GPU ``DeviceSession`` (one ``DeviceContext``).

    A fresh ``DeviceContext`` per call leaks the underlying Metal command
    queue; one shared session avoids that.
    """
    global _SESSION
    if _SESSION is None:
        _SESSION = gpu_kernels().DeviceSession()  # type: ignore[attr-defined]
    return _SESSION
