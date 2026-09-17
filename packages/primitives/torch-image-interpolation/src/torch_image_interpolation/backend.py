"""Select the implementation behind the sampling / insertion functions.

Two backends exist:

* ``"torch"`` -- the pure-PyTorch reference implementation (``grid_sample`` /
  ``index_put_``); always available.
* ``"mojo"`` -- fused Mojo kernels (CPU: multithreaded; GPU: one thread per
  sample). Requires the optional ``mojo`` extra
  (``pip install 'torch-image-interpolation[mojo]'``) and supports float32 /
  complex64 data.

The default, ``"auto"``, uses the Mojo kernels whenever they are installed and
the inputs are supported, and silently falls back to torch otherwise. The
initial value can be set with the ``TORCH_IMAGE_INTERPOLATION_BACKEND``
environment variable.

>>> import torch_image_interpolation as tii
>>> tii.set_backend("torch")  # force the reference implementation
>>> with tii.use_backend("mojo"):  # temporarily require the Mojo kernels
...     ...
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

Backend = Literal["auto", "torch", "mojo"]
_VALID: tuple[str, ...] = ("auto", "torch", "mojo")

_backend: str = os.environ.get("TORCH_IMAGE_INTERPOLATION_BACKEND", "auto").lower()
if _backend not in _VALID:
    raise ValueError(
        f"TORCH_IMAGE_INTERPOLATION_BACKEND must be one of {_VALID}, got {_backend!r}"
    )


def get_backend() -> str:
    """Current backend setting: ``"auto"``, ``"torch"`` or ``"mojo"``."""
    return _backend


def set_backend(name: Backend) -> None:
    """Set the backend for subsequent calls (process-wide).

    ``"mojo"`` raises at call time if the kernels are unavailable or an input
    is unsupported, instead of falling back.
    """
    global _backend
    if name not in _VALID:
        raise ValueError(f"backend must be one of {_VALID}, got {name!r}")
    _backend = name


@contextmanager
def use_backend(name: Backend) -> Iterator[None]:
    """Temporarily select a backend within a ``with`` block."""
    previous = get_backend()
    set_backend(name)
    try:
        yield
    finally:
        set_backend(previous)  # type: ignore[arg-type]


def mojo_available(device: str | torch.device = "cpu") -> bool:
    """Return True if the Mojo kernels compile and load for ``device``.

    The first call for a device kind triggers the (cached) compile.
    """
    from ._mojo_backend import kernels_available

    if torch.device(device).type == "cpu":
        return kernels_available("cpu")
    return kernels_available("gpu")
