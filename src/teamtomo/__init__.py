"""Modular Python packages for cryo-EM and cryo-ET."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("teamtomo")
except PackageNotFoundError:
    __version__ = "uninstalled"

# Import sub-namespaces
from teamtomo import primitives, wip

__all__ = [
    "__version__",
    "primitives",
    "wip",
]


def hello() -> str:
    return "Hello from teamtomo!"
