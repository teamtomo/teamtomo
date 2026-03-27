"""Very brief description of your package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("your-package-name")
except PackageNotFoundError:
    __version__ = "uninstalled"
