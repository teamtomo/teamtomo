import sys
import tomllib
from pathlib import Path

# Define workspace member glob patterns (same as pyproject.toml)
PATTERNS = [
    "packages/io/*/pyproject.toml",
    "packages/primitives/*/pyproject.toml",
    "packages/algorithms/*/pyproject.toml",
    "packages/utils/*/pyproject.toml",
    "packages/wip/*/pyproject.toml",
]


def get_all_packages() -> dict[str, Path]:
    """Get all workspace package names and their directories."""
    workspace_packages = {}

    for pattern in PATTERNS:
        for pyproject in Path(".").glob(pattern):
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
                pkg_name = data.get("project", {}).get("name")
                if pkg_name:
                    workspace_packages[pkg_name] = pyproject.parent

    return dict(sorted(workspace_packages.items()))


def find_package_path(package_name: str) -> Path:
    """Find the workspace directory for a given package name."""
    packages = get_all_packages()
    if package_name not in packages:
        print(f"ERROR: Package '{package_name}' not found in workspace", file=sys.stderr)
        print(f"Available packages: {', '.join(packages.keys())}", file=sys.stderr)
        sys.exit(1)
    return packages[package_name]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If a package name is provided, print its path
        path = find_package_path(sys.argv[1])
        print(path)
    else:
        # Otherwise, print all package names
        for pkg_name in get_all_packages():
            print(pkg_name)
