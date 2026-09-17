"""Discover the workspace packages that participate in tests and releases.

Modes
-----
``get_all_packages.py``
    Print every workspace package name, one per line, sorted.
``get_all_packages.py --publishable``
    As above, but skip packages opted out via ``[tool.teamtomo] publish = false``.
``get_all_packages.py --json [--publishable]``
    Print ``[{"name": ..., "path": ...}, ...]`` for consumption by ``fromJSON()``
    in a GitHub Actions matrix.
``get_all_packages.py <package-name>``
    Print that package's directory, or exit 1 if it is not a workspace member.
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT_PYPROJECT = Path("pyproject.toml")


def _member_patterns() -> list[str]:
    """Derive the glob patterns from the root pyproject's uv workspace members."""
    with open(ROOT_PYPROJECT, "rb") as f:
        data = tomllib.load(f)

    members = data.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", [])
    return ["pyproject.toml"] + [f"{member}/pyproject.toml" for member in members]


def get_all_packages(publishable_only: bool = False) -> dict[str, Path]:
    """Get workspace package names and their directories, sorted by name."""
    workspace_packages = {}

    for pattern in _member_patterns():
        for pyproject in Path(".").glob(pattern):
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)

            pkg_name = data.get("project", {}).get("name")
            if not pkg_name:
                continue

            if publishable_only:
                publish = data.get("tool", {}).get("teamtomo", {}).get("publish", True)
                if not publish:
                    continue

            workspace_packages[pkg_name] = pyproject.parent

    return dict(sorted(workspace_packages.items()))


def find_package_path(package_name: str) -> Path:
    """Find the workspace directory for a given package name."""
    packages = get_all_packages()
    if package_name not in packages:
        print(
            f"ERROR: Package '{package_name}' not found in workspace", file=sys.stderr
        )
        print(f"Available packages: {', '.join(packages.keys())}", file=sys.stderr)
        sys.exit(1)
    return packages[package_name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package_name",
        nargs="?",
        help="if given, print this package's directory instead of listing all packages",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON objects with name and path (for a GitHub Actions matrix)",
    )
    parser.add_argument(
        "--publishable",
        action="store_true",
        help="skip packages that set [tool.teamtomo] publish = false",
    )
    args = parser.parse_args()

    if args.package_name:
        print(find_package_path(args.package_name))
        return

    packages = get_all_packages(publishable_only=args.publishable)

    if args.json:
        print(
            json.dumps(
                [{"name": name, "path": str(path)} for name, path in packages.items()]
            )
        )
    else:
        for pkg_name in packages:
            print(pkg_name)


if __name__ == "__main__":
    main()
