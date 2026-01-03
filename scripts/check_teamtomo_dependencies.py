import tomllib
from pathlib import Path
import sys


def get_all_packages():
    """Finds all pyproject.toml files and extracts relevant metadata."""
    packages = {}
    repo_root = Path(__file__).parent.parent

    # Find all subpackages
    for p in repo_root.glob("**/pyproject.toml"):
        with open(p, "rb") as f:
            data = tomllib.load(f)
            name = data.get("project", {}).get("name")
            if not name:
                continue

            # Try to get version (dynamic via hatch-vcs or static)
            version = data.get("project", {}).get("version", "0.0.0-dynamic")

            packages[name] = {
                "path": p,
                "version": version,
                "dependencies": data.get("project", {}).get("dependencies", []),
            }
    return packages


def check_dependencies():
    packages = get_all_packages()
    metapackage = packages.get("teamtomo")

    all_package_names = set(packages.keys())
    workspace_package_names = all_package_names - {"teamtomo"}

    print(f"--- Checking teamtomo metapackage dependencies ---\n")

    # 1. Check for packages missing from teamtomo metapackage
    teamtomo_deps = {
        dep.split(">")[0].split("=")[0].split("<")[0].strip(): dep
        for dep
        in metapackage["dependencies"]
    }
    not_in_teamtomo = workspace_package_names - set(teamtomo_deps.keys())

    if not_in_teamtomo:
        print("⚠️  Missing from 'teamtomo' dependency list:")
        for package_name in not_in_teamtomo:
            print(f"   - {package_name}")
        print()

    else:
        print("✅️ All workspace packages are in 'teamtomo' package dependencies")
        sys.exit(1)


if __name__ == "__main__":
    check_dependencies()