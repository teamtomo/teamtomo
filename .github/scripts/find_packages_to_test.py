import subprocess
import json
import os
import sys
from pathlib import Path


def log(msg):
    print(msg, file=sys.stderr)


def run(cmd):
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
        return result
    except subprocess.CalledProcessError as e:
        # We print to stderr so it doesn't mess up the GITHUB_OUTPUT
        print(f"Command failed: {cmd}\n{e.output.decode()}", file=sys.stderr)
        return ""


def get_workspace_packages() -> list[dict[str, str]]:  # pkg_name: relative_path
    package_names = run("uv workspace list --preview-features workspace-list").splitlines()
    package_paths = run("uv workspace list --paths --preview-features workspace-list").splitlines()
    if not package_names:
        return []
    packages = [
        {
            "name": name,
            "path": Path(abs_path),
        }
        for name, abs_path
        in zip(package_names, package_paths)
    ]
    return packages


def get_affected_packages() -> list[dict[str, str]]:  # pkg_name: path:
    workspace_packages = get_workspace_packages()
    if not workspace_packages:
        log("no packages found")
        return []
    log(
        f"found following packages in workspace: {
        [
            f"{pkg["name"]} @ {pkg['path']}\n"
            for pkg in workspace_packages
        ]
        }"
    )

    # 1. Map package names to relative paths
    package_name_to_path = {
        package["name"]: package["path"]
        for package
        in workspace_packages
    }

    # 2. FIX: Determine base for comparison
    base_ref = os.getenv("GITHUB_BASE_REF")

    if base_ref:
        # If in a PR, compare against the remote tracking branch
        comparison = f"origin/{base_ref}...HEAD"
    else:
        # If on main or a tag, compare against the previous commit
        # We check if HEAD~1 exists (fails on first commit of a repo)
        has_parent = run("git rev-parse --verify HEAD~1")
        comparison = "HEAD~1...HEAD" if has_parent else "HEAD"

    print(f"Running diff: git diff --name-only {comparison}", file=sys.stderr)
    changed_files = run(f"git diff --name-only {comparison}").splitlines()

    # 3. Identify packages with direct changes
    modified_package_names = set()
    for pkg_name, pkg_path in package_name_to_path.items():
        if any(f.startswith(pkg_path) for f in changed_files):
            modified_package_names.add(pkg_name)

    # 4. Global Overrides
    global_triggers = ["uv.lock", "pyproject.toml", ".python-version", ".github/workflows/ci.yml"]
    if any(f in changed_files for f in global_triggers):
        print("Global change detected. Testing all.", file=sys.stderr)
        modified_package_names = set(package_name_to_path.keys())

    # 5. Tag-Force Logic
    ref_type = os.getenv("GITHUB_REF_TYPE")
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    if ref_type == "tag" and "@v" in ref_name:
        tagged_pkg = ref_name.split("@")[0]
        if tagged_pkg in package_name_to_path:
            modified_package_names.add(tagged_pkg)

    # 6. Expand Packages to Test list to Include Dependents
    packages_to_test = set(modified_package_names)
    for pkg in modified_package_names:
        rev_deps = run(f"uv tree --package {pkg} --reverse --depth 1")
        for line in rev_deps.splitlines():
            parts = line.replace("│", "").replace("├", "").replace("─", "").replace("└", "").split()
            if parts:
                dep_name = parts[0]
                if dep_name in package_name_to_path:
                    packages_to_test.add(dep_name)

    return [{"name": name, "path": package_name_to_path[name]} for name in sorted(packages_to_test)]


if __name__ == "__main__":
    try:
        affected = get_affected_packages()
        print(f"packages={json.dumps(affected)}")
    except Exception as e:
        all_packages = get_workspace_packages()
        print(f"packages={json.dumps(all_packages)}")
