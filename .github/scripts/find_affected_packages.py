import subprocess
import json
import os
import sys


def run(cmd):
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
        return result
    except subprocess.CalledProcessError as e:
        # Log error to stderr so it shows up in GH Action logs but doesn't break JSON output
        print(f"Error running command '{cmd}': {e.output.decode()}", file=sys.stderr)
        return ""


def get_affected_packages():
    # 1. Get all packages in the workspace via uv
    workspace_raw = run("uv workspace list")
    if not workspace_raw:
        return []

    packages = {}
    for line in workspace_raw.splitlines():
        # uv output format: name @ file:///path/to/pkg
        if " @ file://" in line:
            name, path = line.split(" @ file://")
            packages[name.strip()] = os.path.relpath(path.strip(), os.getcwd())

    # 2. Determine the base for comparison
    # For PRs: compare against the target branch (main)
    # For Push/Tag: compare against the previous commit (HEAD~)
    base_ref = os.getenv("GITHUB_BASE_REF")
    if not base_ref:
        base_ref = "HEAD~1"

    changed_files = run(f"git diff --name-only {base_ref}...HEAD").splitlines()

    # 3. Identify packages with direct file changes
    affected = set()
    for pkg_name, pkg_path in packages.items():
        for f in changed_files:
            if f.startswith(pkg_path):
                affected.add(pkg_name)
                break

    # 4. Handle Global Overrides (Lockfile, Config, CI changes)
    global_triggers = ["uv.lock", "pyproject.toml", ".python-version", ".github/workflows/ci.yml"]
    if any(f in changed_files for f in global_triggers):
        print("Global configuration change detected. Testing all packages.", file=sys.stderr)
        return list(packages.keys())

    # 5. Handle Tag-Force Logic
    ref_type = os.getenv("GITHUB_REF_TYPE")
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    if ref_type == "tag" and "@v" in ref_name:
        tagged_pkg = ref_name.split("@")[0]
        if tagged_pkg in packages:
            print(f"Tag detected for {tagged_pkg}. Forcing inclusion.", file=sys.stderr)
            affected.add(tagged_pkg)

    # 6. Expand to Dependents (Reverse Dependency Tree)
    # If A changes, and B depends on A, we must test B.
    final_affected = set(affected)
    for pkg in affected:
        # --reverse gives us the consumers of the package
        rev_deps = run(f"uv tree --package {pkg} --reverse --depth 1")
        for line in rev_deps.splitlines():
            # Filter tree characters to find package names
            # Example line: "└── pkg-b v0.1.0"
            parts = line.replace("│", "").replace("├", "").replace("─", "").replace("└", "").split()
            if parts:
                dep_name = parts[0]
                if dep_name in packages:
                    final_affected.add(dep_name)

    return sorted(list(final_affected))


if __name__ == "__main__":
    try:
        # find packages affected by changes
        affected_list = get_affected_packages()

        # If no changes detected but we are on a branch, default to empty (standard CI)
        # If the script fails to find anything but it's a critical run, you could return packages.keys()

        # format GitHub Actions Output
        print(f"packages={json.dumps(affected_list)}")
    except Exception as e:
        print(f"Discovery script failed: {e}", file=sys.stderr)
        # Fail-safe: Output ALL packages so CI doesn't skip tests on error
        workspace_raw = subprocess.check_output("uv workspace list", shell=True).decode().splitlines()
        all_pkgs = [l.split(" @ ")[0].strip() for l in workspace_raw]
        print(f"packages={json.dumps(all_pkgs)}")