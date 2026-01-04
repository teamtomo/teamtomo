import subprocess
import json
import os
import sys


def run(cmd):
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
        return result
    except subprocess.CalledProcessError as e:
        # We print to stderr so it doesn't mess up the GITHUB_OUTPUT
        print(f"Command failed: {cmd}\n{e.output.decode()}", file=sys.stderr)
        return ""


def get_affected_packages():
    # 1. Map package names to relative paths
    workspace_raw = run("uv workspace list")
    if not workspace_raw:
        return []

    packages = {}
    for line in workspace_raw.splitlines():
        if " @ file://" in line:
            name, path = line.split(" @ file://")
            packages[name.strip()] = os.path.relpath(path.strip(), os.getcwd())

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
    affected_names = set()
    for pkg_name, pkg_path in packages.items():
        if any(f.startswith(pkg_path) for f in changed_files):
            affected_names.add(pkg_name)

    # 4. Global Overrides
    global_triggers = ["uv.lock", "pyproject.toml", ".python-version", ".github/workflows/ci.yml"]
    if any(f in changed_files for f in global_triggers):
        print("Global change detected. Testing all.", file=sys.stderr)
        affected_names = set(packages.keys())

    # 5. Tag-Force Logic
    ref_type = os.getenv("GITHUB_REF_TYPE")
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    if ref_type == "tag" and "@v" in ref_name:
        tagged_pkg = ref_name.split("@")[0]
        if tagged_pkg in packages:
            affected_names.add(tagged_pkg)

    # 6. Expand to Dependents
    final_names = set(affected_names)
    for pkg in affected_names:
        rev_deps = run(f"uv tree --package {pkg} --reverse --depth 1")
        for line in rev_deps.splitlines():
            parts = line.replace("│", "").replace("├", "").replace("─", "").replace("└", "").split()
            if parts:
                dep_name = parts[0]
                if dep_name in packages:
                    final_names.add(dep_name)

    return [{"name": name, "path": packages[name]} for name in sorted(final_names)]


if __name__ == "__main__":
    try:
        affected = get_affected_packages()
        print(f"packages={json.dumps(affected)}")
    except Exception as e:
        print(f"Discovery script failed: {e}", file=sys.stderr)
        print("Falling back to testing everything...", file=sys.stdout)
        # Fallback to everything
        all_pkgs = []
        raw = subprocess.check_output("uv workspace list", shell=True).decode().splitlines()
        for l in raw:
            if " @ file://" in l:
                n, p = l.split(" @ file://")
                all_pkgs.append({"name": n.strip(), "path": os.path.relpath(p.strip(), os.getcwd())})
        print(f"packages={json.dumps(all_pkgs)}")