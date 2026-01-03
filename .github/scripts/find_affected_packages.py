import subprocess
import json
import os
import sys

def run(cmd):
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8").strip()
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{cmd}': {e.output.decode()}", file=sys.stderr)
        return ""

def get_affected_packages():
    # 1. Map package names to relative paths using uv workspace list
    workspace_raw = run("uv workspace list")
    if not workspace_raw:
        return []

    packages = {}
    for line in workspace_raw.splitlines():
        if " @ file://" in line:
            name, path = line.split(" @ file://")
            # Store the relative path (e.g., packages/primitives/torch-grid-utils)
            packages[name.strip()] = os.path.relpath(path.strip(), os.getcwd())

    # 2. Determine base for comparison
    base_ref = os.getenv("GITHUB_BASE_REF", "HEAD~1")
    changed_files = run(f"git diff --name-only {base_ref}...HEAD").splitlines()

    # 3. Identify packages with direct changes
    affected_names = set()
    for pkg_name, pkg_path in packages.items():
        if any(f.startswith(pkg_path) for f in changed_files):
            affected_names.add(pkg_name)

    # 4. Global Overrides (Lockfile, Config, CI changes)
    global_triggers = ["uv.lock", "pyproject.toml", ".python-version", ".github/workflows/ci.yml"]
    if any(f in changed_files for f in global_triggers):
        print("Global change detected. Testing all packages.", file=sys.stderr)
        affected_names = set(packages.keys())

    # 5. Tag-Force Logic
    ref_type = os.getenv("GITHUB_REF_TYPE")
    ref_name = os.getenv("GITHUB_REF_NAME", "")
    if ref_type == "tag" and "@v" in ref_name:
        tagged_pkg = ref_name.split("@")[0]
        if tagged_pkg in packages:
            print(f"Tag detected for {tagged_pkg}. Forcing inclusion.", file=sys.stderr)
            affected_names.add(tagged_pkg)

    # 6. Expand to Dependents (Reverse Dependency Tree)
    final_names = set(affected_names)
    for pkg in affected_names:
        rev_deps = run(f"uv tree --package {pkg} --reverse --depth 1")
        for line in rev_deps.splitlines():
            # Filter tree characters to find package names
            parts = line.replace("│", "").replace("├", "").replace("─", "").replace("└", "").split()
            if parts:
                dep_name = parts[0]
                if dep_name in packages:
                    final_names.add(dep_name)

    # 7. Format output as list of objects: [{"name": "pkg", "path": "path"}]
    return [{"name": name, "path": packages[name]} for name in sorted(final_names)]

if __name__ == "__main__":
    try:
        affected_objects = get_affected_packages()
        print(f"packages={json.dumps(affected_objects)}")
    except Exception as e:
        print(f"Discovery script failed: {e}", file=sys.stderr)
        # Fail-safe: Output ALL packages as objects so CI doesn't skip
        workspace_raw = subprocess.check_output("uv workspace list", shell=True).decode().splitlines()
        all_pkgs = []
        for l in workspace_raw:
            if " @ file://" in l:
                n, p = l.split(" @ file://")
                all_pkgs.append({"name": n.strip(), "path": os.path.relpath(p.strip(), os.getcwd())})
        print(f"packages={json.dumps(all_pkgs)}")