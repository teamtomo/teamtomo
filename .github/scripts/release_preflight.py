"""Pre-release checks for a coordinated TeamTomo release.

Each check returns the problems it found. Nothing is tagged or published from here;
this only reports. `scripts/coordinated_release.sh` runs it before tagging, and
`release-check.yml` runs it on demand.

Usage::

    python3 .github/scripts/release_preflight.py 0.6.0
    python3 .github/scripts/release_preflight.py 0.6.0 --no-git
"""

import argparse
import json
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from get_all_packages import get_all_packages

UPSTREAM_REPO = "https://github.com/teamtomo/teamtomo.git"
CANONICAL_REPO = "https://github.com/teamtomo/teamtomo"
GITHUB_REPO = "teamtomo/teamtomo"
CI_JOB_PREFIX = "Test All Packages"
EXPECTED_CI_LEGS = 3
PYPI_JSON = "https://pypi.org/pypi/{package}/json"
PUBLISHING_URL = "https://pypi.org/manage/account/publishing/"
PRE_RELEASE = re.compile(r"(a|b|rc)\d+$")


@dataclass
class Check:
    """The outcome of one group of related checks."""

    name: str
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems


def git(*args: str) -> str:
    """Run a git command and return its stdout, or "" if it failed."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def find_canonical_remote() -> str | None:
    """Return the name of the remote pointing at teamtomo/teamtomo, if there is one."""

    def normalise(url: str) -> str:
        url = url.strip().replace("git@github.com:", "https://github.com/")
        return url.removesuffix(".git").rstrip("/")

    for name in git("remote").splitlines():
        if normalise(git("remote", "get-url", name)) == CANONICAL_REPO:
            return name
    return None


def check_repository_state() -> Check:
    """The release must be cut from a clean main that matches upstream."""
    check = Check("Repository state")

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "main":
        check.notes.append("on branch main")
    else:
        check.problems.append(f"on branch '{branch}', expected 'main'")

    remote = git("remote", "get-url", "upstream")
    if remote == UPSTREAM_REPO:
        check.notes.append(f"remote 'upstream' is {UPSTREAM_REPO}")
    else:
        check.problems.append(
            f"remote 'upstream' is '{remote or 'unset'}', expected {UPSTREAM_REPO}"
        )

    # A dirty tree is invisible to the tag, so uncommitted work silently does not ship.
    dirty = git("status", "--porcelain")
    if dirty:
        check.problems.append("working tree has uncommitted changes")
        check.problems.extend(f"    {line}" for line in dirty.splitlines())
    else:
        check.notes.append("working tree is clean")

    subprocess.run(["git", "fetch", "--quiet", "upstream", "main"], check=False)
    subprocess.run(
        ["git", "fetch", "--quiet", "upstream", "--tags", "--prune"], check=False
    )

    local = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "upstream/main")
    if local and local == upstream:
        check.notes.append(f"HEAD matches upstream/main ({local[:12]})")
    else:
        check.problems.append(
            f"HEAD ({local[:12]}) does not match upstream/main ({upstream[:12]})"
        )

    return check


def check_ci_status(commit: str) -> Check:
    """All three CI legs must be green on the exact commit being released."""
    check = Check("Continuous integration")

    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{GITHUB_REPO}/commits/{commit}/check-runs?filter=latest",
            "--paginate",
            "--jq",
            f'.check_runs[] | select(.name | startswith("{CI_JOB_PREFIX}")) | "\\(.name)=\\(.conclusion)"',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        check.problems.append(f"could not query CI status ({result.stderr.strip()})")
        return check

    legs = sorted(set(result.stdout.split()))
    green = [leg for leg in legs if leg.endswith("=success")]

    if len(green) == EXPECTED_CI_LEGS:
        check.notes.append(f"all {EXPECTED_CI_LEGS} CI legs green on {commit[:12]}")
    else:
        check.problems.append(
            f"expected {EXPECTED_CI_LEGS} green CI legs, found {len(green)}"
        )
        check.problems.extend(f"    {leg}" for leg in legs)
        # ci.yml cancels in-progress runs, so a rapid second push can leave an
        # earlier commit's checks 'cancelled' rather than failed.
        check.problems.append("    re-run CI on this commit if any leg is 'cancelled'")

    return check


def check_tags_are_free(version: str, packages: list[str]) -> Check:
    """A leftover tag is not harmless: CI skips tags that already exist."""
    check = Check("Git tags")

    remote_tags = {
        line.split("\t")[-1].removeprefix("refs/tags/").removesuffix("^{}")
        for line in git("ls-remote", "--tags", "upstream").splitlines()
    }

    taken = []
    for package in packages:
        tag = f"{package}@v{version}"
        locally = bool(git("rev-parse", "-q", "--verify", f"refs/tags/{tag}"))
        if locally or tag in remote_tags:
            taken.append(tag)

    if taken:
        check.problems.append(f"{len(taken)} tag(s) for v{version} already exist")
        check.problems.extend(f"    {tag}" for tag in taken)
    else:
        check.notes.append(f"no tags for v{version} exist yet")

    return check


def check_no_direct_references(packages: dict[str, Path]) -> Check:
    """PyPI rejects any Requires-Dist carrying a direct URL reference."""
    check = Check("Package metadata")

    offenders = []
    for name, directory in packages.items():
        with open(directory / "pyproject.toml", "rb") as f:
            project = tomllib.load(f).get("project", {})

        # Inspect declared dependencies rather than the allow-direct-references key:
        # some packages set that key without actually using a direct reference.
        dependencies = list(project.get("dependencies", []))
        for extra in (project.get("optional-dependencies") or {}).values():
            dependencies.extend(extra)

        for dependency in dependencies:
            if " @ " in dependency:
                offenders.append(f"{name}: {dependency}")

    if offenders:
        check.problems.append(
            f"{len(offenders)} direct URL dependency(ies) — PyPI will reject these"
        )
        check.problems.extend(f"    {offender}" for offender in offenders)
    else:
        check.notes.append("no direct URL dependencies")

    return check


def fetch_pypi_releases(package: str) -> set[str] | None:
    """Return the versions on PyPI, or None if the project does not exist."""
    try:
        with urllib.request.urlopen(
            PYPI_JSON.format(package=package), timeout=30
        ) as response:
            return set(json.load(response)["releases"])
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def check_pypi(version: str, packages: list[str]) -> Check:
    """Every project must exist on PyPI, and the target version must be free."""
    check = Check("PyPI")

    unpublished = []
    version_taken = []
    for package in packages:
        releases = fetch_pypi_releases(package)
        if releases is None:
            unpublished.append(package)
        elif version in releases:
            version_taken.append(package)
        time.sleep(1.2)  # attempt to avoid rate-limit response

    if unpublished:
        # A first upload via trusted publishing fails with 403 unless a pending
        # publisher was registered beforehand. skip-existing does not mask a 403.
        check.problems.append(
            f"{len(unpublished)} package(s) not on PyPI and need a pending publisher"
        )
        check.problems.extend(f"    {package}" for package in unpublished)
        check.problems.append(f"    register at {PUBLISHING_URL}")
        check.problems.append(
            "    owner=teamtomo repo=teamtomo workflow=deploy.yml environment=(blank)"
        )
    else:
        check.notes.append("every package already exists on PyPI")

    if version_taken:
        # Versions on PyPI are immutable. skip-existing would turn these into silent
        # no-ops and the release would go green having shipped nothing.
        check.problems.append(
            f"{len(version_taken)} package(s) already have {version} on PyPI"
        )
        check.problems.extend(f"    {package}" for package in version_taken)
    else:
        check.notes.append(f"version {version} is free on PyPI for every package")

    return check


def check_release_candidate(version: str) -> Check:
    """Final releases should be rehearsed with a release candidate first."""
    check = Check("Rehearsal")

    if PRE_RELEASE.search(version):
        check.notes.append(f"{version} is itself a pre-release")
        return check

    if git("tag", "-l", f"teamtomo@v{version}rc*"):
        check.notes.append(f"a release candidate for {version} was published first")
    else:
        check.warnings.append(
            f"no teamtomo@v{version}rc* tag exists — consider cutting v{version}rc1 first"
        )

    return check


def report(checks: list[Check], label: str = "Pre-flight") -> bool:
    """Print each check's outcome. Returns True if everything passed."""
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}")
        for note in check.notes:
            print(f"       {note}")
        for warning in check.warnings:
            print(f"       WARNING: {warning}")
        for problem in check.problems:
            print(f"       {problem}")

    failed = [check for check in checks if not check.passed]
    print()
    if failed:
        print(f"{label} FAILED: {len(failed)} of {len(checks)} checks did not pass.")
        return False
    print(f"{label} passed: {len(checks)} of {len(checks)} checks.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version", help="target version without a leading v (e.g. 0.6.0)"
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="skip repository and CI checks (for running inside CI)",
    )
    args = parser.parse_args()

    packages = get_all_packages(publishable_only=True)
    if not packages:
        print(
            "ERROR: no publishable packages found — run from the repository root",
            file=sys.stderr,
        )
        return 1

    names = list(packages)
    print(f"Checking {len(names)} publishable packages for version {args.version}\n")

    checks = []
    if not args.no_git:
        checks.append(check_repository_state())
        checks.append(check_ci_status(git("rev-parse", "HEAD")))
        checks.append(check_tags_are_free(args.version, names))
    checks.append(check_no_direct_references(packages))
    checks.append(check_pypi(args.version, names))
    if not args.no_git:
        checks.append(check_release_candidate(args.version))

    return 0 if report(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
