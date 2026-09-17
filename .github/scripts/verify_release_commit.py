"""Check that a commit is fit to release from.

Two conditions, both of which have let bad releases through before:

* the commit must be on the canonical repository's ``main`` -- a tag pointing anywhere
  else, or a comparison against a fork's main, must not publish;
* all three CI legs must be green on that exact commit.

Usage::

    python3 .github/scripts/verify_release_commit.py <sha>
"""

import argparse
import subprocess
import sys

from release_preflight import (
    CANONICAL_REPO,
    Check,
    check_ci_status,
    find_canonical_remote,
    report,
)


def check_commit_is_on_main(commit: str) -> Check:
    """The commit must be an ancestor of main in the canonical repository."""
    check = Check("Commit is on main")

    remote = find_canonical_remote()
    if remote is None:
        check.problems.append(f"no git remote points at {CANONICAL_REPO}")
        return check

    subprocess.run(["git", "fetch", "--quiet", remote, "main"], check=False)
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, f"{remote}/main"],
        check=False,
    )

    if is_ancestor.returncode == 0:
        check.notes.append(f"{commit[:12]} is an ancestor of {remote}/main")
    else:
        check.problems.append(f"{commit[:12]} is not an ancestor of {remote}/main")

    return check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit", help="the commit the release tag points at")
    args = parser.parse_args()

    checks = [
        check_commit_is_on_main(args.commit),
        check_ci_status(args.commit),
    ]
    return 0 if report(checks, label="Release commit check") else 1


if __name__ == "__main__":
    sys.exit(main())
