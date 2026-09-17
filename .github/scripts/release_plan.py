"""Work out what a release tag means, and which packages it covers.

Given a tag, this decides:

* which package and version it refers to,
* whether it is a coordinated release (``teamtomo@vX.Y.Z``) or a single-package one,
* which packages to build, and which to publish.

Results are written to ``$GITHUB_OUTPUT`` when running under GitHub Actions, and
printed either way.

Usage::

    python3 .github/scripts/release_plan.py torch-so3@v0.6.0
"""

import argparse
import json
import os
import re
import subprocess
import sys

from get_all_packages import get_all_packages

META_PACKAGE = "teamtomo"

# Deliberately strict: only already-normalised PEP 440 release versions. The old
# pattern accepted things like v1.2.3-foo, which built fine and was rejected by PyPI
# only at upload time, after every tag had already been pushed.
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+((a|b|rc)\d+)?$")


def parse_tag(tag: str) -> tuple[str, str]:
    """Split ``<package>@v<version>`` into its package name and version."""
    if "@" not in tag:
        sys.exit(f"ERROR: tag '{tag}' is not of the form <package>@v<version>")

    package, _, version = tag.partition("@")
    version = version.removeprefix("v")

    if not VERSION_PATTERN.match(version):
        sys.exit(
            f"ERROR: '{version}' (from tag '{tag}') is not a normalised PEP 440 version"
        )

    return package, version


def resolve_commit(tag: str) -> str:
    """Find the commit a tag points at."""
    result = subprocess.run(
        ["git", "rev-list", "-n", "1", f"refs/tags/{tag}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"ERROR: tag '{tag}' not found in this checkout")
    return result.stdout.strip()


def select_packages(
    package: str, publishable: list[str]
) -> tuple[list[str], list[str]]:
    """Return the packages to build and the packages to publish."""
    if package == META_PACKAGE:
        return publishable, [name for name in publishable if name != META_PACKAGE]

    if package not in publishable:
        sys.exit(
            f"ERROR: '{package}' is not a publishable workspace member. "
            "It is either missing, or sets [tool.teamtomo] publish = false."
        )

    return [package], [package]


def write_outputs(outputs: dict[str, str]) -> None:
    """Print the plan, and expose it to later workflow steps."""
    for key, value in outputs.items():
        print(f"{key}={value}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            for key, value in outputs.items():
                f.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="the release tag, e.g. torch-so3@v0.6.0")
    args = parser.parse_args()

    package, version = parse_tag(args.tag)
    commit = resolve_commit(args.tag)
    publishable = list(get_all_packages(publishable_only=True))
    build, publish = select_packages(package, publishable)

    write_outputs(
        {
            "tag_name": args.tag,
            "package": package,
            "version": version,
            "sha": commit,
            "coordinated": "true" if package == META_PACKAGE else "false",
            "build_packages": json.dumps(build),
            "publish_packages": json.dumps(publish),
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
