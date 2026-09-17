"""Record a release that has already been published to PyPI.

Creates one ``<package>@v<version>`` tag per package and a GitHub Release for each.
Runs *after* publishing, so a tag existing means that version really shipped.

Both steps skip anything that already exists, so re-running is safe.

Usage::

    python3 .github/scripts/record_release.py \\
        --version 0.6.0 --commit <sha> --packages '["torch-so3", ...]' \\
        --artifacts-dir artifacts
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BOT_NAME = "github-actions[bot]"
BOT_EMAIL = "github-actions[bot]@users.noreply.github.com"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def tag_exists(tag: str) -> bool:
    return run("git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode == 0


def release_exists(tag: str) -> bool:
    return run("gh", "release", "view", tag).returncode == 0


def create_tags(packages: list[str], version: str, commit: str) -> list[str]:
    """Create any missing tags locally and push them in one atomic push."""
    run("git", "config", "user.name", BOT_NAME)
    run("git", "config", "user.email", BOT_EMAIL)

    new_tags = []
    for package in packages:
        tag = f"{package}@v{version}"
        if tag_exists(tag):
            print(f"  tag exists, skipping: {tag}")
            continue

        created = run("git", "tag", "-a", tag, commit, "-m", f"Release {tag}")
        if created.returncode != 0:
            sys.exit(f"ERROR: could not create tag {tag}: {created.stderr.strip()}")
        new_tags.append(tag)

    if not new_tags:
        print("  all tags already present")
        return []

    pushed = run("git", "push", "--atomic", "origin", *new_tags)
    if pushed.returncode != 0:
        sys.exit(f"ERROR: atomic tag push failed: {pushed.stderr.strip()}")

    print(f"  pushed {len(new_tags)} tag(s)")
    return new_tags


def create_releases(packages: list[str], version: str, artifacts_dir: Path) -> None:
    """Create a GitHub Release per package, attaching that package's artifacts."""
    for package in packages:
        tag = f"{package}@v{version}"
        if release_exists(tag):
            print(f"  release exists, skipping: {tag}")
            continue

        files = sorted((artifacts_dir / f"dist-{package}").glob("*"))
        if not files:
            print(
                f"  WARNING: no artifacts found for {package}, creating release without files"
            )

        created = run(
            "gh",
            "release",
            "create",
            tag,
            "--title",
            tag,
            "--generate-notes",
            *[str(path) for path in files],
        )
        if created.returncode != 0:
            sys.exit(f"ERROR: could not create release {tag}: {created.stderr.strip()}")

        print(f"  created release: {tag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="released version, e.g. 0.6.0")
    parser.add_argument(
        "--commit", required=True, help="commit the tags should point at"
    )
    parser.add_argument("--packages", required=True, help="JSON array of package names")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    packages = json.loads(args.packages)

    print(f"Tagging {len(packages)} package(s) at v{args.version}")
    create_tags(packages, args.version, args.commit)

    print(f"\nCreating GitHub Releases for {len(packages)} package(s)")
    create_releases(packages, args.version, args.artifacts_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
