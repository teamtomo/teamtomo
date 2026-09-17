"""Verify built distributions before they are uploaded to PyPI.

Catches the two classes of upload rejection this repo has actually hit:

* a direct URL reference in ``Requires-Dist`` (PyPI returns 400), and
* an artifact whose version is not the one the release tag asked for -- which
  ``skip-existing: true`` would otherwise hide by silently skipping the upload.

Usage::

    python3 .github/scripts/check_artifacts.py --dist-dir dist --expect-version 0.6.0
"""

import argparse
import re
import sys
import tarfile
import zipfile
from email.parser import HeaderParser
from pathlib import Path

# A distribution filename encodes the version, but metadata is authoritative --
# read Version out of METADATA (wheel) or PKG-INFO (sdist).
WHEEL_METADATA = re.compile(r"^[^/]+\.dist-info/METADATA$")


def _wheel_metadata(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if WHEEL_METADATA.match(n))
        return zf.read(name).decode("utf-8")


def _sdist_metadata(path: Path) -> str:
    with tarfile.open(path, "r:gz") as tf:
        # PKG-INFO sits at <name>-<version>/PKG-INFO
        member = next(
            m
            for m in tf.getmembers()
            if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")
        )
        extracted = tf.extractfile(member)
        assert extracted is not None
        return extracted.read().decode("utf-8")


def check(path: Path, expect_version: str) -> list[str]:
    """Return a list of problems found in one distribution file."""
    if path.suffix == ".whl":
        raw = _wheel_metadata(path)
    elif path.name.endswith(".tar.gz"):
        raw = _sdist_metadata(path)
    else:
        return [f"{path.name}: unrecognised artifact type"]

    metadata = HeaderParser().parsestr(raw)
    problems = []

    version = metadata.get("Version")
    if version != expect_version:
        problems.append(
            f"{path.name}: Version is {version!r}, expected {expect_version!r}"
        )

    # PyPI rejects any Requires-Dist carrying a direct URL reference (PEP 440
    # direct references). Checking the built metadata rather than pyproject.toml
    # avoids false positives from packages that merely set
    # `allow-direct-references = true` without using it.
    for requirement in metadata.get_all("Requires-Dist") or []:
        if " @ " in requirement:
            problems.append(
                f"{path.name}: direct reference in Requires-Dist: {requirement}"
            )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", default="dist", type=Path)
    parser.add_argument("--expect-version", required=True)
    args = parser.parse_args()

    if not args.dist_dir.is_dir():
        print(
            f"ERROR: {args.dist_dir} does not exist (did the build step run?)",
            file=sys.stderr,
        )
        return 1

    artifacts = sorted(
        p
        for p in args.dist_dir.iterdir()
        if p.suffix == ".whl" or p.name.endswith(".tar.gz")
    )
    if not artifacts:
        print(f"ERROR: no artifacts found in {args.dist_dir}", file=sys.stderr)
        return 1

    problems = []
    for artifact in artifacts:
        found = check(artifact, args.expect_version)
        status = "FAIL" if found else "ok"
        print(f"  [{status}] {artifact.name}")
        problems.extend(found)

    if problems:
        sys.stdout.flush()
        print("\nERROR: artifact verification failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(
        f"\nAll {len(artifacts)} artifact(s) verified at version {args.expect_version}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
