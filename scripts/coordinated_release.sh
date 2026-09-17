#!/bin/bash
#
# Cut a coordinated release of every TeamTomo package.
#
# Pushes one tag, teamtomo@vX.Y.Z, which is the only trigger for the Deploy workflow.
# Deploy builds and tests every package and publishes them only if all of them pass.
# The per-package tags are created by CI afterwards, recording what actually shipped.
#
# Usage:
#   ./scripts/coordinated_release.sh v0.6.0
#   ./scripts/coordinated_release.sh v0.6.0 --dry-run

set -euo pipefail

VERSION="${1:-}"
MODE="${2:-}"

# --- Validate arguments -----------------------------------------------------------

# Display usage if no version given
if [[ -z "$VERSION" ]]; then
    echo "Usage: $0 vX.Y.Z [--dry-run]" >&2
    exit 1
fi

# Validate that no additional arguments appear in the command
if [[ -n "$MODE" && "$MODE" != "--dry-run" ]]; then
    echo "ERROR: unknown option '$MODE' (the only option is --dry-run)" >&2
    exit 1
fi

# Validate the version string format to meet PEP 440 and release requirements.
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]]; then
    echo "ERROR: version must look like v1.2.3, v1.2.3rc1, v1.2.3a1 or v1.2.3b1" >&2
    exit 1
fi

TAG="teamtomo@${VERSION}"

# --- Run the pre-flight checks ----------------------------------------------------

if ! python3 .github/scripts/release_preflight.py "${VERSION#v}"; then
    echo
    echo "Nothing has been tagged. Fix the problems above and try again." >&2
    exit 1
fi

if [[ "$MODE" == "--dry-run" ]]; then
    echo
    echo "Dry run: stopping before tagging."
    exit 0
fi

# --- Confirm ----------------------------------------------------------------------

echo
read -rp "Create and push '$TAG'? (y/n) " -n 1
echo

if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# --- Tag and push -----------------------------------------------------------------

git tag -a "$TAG" -m "Release $TAG"

if ! git push upstream "$TAG"; then
    # Leaving the local tag behind would make a retry fail on "tag already exists".
    git tag -d "$TAG"
    echo "ERROR: push failed; the local tag was removed so you can retry." >&2
    exit 1
fi

echo
echo "Pushed $TAG. Follow the release at:"
echo "  https://github.com/teamtomo/teamtomo/actions/workflows/deploy.yml"
