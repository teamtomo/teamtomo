#!/bin/bash
set -eou pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 vX.Y.Z" >&2
    exit 1
fi

VERSION="$1"

if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([A-Za-z0-9._-]+)?$ ]]; then
    echo "ERROR: version must match vX.Y.Z with optional suffix (e.g., v1.2.3, v1.2.3rc1)" >&2
    exit 1
fi

# Validate branch and remote sync state
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "ERROR: current branch is '$CURRENT_BRANCH', expected 'main'" >&2
    exit 1
fi

UPSTREAM_URL=$(git remote get-url upstream 2>/dev/null || true)
if [[ "$UPSTREAM_URL" != "https://github.com/teamtomo/teamtomo.git" ]]; then
    echo "ERROR: remote 'upstream' points to '$UPSTREAM_URL', expected 'https://github.com/teamtomo/teamtomo.git'" >&2
    exit 1
fi

git fetch upstream main
LOCAL_HASH=$(git rev-parse HEAD)
UPSTREAM_HASH=$(git rev-parse upstream/main)
if [[ "$LOCAL_HASH" != "$UPSTREAM_HASH" ]]; then
    echo "ERROR: local branch is not up to date with upstream/main" >&2
    echo "  Local:    $LOCAL_HASH" >&2
    echo "  Upstream: $UPSTREAM_HASH" >&2
    exit 1
fi

TAG="teamtomo@${VERSION}"

# Check tag doesn't already exist locally or on upstream
if git rev-parse --verify "$TAG" >/dev/null 2>&1; then
    echo "ERROR: tag '$TAG' already exists locally" >&2
    exit 1
fi
if git ls-remote --tags upstream "$TAG" | grep -q "$TAG"; then
    echo "ERROR: tag '$TAG' already exists on upstream" >&2
    exit 1
fi

read -rp "Create and push '$TAG'? CI will tag all packages and publish to PyPI. (y/n) " -n 1
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

git tag -a "$TAG" -m "Release $TAG"
git push upstream "$TAG"

echo "OK: '$TAG' pushed. CI will fan out individual package tags and handle all releases."
