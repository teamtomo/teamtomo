#!/bin/bash
set -eou pipefail

# Check only one argument (version number) provided
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 vX.Y.Z" >&2
  exit 1
fi

# Check that version matches expected format (vX.Y.Z)
VERSION="$1"
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "ERROR: version must match vX.Y.Z (e.g., v1.2.3)" >&2
  exit 1
fi

# Check the following:
# 1. current branch is main
# 2. remote name 'upstream' points to teamtomo/teamtomo
# 3. local branch is up to date with upstream/main
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
  echo "Local HEAD: $LOCAL_HASH" >&2
  echo "Upstream HEAD: $UPSTREAM_HASH" >&2
  exit 1
fi

# Fetch all package names from repo helper
PACKAGES=$(uv run python .github/scripts/get_all_packages.py)
if [[ -z "$PACKAGES" ]]; then
  echo "ERROR: no packages found" >&2
  exit 1
fi

TAGS=("teamtomo@${VERSION}")
for pkg in $PACKAGES; do
  TAGS+=("${pkg}@${VERSION}")
done

# Check for existing tags (local + remote)
EXISTING=()
for tag in "${TAGS[@]}"; do
  if git rev-parse --verify "$tag" >/dev/null 2>&1; then
    EXISTING+=("$tag")
    continue
  fi
  if git ls-remote --tags origin "$tag" | grep -q "$tag"; then
    EXISTING+=("$tag")
  fi
done

if [[ ${#EXISTING[@]} -gt 0 ]]; then
  echo "ERROR: tags already exist: ${EXISTING[*]}" >&2
  exit 1
fi

# Create annotated tags
for tag in "${TAGS[@]}"; do
  git tag -a "$tag" -m "Release $tag"
done

# Wait for user confirmation before pushing.
# If cancel, then delete the local tags.
read -p "Tags created: ${TAGS[*]}. Push to origin? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborting. Deleting local tags: ${TAGS[*]}"
  for tag in "${TAGS[@]}"; do
    git tag -d "$tag"
  done
  exit 1
fi

# # Push all tags
# git push origin --tags

echo "OK: Coordinated release tags created and pushed"