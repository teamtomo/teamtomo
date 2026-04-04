# Release Instructions

## Prerequisites

### Place yourself on the main branch

Ensure you have push access to the repository and are on an up-to-date main branch:

```bash
git checkout main
git pull upstream main
```

### Check that the authors in `CITATION.cff` are up to date

Before making a new release, run the following command to update the `CITATION.cff` file with the latest list of contributors:

```bash
uv run python scripts/update_citation_authors.py
```

## Single Package Release

To release a single package, create a GitHub Release on the tag matching `package-name@vX.Y.Z`:

```bash
# Create the tag
git tag -a package-name@v3.4.5 -m "Release package-name@v3.4.5"
git push upstream main --follow-tags

# Create a GitHub release for the tag
gh release create package-name@v3.4.5 \
  --title "package-name@v3.4.5" \
  --notes "Release package-name@v3.4.5"
```

For a pre-release, add the `--prerelease` flag:

```bash
gh release create package-name@v3.4.5 \
  --title "package-name@v3.4.5" \
  --notes "Release package-name@v3.4.5" \
  --prerelease
```

**What happens next:**

1. Publishing the release triggers the `Deploy` workflow automatically
2. CI verification ensures tests passed on main
3. The package is built and published to PyPI
4. The release details are populated with built artifacts

## Coordinated Release (All Packages)

To release all packages in the workspace with the same version, run the coordinated release script locally.

For a final release:

```bash
cd path/to/teamtomo
./scripts/coordinated_release.sh v3.4.5
```

For a pre-release (rc, beta, alpha, etc.):

```bash
./scripts/coordinated_release.sh v3.4.5rc1 --prerelease
```

**What the script does:**

1. Validates that no tags already exist for the version
2. Creates annotated tags for `teamtomo@vX.Y.Z` and each package `package-name@vX.Y.Z`
3. Pushes all tags to `upstream`
4. Creates GitHub releases for each tag (marked as pre-release if `--prerelease` is used)
5. Publishing the releases automatically triggers individual `Deploy` workflows for each package
6. All packages are built and published to PyPI simultaneously

**Note:** The coordinated release approach is recommended when you want to ensure version consistency across all packages in the TeamTomo workspace.
