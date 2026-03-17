# Release Instructions

## Prerequisites
Ensure you have push access to the repository and are on an up-to-date main branch:

```bash
git checkout main
git pull upstream main
```

## Single Package Release

To release a single package, tag it with the pattern `package-name@vX.Y.Z`:

```bash
git tag package-name@v3.4.5
git push upstream main --follow-tags
```

**What happens next:**

1. The `Deploy` workflow triggers automatically
2. CI verification ensures tests passed on main
3. The package is built and published to PyPI
4. A GitHub Release is created with the built artifacts

## Coordinated Release (All Packages)

To release all packages in the workspace with the same version, run the coordinated tagging script locally.

```bash
cd path/to/teamtomo
./scripts/coordinated_release.sh v3.4.5
```

**What happens next:**

1. The script validates that no tags already exist for the version
2. Tags `teamtomo@vX.Y.Z` and every package `package-name@vX.Y.Z`
3. Pushes the tags to `upstream`
4. Individual `Deploy` workflows trigger for each package
5. All packages are built and published to PyPI simultaneously

**Note:** The coordinated release approach is recommended when you want to ensure version consistency across all packages in the TeamTomo workspace.
