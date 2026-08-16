# Release Instructions

## Prerequisites

### Place yourself on the main branch

Ensure you have push access to the repository and are on an up-to-date main branch:

```bash
git checkout main
git pull upstream main
```

## Single Package Release

Push an annotated tag matching `package-name@vX.Y.Z`. The `Deploy` workflow triggers automatically on the tag push.

```bash
git tag -a package-name@v3.4.5 -m "Release package-name@v3.4.5"
git push upstream package-name@v3.4.5
```

**What happens next:**

1. The `Deploy` workflow triggers on the tag push
2. CI verification ensures tests passed on main
3. The package is built and published to PyPI
4. A GitHub Release is created with the built artifacts attached

## Coordinated Release (All Packages)

To release all packages in the workspace at the same version, run the coordinated release script:

```bash
cd path/to/teamtomo
./scripts/coordinated_release.sh vX.Y.Z
```

The script validates branch state, creates the `teamtomo@vX.Y.Z` tag, and pushes it. CI handles everything from there.

**What happens next:**

1. The `Coordinate Release` workflow triggers on the `teamtomo@vX.Y.Z` tag push
2. It waits for CI to pass on that commit
3. It updates `CITATION.cff` with the latest contributor list and commits to main
4. It creates and pushes individual `package-name@vX.Y.Z` tags for every workspace package, one at a time
5. Each tag push triggers an individual `Deploy` workflow run
6. Each `Deploy` run verifies CI, builds the package, publishes to PyPI, and creates a GitHub Release

**Note:** The `CITATION.cff` update happens automatically as part of step 3 — you do not need to run `update_citation_authors.py` manually before a coordinated release.
