# Release Instructions

## How releases work

A release is driven by one tag. Pushing `teamtomo@vX.Y.Z` starts the `Deploy` workflow, which:

1. Resolves the tag, checks it is on `main`, and waits for all three CI legs to pass.
2. Builds and tests **every** publishable package, verifying each artifact's version and metadata.
3. Stops at a gate. If any package failed, **nothing is published**.
4. Publishes every component package to PyPI, then the `teamtomo` meta-package.
5. Creates the per-package `<package>@vX.Y.Z` tags and GitHub Releases.

Per-package tags are records and only created after a successful upload, so a tag existing means that version really did ship.

## Coordinated release (all packages)

```bash
git checkout main
git pull upstream main

./scripts/coordinated_release.sh v0.6.0 --dry-run   # check without tagging
./scripts/coordinated_release.sh v0.6.0
```

The script refuses to tag unless the following conditions are met:

- Current working tree is clean,
- `HEAD` matches `upstream/main` (with remote `upstream` referencing teamtomo/teamtomo),
- All three CI legs are green,
- No tags for the desired version already exist,
- No package declares a direct URL dependency, and
- Every package exists on PyPI with the target version still free.

## Single package release

For an independent patch release (see `package-versioning-policy.md`), push that package's tag directly:

```bash
git tag -a torch-ctf@v0.6.1 -m "Release torch-ctf@v0.6.1"
git push upstream torch-ctf@v0.6.1
```

`Deploy` runs in single-package mode: it builds, tests and publishes only that package.

## Before a release

Run the preflight and fix anything it reports:

```bash
# On demand, from the Actions tab: "Release Check", with the target version.
# Or locally:
python3 .github/scripts/release_preflight.py 0.6.0
```

**Rehearse final releases with a release candidate first.**
Cut `vX.Y.Zrc1`, confirm every package landed, then cut `vX.Y.Z`.
A release candidate exercises the identical code path against real PyPI, and pre-releases are ignored by default `pip install`.

### Adding a new package

A package that has never been published needs a **pending publisher** registered on
PyPI *before* its first release, otherwise its first upload fails with a 403 that
`skip-existing` will not mask.
At <https://pypi.org/manage/account/publishing/>:

| Field | Value |
| --- | --- |
| Owner | `teamtomo` |
| Repository | `teamtomo` |
| Workflow | `deploy.yml` |
| Environment | *(leave blank)* |

The preflight lists every package that still needs this.

### Holding a package back

Set the following in that package's `pyproject.toml`.
It will still be tested, but will not be tagged or published:

```toml
[tool.teamtomo]
publish = false
```

If you hold back a package that the root `teamtomo` meta-package depends on, remove it
from the root `dependencies` too, or `pip install teamtomo` will not resolve.

## CITATION.cff

The author list is refreshed by the `Update CITATION authors` workflow, which runs monthly and opens a pull request.
It is deliberately **not** part of the release path: it used to run mid-release and could abort a release before any package was tagged.
