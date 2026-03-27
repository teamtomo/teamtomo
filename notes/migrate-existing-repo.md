# Migrating Existing TeamTomo Repositories to Monorepo

This guide outlines the process for migrating existing standalone TeamTomo GitHub repositories into the monorepo workspace structure. The same should also apply when migrating in a new package not previously part of TeamTomo (e.g. moving personal package into TeamTomo).

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Migration Process](#step-by-step-migration-process)
4. [Before vs After Comparison](#before-vs-after-comparison)
5. [Post-Migration Checklist](#post-migration-checklist)
<!-- 5. [Configuration Changes](#configuration-changes) -->
<!-- 6. [CI/CD Considerations](#cicd-considerations) -->
<!-- 7. [Git History Options](#git-history-options) -->
<!-- 9. [Common Issues and Solutions](#common-issues-and-solutions) -->
<!-- 10. [Example: torch-grid-utils Migration](#example-torch-grid-utils-migration) -->

## Overview

### Purpose

To consolidate existing standalone TeamTomo packages into a unified monorepo workspace for better maintainability, shared tooling, and easier cross-package development. This comes with a few key benefits:

- **Unified versioning**: Coordinated releases across packages
- **Shared tooling**: Single set of dev tools, linters, and CI/CD
- **Easier development**: Cross-package changes in single PR
- **Simplified dependencies**: Workspace packages automatically resolved
- **Consistent standards**: Shared configuration and best practices

### Reference Example

This guide uses the `torch-grid-utils` migration as a reference. The original repository at `teamtomo/torch-grid-utils` was successfully migrated to `packages/primitives/torch-grid-utils/` in the monorepo.

## Prerequisites

This guide assumes you've successfully cloned and set up the `teamtomo/teamtomo` repository locally. Please see [README.md](../README.md) for development installation instructions.

### Before Starting

1. **Identify the repository**
   - GitHub organization/repo name (e.g., `teamtomo/torch-grid-utils`)
   - Current version/release status
   - Active development status

2. **Determine category**
   - `primitives/`: Core data structures, types, arrays
   - `algorithms/`: Processing algorithms, analysis tools
   - `utils/`: (unlikely) Organization utilities unrelated to data processing

3. **Package naming**
   - Confirm package name (usually same as repo name)
   - Check for naming conflicts with existing packages

4. **Review dependencies**
   - List external dependencies
   - Identify any dependencies on other TeamTomo packages

## Step-by-Step Migration Process

### Phase 1: Preparation of old repository for migration

First, clone the existing repository into some directory on your system. Here, we are using `/tmp` as temporary storage, but a different storage location may be more practical

```bash
cd /tmp
git clone https://github.com/teamtomo/<repo-name>.git
cd <repo-name>
```

<!-- **IMPORTANT**: We'll use `git subtree add --squash` which preserves history while keeping it clean. -->

Next, check the structure of and configuration files for the existing repository. Generally, there should be no issues since packages have an `src/` style layout and a well-defined `pyproject.toml` file.

```bash
# Check directory structure
tree -L 2

# Review pyproject.toml
cat pyproject.toml

# Check for special config files
ls -la | grep "^\."
```

The files and directories to **COPY** into the monorepo are:

- `src/` - Entire source directory
- `tests/` - All test files
- `README.md` - Package documentation
- `LICENSE` - License file (should be BSD-3 or MIT)
- `pyproject.toml` - Package dependencies and setup

Files to **SKIP** (not needed in monorepo):

- `.github/` - CI/CD workflows (monorepo handles this)
- `.gitignore` - Use monorepo's root .gitignore
- `.pre-commit-config.yaml` - Use monorepo's root config
- `.copier-answers.yml` - Template metadata
- Any other repo-specific config files

### Phase 2: Directory Setup and File Migration

Navigate to the root of your TeamTomo repository

```bash
cd /path/to/teamtomo
```

And based on the decisions from the prerequisite steps, set the following environment variables for easier re-use. Again, we are using `torch-grid-utils` as an example.

```bash
export PACKAGE_CATEGORY=primitives  # one of {io, primitives, algorithms}
export PACKAGE_NAME=torch-grid-utils  # e.g., torch-grid-utils
mkdir -p packages/${PACKAGE_CATEGORY}/${PACKAGE_NAME}
```

Then run the following commands to copy the relevant files from the original repository into the monorepo structure:

#### Copy files into monorepo

```bash
# Copy from original repo to monorepo
cp -r /tmp/${PACKAGE_NAME}/src packages/${PACKAGE_CATEGORY}/${PACKAGE_NAME}/
cp -r /tmp/${PACKAGE_NAME}/tests packages/${PACKAGE_CATEGORY}/${PACKAGE_NAME}/
cp /tmp/${PACKAGE_NAME}/README.md packages/${PACKAGE_CATEGORY}/${PACKAGE_NAME}/
cp /tmp/${PACKAGE_NAME}/LICENSE packages/${PACKAGE_CATEGORY}/${PACKAGE_NAME}/
cp /tmp/${PACKAGE_NAME}/pyproject.toml packages/${PACKAGE_CATEGORY}/${PACKAGE_NAME}/
```

<!-- ### Phase 3: Clean Up and Configure Package

**1. Remove monorepo-level config files from package**

These are now handled at the root level:

```bash
cd packages/<category>/<package-name>

# Remove package-level configs (use monorepo's instead)
rm -rf .github/              # CI/CD workflows
rm .gitignore                # Use root .gitignore
rm .pre-commit-config.yaml   # Use root pre-commit
rm .copier-answers.yml       # Template metadata (if exists)

# Keep these files:
# - LICENSE (package-specific)
# - README.md (package docs)
# - pyproject.toml (will modify)
# - src/ (source code)
# - tests/ (tests)
``` -->

### Phase 3: Configure Package in the Monorepo

TeamTomo uses a standardized BSD 3-Clause License across all packages. Copy over the root LICENSE file to ensure consistent copyright attribution:

```bash
cp LICENSE packages/${PACKAGE_CATEGORY}/${PACKAGE_NAME}/LICENSE
```

Next, we need to update the `pyproject.toml` file for the package to ensure it is correctly configured for the monorepo environment. This includes setting the versioning scheme, repository URLs, and verifying any coverage configuration.

#### Update hatch versioning scheme

The hatch versioning block either needs added or updated to the following. Note that `<PACKAGE_NAME>` should be replaced with the actual package name (e.g., `torch-grid-utils`):

```toml
# https://hatch.pypa.io/latest/config/metadata/
[tool.hatch.version]
source = "vcs"
tag-pattern = "^<PACKAGE_NAME>@v(?P<version>.+)$"
fallback-version = "0.5.0"

[tool.hatch.version.raw-options]
search_parent_directories = true
# Parse tags of the form: <package-name>@v<semver>
tag_regex = "^<PACKAGE_NAME>@v(?P<version>\\d+\\.\\d+\\.\\d+.*)$"
# Constrain git-describe so it only considers TeamTomo's own tags, not other workspace tags.
# See https://github.com/ofek/hatch-vcs/issues/71
git_describe_command = "git describe --dirty --tags --long --match '<PACKAGE_NAME>@v[0-9]*.[0-9]*.[0-9]*'"
```

#### Update repository URLs

The `homepage` and `repository` fields should point to the monorepo location.

```toml
[project.urls]
homepage = "https://github.com/teamtomo/teamtomo"
repository = "https://github.com/teamtomo/teamtomo"
```

#### Verify coverage source (if present)

Make sure `[tool.coverage.run]` has the correct source:

```toml
[tool.coverage.run]
source = ["<PACKAGE_NAME>"]  # e.g., "torch_grid_utils"
```

#### Review and verify configuration

Check that:

- Package name matches directory structure
- Dependencies are correct
- Python version requirement is compatible (>=3.12 for monorepo)
- Build system uses hatchling
- All monorepo-level configs removed (.github/, .gitignore, .pre-commit-config.yaml)
- LICENSE updated to TeamTomo copyright

### Phase 4: Workspace Integration

#### Register package with uv workspace

```bash
cd /path/to/teamtomo
uv sync
```

This will:

- Discover the new package via glob patterns in root `pyproject.toml`
- Install it as an editable package
- Update `uv.lock`

#### Update root `pyproject.toml` dependency list

Packages should be added as dependencies to the root `pyproject.toml`. Edit the `[project.dependencies]` section to include the new package:

```toml
[project]
name = "teamtomo"
# ... other fields ...
dependencies = [
    # ... other dependencies ...
    "torch-grid-utils",  # Add your package
]
```

#### Update metapackage exports

Packages can be imported as `from teamtomo.PACKAGE_CATEGORY import PACKAGE_NAME` for convenience.

Edit the `src/teamtomo/PACKAGE_CATEGORY/__init__.py` file to re-export the package namespace. For example, for `torch-grid-utils` in the `primitives` category, edit `src/teamtomo/primitives/__init__.py`:

```python
# File: src/teamtomo/primitives/__init__.py
# ... other imports ...

# Add this block
try:
    import torch_grid_utils
except ImportError:
   torch_grid_utils = None

# Add package to __all__ for export
__all__ = [
    # ... other exports
    "torch_grid_utils",
]
```

### Phase 5: Verification

#### Unit tests and imports

Run unit tests for the migrated package to ensure everything is working correctly in the monorepo environment.

```bash
uv run pytest packages/${PACKAGE_CATEGORY}/${PACKAGE_NAME}/tests/

# With coverage
uv run pytest packages/${PACKAGE_CATEGORY}/${PACKAGE_NAME}/tests/ \
    --cov=${PACKAGE_MODULE} --cov-report=term-missing
```

Verify that the package can be imported and used:

```bash
# Check import works
uv run python -c "import <package_module>; print(<package_module>.__version__)"
```

#### Building the package

Finally, check that the package builds correctly:

```bash
uv build packages/<category>/<package-name>/
```

Check that `dist/` contains `.whl` and `.tar.gz` files.

## Before vs After Comparison

### Original Standalone Repository Structure

```
torch-grid-utils/                    # Standalone repo root
├── .github/
│   ├── workflows/
│   │   └── ci.yml                   # Package-specific CI
│   ├── dependabot.yml
│   └── ISSUE_TEMPLATE.md
├── .gitignore                       # Package-specific ignores
├── .pre-commit-config.yaml          # Package-specific hooks
├── .copier-answers.yml              # Template metadata
├── src/
│   └── torch_grid_utils/            # Source code
│       ├── __init__.py
│       └── *.py
├── tests/                           # Test files
│   └── test_*.py
├── pyproject.toml                   # Standalone config
├── README.md
└── LICENSE
```

### Monorepo Structure After Migration

```
teamtomo/                            # Monorepo root
├── .github/                         # Monorepo-level CI (shared)
├── .gitignore                       # Monorepo-level ignores (shared)
├── .pre-commit-config.yaml          # Monorepo-level hooks (shared)
├── LICENSE                          # Root license (source of truth)
├── packages/
│   └── primitives/
│       └── torch-grid-utils/        # Migrated package
│           ├── src/
│           │   └── torch_grid_utils/
│           │       ├── __init__.py
│           │       └── *.py
│           ├── tests/
│           │   └── test_*.py
│           ├── pyproject.toml       # Modified for monorepo
│           ├── README.md
│           └── LICENSE              # Copy of root LICENSE
├── pyproject.toml                   # Workspace configuration
└── uv.lock                          # Unified lockfile
```

**Key differences:**

- No `.github/`, `.gitignore`, or `.pre-commit-config.yaml` in package (uses monorepo's)
- Package LICENSE is copy of root LICENSE (TeamTomo copyright)
- Package lives in `packages/<category>/<name>/`
- Modified `pyproject.toml` with monorepo-specific configuration
- Shared tooling and CI/CD at monorepo level

<!-- ## Configuration Changes

### Version Configuration

Teamtomo uses a dynamic versioning process through `hatch-vcs` to detect git tags with a package's specific version; these git tags follow a scheme of `<package-name>@v<x.y.z>`.
Ensure the `[tool.hatch.version]` and `[tool.hatch.version.raw-options]` tables are updated accordingly:

**Before (standalone):**

```toml
[tool.hatch.version]
source = "vcs"
```

**After (monorepo):**

```toml
[tool.hatch.version]
source = "vcs"
tag-pattern = "^torch-grid-utils@v(?P<version>.+)$"
fallback-version = "0.0.1"

[tool.hatch.version.raw-options]
search_parent_directories = true
# Parse tags of the form: <package-name>@v<semver>
tag_regex = "^torch-grid-utils@v(?P<version>\\d+\\.\\d+\\.\\d+.*)$"
# Constrain git-describe so it only considers TeamTomo's own tags, not other workspace tags.
# See https://github.com/ofek/hatch-vcs/issues/71
git_describe_command = "git describe --dirty --tags --long --match 'torch-grid-utils@v[0-9]*.[0-9]*.[0-9]*'"

```

**Why:**

- `tag-pattern`: Enables package-specific tags (e.g., `torch-grid-utils@v1.0.0`) in monorepo
- `fallback-version`: Provides default when no tags exist
- `search_parent_directories`: Finds `.git` directory in monorepo root

### Repository URLs

**Before (standalone):**

```toml
[project.urls]
homepage = "https://github.com/alisterburt/torch-grids"
repository = "https://github.com/alisterburt/torch-grids"
```

**After (monorepo):**

```toml
[project.urls]
homepage = "https://github.com/teamtomo/teamtomo"
repository = "https://github.com/teamtomo/teamtomo"
```

### Coverage Configuration

**Before (may have incorrect name):**

```toml
[tool.coverage.run]
source = ["torch_grids"]  # Wrong!
```

**After (corrected):**

```toml
[tool.coverage.run]
source = ["torch_grid_utils"]  # Correct package name
```

## CI/CD Considerations

### Monorepo CI/CD Strategy

**Individual package CI is replaced by monorepo-level CI:**

- **Testing**: Monorepo CI runs tests for all packages
- **Linting**: Shared ruff/mypy configuration at root level
- **Coverage**: Aggregated coverage across packages
- **Releases**: Tag-based releases using package-specific tags

### Package-Specific Tags

In the monorepo, each package uses prefixed tags:

```bash
# Tag format: <package-name>@v<version>
git tag torch-grid-utils@v1.0.0
git push origin torch-grid-utils@v1.0.0
```

The `tag-pattern` in `pyproject.toml` extracts the version from these tags.

### Dependabot and Pre-commit

- **Dependabot**: Configured at monorepo root for all packages
- **Pre-commit hooks**: Shared across all packages

## Git History Options

### Option A: Fresh Start (Recommended)

**Approach**: Copy files without git history

**Pros:**

- Cleaner monorepo history
- Simpler process
- No merge conflicts

**Cons:**

- Loses individual package history

**When to use:**

- Most migrations (default choice)
- Package has limited history value
- Simplicity preferred

**How:**
This guide uses Option A (copy files directly)

### Option B: Git Subtree Merge

**Approach**: Preserve full git history using subtree merge

**Pros:**

- Preserves complete package history
- Maintains commit attribution

**Cons:**

- More complex process
- Can clutter monorepo history
- Potential merge conflicts

**When to use:**

- Package has valuable historical context
- Attribution/blame history important

**How:**

```bash
# In monorepo root
git subtree add --prefix=packages/<category>/<name> \
    https://github.com/teamtomo/<repo-name>.git main --squash
```

Note: This guide focuses on Option A. Use Option B only if history preservation is critical. -->

## Post-Migration Checklist

Use this checklist to ensure complete migration:

```markdown
## Migration Checklist for <package-name>

### Preparation
- [ ] Original repo cloned to `/tmp/<repo-name>/`
- [ ] Category determined: [ ] io / [ ] primitives / [ ] algorithms
- [ ] Package name confirmed: _______________
- [ ] Dependencies reviewed

### File Migration
- [ ] Directory created: `packages/<category>/<name>/`
- [ ] Source copied: `src/` directory
- [ ] Tests copied: `tests/` directory
- [ ] Documentation copied: `README.md`
- [ ] License copied: `LICENSE`
- [ ] Config copied: `pyproject.toml`

### Configuration Updates
- [ ] `[tool.hatch.version]` updated with tag-pattern
- [ ] `[tool.hatch.version.raw-options]` added with search_parent_directories
- [ ] `[project.urls]` updated to monorepo URLs
- [ ] `[tool.coverage.run]` source verified (if present)
- [ ] Repository-specific files removed (not copied)

### Workspace Integration
- [ ] Ran `uv sync` successfully
- [ ] Package appears in `uv.lock`
- [ ] Root `pyproject.toml` updated (if package is default dependency)
- [ ] Metapackage `src/teamtomo/__init__.py` updated (if re-exporting)

### Verification
- [ ] Tests pass: `uv run pytest packages/<category>/<name>/tests/`
- [ ] Package builds: `uv build packages/<category>/<name>/`
- [ ] Import works: `uv run python -c "import <package>"`
- [ ] Type checking passes: `uv run mypy packages/<category>/<name>/src/`
- [ ] Coverage runs correctly

### Cleanup
- [ ] Migration documented in monorepo changelog/PR
```

<!-- ## Common Issues and Solutions

### Issue: Tests fail after migration

**Symptoms:**

```
ERROR: file not found: tests/
```

**Solutions:**

1. Check `[tool.pytest.ini_options]` in `pyproject.toml`:

   ```toml
   [tool.pytest.ini_options]
   testpaths = ["tests"]  # Should be relative to package root
   ```

2. Verify dependencies installed:

   ```bash
   uv sync
   uv run pytest packages/<category>/<name>/tests/
   ```

3. Check test imports match new structure

### Issue: Version detection fails

**Symptoms:**

```
Version not detected, using fallback: 0.0.1
```

**Solutions:**

1. Verify tag pattern in `pyproject.toml`:

   ```toml
   [tool.hatch.version]
   tag-pattern = "^<package-name>@v(?P<version>.+)$"
   ```

2. Ensure `search_parent_directories = true` and the `tag_regex` and `git_describe_command` fields match exactly:

   ```toml
   [tool.hatch.version.raw-options]
   search_parent_directories = true
   # Parse tags of the form: <package-name>@v<semver>
   tag_regex = "^torch-affine-utils@v(?P<version>\\d+\\.\\d+\\.\\d+.*)$"
   # Constrain git-describe so it only considers TeamTomo's own tags, not other workspace tags.
   # See https://github.com/ofek/hatch-vcs/issues/71
   git_describe_command = "git describe --dirty --tags --long --match 'torch-affine-utils@v[0-9]*.[0-9]*.[0-9]*'"
   ```

3. Create a test tag:

   ```bash
   git tag <package-name>@v0.0.1
   ```

4. Check fallback is set:

   ```toml
   [tool.hatch.version]
   fallback-version = "0.0.1"
   ```

### Issue: Import errors

**Symptoms:**

```python
ModuleNotFoundError: No module named '<package>'
```

**Solutions:**

1. Run `uv sync` to reinstall editable package:

   ```bash
   uv sync
   ```

2. Verify package name matches directory structure:
   - Distribution name: `torch-grid-utils` (in `pyproject.toml`)
   - Module name: `torch_grid_utils` (in `src/`)

3. Check workspace members in root `pyproject.toml`:

   ```toml
   [tool.uv.workspace]
   members = [
       "packages/primitives/*",  # Must match package location
   ]
   ```

### Issue: Coverage source incorrect

**Symptoms:**

```
Coverage.py warning: No data was collected
```

**Solution:**
Update `[tool.coverage.run]` source to match actual module name:

```toml
[tool.coverage.run]
source = ["torch_grid_utils"]  # Use underscore, not hyphen
```

### Issue: Build fails with "no files found"

**Symptoms:**

```
WARNING: No files found for package
```

**Solution:**
Check `[tool.hatch.build.targets.wheel]` configuration:

```toml
[tool.hatch.build.targets.wheel]
only-include = ["src"]
sources = ["src"]
```

## Example: torch-grid-utils Migration

### Complete pyproject.toml Diff

Here are the exact changes made to migrate `torch-grid-utils`:

**Added lines (monorepo-specific configuration):**

```diff
 [tool.hatch.version]
 source = "vcs"
+tag-pattern = "^torch-grid-utils@v(?P<version>.+)$"
+fallback-version = "0.0.1"
+
+[tool.hatch.version.raw-options]
+search_parent_directories = true
+# Parse tags of the form: <package-name>@v<semver>
+tag_regex = "^torch-affine-utils@v(?P<version>\\d+\\.\\d+\\.\\d+.*)$"
+# Constrain git-describe so it only considers TeamTomo's own tags, not other workspace tags.
+# See https://github.com/ofek/hatch-vcs/issues/71
+git_describe_command = "git describe --dirty --tags --long --match 'torch-affine-utils@v[0-9]*.[0-9]*.[0-9]*'"
```

**Changed lines (repository URLs):**

```diff
 [project.urls]
-homepage = "https://github.com/alisterburt/torch-grids"
-repository = "https://github.com/alisterburt/torch-grids"
+homepage = "https://github.com/teamtomo/teamtomo"
+repository = "https://github.com/teamtomo/teamtomo"
```

**Fixed lines (coverage configuration):**

```diff
 [tool.coverage.run]
-source = ["torch_grids"]
+source = ["torch_grid_utils"]
```

### Migration Summary

**Files kept in package:**

- `src/torch_grid_utils/` (12 Python modules, 2,228 lines)
- `tests/` (8 test files, 1,718 lines, 67 tests)
- `README.md` (package documentation)
- `LICENSE` (updated to TeamTomo copyright)
- `pyproject.toml` (modified for monorepo)

**Files removed (now at monorepo level):**

- `.github/workflows/ci.yml` → Root `.github/workflows/`
- `.gitignore` → Root `.gitignore`
- `.pre-commit-config.yaml` → Root `.pre-commit-config.yaml`
- `.copier-answers.yml` (template metadata, not needed)
- `.github/dependabot.yml` → Root `.github/dependabot.yml`

**Configuration changes:**

- 3 additions (tag-pattern, fallback-version, search_parent_directories)
- 2 URL updates (homepage, repository)
- 1 fix (coverage source)

**Verification:**

```bash
# Tests pass
uv run pytest packages/primitives/torch-grid-utils/tests/
# 67 passed

# Import works
uv run python -c "import torch_grid_utils; print(torch_grid_utils.__version__)"
# 0.0.1

# Build succeeds
uv build packages/primitives/torch-grid-utils/
# Successfully built torch_grid_utils-0.0.1.tar.gz and .whl
``` -->

---

## Next Steps

After successfully migrating a package:

1. **Archive original repo** (optional)
   - Add deprecation notice to README
   - Point to monorepo location
   - Archive repository on GitHub

2. **Update documentation**
   - Add package to monorepo README
   - Document any cross-package dependencies

3. **Set up CI/CD**
   - Ensure monorepo CI includes new package
   - Configure package-specific release workflow

4. **Communicate changes**
   - Notify users of new location
   - Update package documentation/website

For questions or issues not covered here, please open an issue on the TeamTomo monorepo.
