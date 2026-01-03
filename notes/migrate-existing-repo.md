# Migrating Existing TeamTomo Repositories to Monorepo

This guide outlines the process for migrating existing standalone TeamTomo GitHub repositories into the monorepo workspace structure.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Migration Process](#step-by-step-migration-process)
4. [Before vs After Comparison](#before-vs-after-comparison)
5. [Configuration Changes](#configuration-changes)
6. [CI/CD Considerations](#cicd-considerations)
7. [Git History Options](#git-history-options)
8. [Post-Migration Checklist](#post-migration-checklist)
9. [Common Issues and Solutions](#common-issues-and-solutions)
10. [Example: torch-grid-utils Migration](#example-torch-grid-utils-migration)

## Overview

### Purpose
Consolidate existing standalone TeamTomo packages into a unified monorepo workspace for better maintainability, shared tooling, and easier cross-package development.

### Benefits
- **Unified versioning**: Coordinated releases across packages
- **Shared tooling**: Single set of dev tools, linters, and CI/CD
- **Easier development**: Cross-package changes in single PR
- **Simplified dependencies**: Workspace packages automatically resolved
- **Consistent standards**: Shared configuration and best practices

### Reference Example
This guide uses the `torch-grid-utils` migration as a reference. The original repository at `teamtomo/torch-grid-utils` was successfully migrated to `packages/primitives/torch-grid-utils/` in the monorepo.

## Prerequisites

### Before Starting

1. **Identify the repository**
   - GitHub organization/repo name (e.g., `teamtomo/torch-grid-utils`)
   - Current version/release status
   - Active development status

2. **Determine category**
   - `io/`: Data I/O, file format handling, geometry utilities
   - `primitives/`: Core data structures, types, arrays
   - `algorithms/`: Processing algorithms, analysis tools

3. **Package naming**
   - Confirm package name (usually same as repo name)
   - Check for naming conflicts with existing packages

4. **Review dependencies**
   - List external dependencies
   - Identify any dependencies on other TeamTomo packages

## Step-by-Step Migration Process

### Phase 1: Preparation

**1. Clone the original repository**

```bash
cd /tmp
git clone https://github.com/teamtomo/<repo-name>.git
cd <repo-name>
```

**IMPORTANT**: We'll use `git subtree add --squash` which preserves history while keeping it clean.

**2. Review current structure**

```bash
# Check directory structure
tree -L 2

# Review pyproject.toml
cat pyproject.toml

# Check for special config files
ls -la | grep "^\."
```

**3. Identify files to migrate**

Files to **COPY**:
- `src/` - Entire source directory
- `tests/` - All test files
- `README.md` - Package documentation
- `LICENSE` - License file

Files to **SKIP** (not needed in monorepo):
- `.github/` - CI/CD workflows (monorepo handles this)
- `.gitignore` - Use monorepo's root .gitignore
- `.pre-commit-config.yaml` - Use monorepo's root config
- `.copier-answers.yml` - Template metadata
- Any other repo-specific config files

### Phase 2: Directory Setup

**1. Create package directory in monorepo**

```bash
cd /path/to/teamtomo  # Your monorepo root
mkdir -p packages/<category>/<package-name>
```

**Example:**
```bash
mkdir -p packages/primitives/torch-grid-utils
```

**2. Copy source files**

```bash
# Copy from original repo to monorepo
cp -r /tmp/<repo-name>/src packages/<category>/<package-name>/
cp -r /tmp/<repo-name>/tests packages/<category>/<package-name>/
cp /tmp/<repo-name>/README.md packages/<category>/<package-name>/
cp /tmp/<repo-name>/LICENSE packages/<category>/<package-name>/
cp /tmp/<repo-name>/pyproject.toml packages/<category>/<package-name>/
```

### Phase 3: Clean Up and Configure Package

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
```

**2. Update LICENSE to TeamTomo copyright**

```bash
# Copy root LICENSE to standardize copyright
cp ../../../LICENSE ./LICENSE
```

**3. Modify `pyproject.toml`**

Edit `packages/<category>/<package-name>/pyproject.toml`:

**Add/Update `[tool.hatch.version]` section:**

```toml
[tool.hatch.version]
source = "vcs"
tag-pattern = "^<package-name>@v(?P<version>.+)$"
fallback-version = "0.0.1"
```

**Add `[tool.hatch.version.raw-options]` section:**

```toml
[tool.hatch.version.raw-options]
search_parent_directories = true  # Find .git in monorepo root
```

**Update `[project.urls]`:**

```toml
[project.urls]
homepage = "https://github.com/teamtomo/teamtomo"
repository = "https://github.com/teamtomo/teamtomo"
```

**Verify coverage source (if present):**

Make sure `[tool.coverage.run]` has the correct source:

```toml
[tool.coverage.run]
source = ["<actual_package_name>"]  # e.g., "torch_grid_utils"
```

**4. Review and verify configuration**

Check that:
- Package name matches directory structure
- Dependencies are correct
- Python version requirement is compatible (>=3.12 for monorepo)
- Build system uses hatchling
- All monorepo-level configs removed (.github/, .gitignore, .pre-commit-config.yaml)
- LICENSE updated to TeamTomo copyright

### Phase 4: Workspace Integration

**1. Register package with workspace**

```bash
cd /path/to/teamtomo
uv sync
```

This will:
- Discover the new package via glob patterns in root `pyproject.toml`
- Install it as an editable package
- Update `uv.lock`

**2. Update root `pyproject.toml` (if needed)**

Only if the package should be a default dependency of the metapackage:

```toml
[project]
dependencies = [
    "torch-grid-utils",  # Add your package
]
```

**3. Update metapackage exports (optional)**

If the package should be re-exported through the main `teamtomo` namespace, edit `src/teamtomo/__init__.py`:

```python
# Re-export from subpackage
try:
    from <package_module> import function1, function2
except ImportError:
    pass  # Package not installed

__all__ = [
    "function1",
    "function2",
    # ... other exports
]
```

### Phase 5: Verification

**1. Run tests**

```bash
# Test the migrated package
uv run pytest packages/<category>/<package-name>/tests/

# With coverage
uv run pytest packages/<category>/<package-name>/tests/ \
    --cov=<package_module> --cov-report=term-missing
```

**2. Verify imports**

```bash
# Check import works
uv run python -c "import <package_module>; print(<package_module>.__version__)"

# Test specific functions
uv run python -c "from <package_module> import <function>; print(<function>)"
```

**3. Build package**

```bash
uv build packages/<category>/<package-name>/
```

Check that `dist/` contains `.whl` and `.tar.gz` files.

**4. Type checking**

```bash
uv run mypy packages/<category>/<package-name>/src/
```

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

## Configuration Changes

### Version Configuration

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

Note: This guide focuses on Option A. Use Option B only if history preservation is critical.

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
- [ ] Original repo archived or marked as deprecated on GitHub
- [ ] README updated with deprecation notice (point to monorepo)
- [ ] Migration documented in monorepo changelog/PR
```

## Common Issues and Solutions

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

2. Ensure `search_parent_directories = true`:
   ```toml
   [tool.hatch.version.raw-options]
   search_parent_directories = true
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
```

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
