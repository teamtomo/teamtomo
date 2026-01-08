# TeamTomo Package Versioning policy

Versioning packages in a consistent manner is essential for long-term maintainability and dependency management.
This policy document outlines the versioning semantics used for TeamTomo and broadly draws from [Semantic Versioning 2.0.0](https://semver.org).

## Table of Contents

1. [Overview](#overview)
2. [Versioning Semantics](#versioning-semantics)
3. [Package Examples](#package-examples)
3. [Preparing for a New Version Release](#package-releases)

## Overview

Packages distributed through TeamTomo are versioned together under major and minor releases.
Adopting a coupled versioning policy helps users and developers reason about the interoperability of packages and the overall development state of Teamtomo.

All packages within the monorepo share the same major and minor version numbers (e.g., `1.2.x`), ensuring that users can rely on compatibility across the entire TeamTomo ecosystem at any given release.
Individual packages may have different patch versions to accommodate bug fixes and minor improvements without requiring a full ecosystem release.
This approach balances the benefits of coordinated releases with the flexibility to address package-specific issues independently.

## Versioning Semantics

TeamTomo draws versioning semantic definitions from SemVer:

> Given a version number MAJOR.MINOR.PATCH, increment the:
>
> * MAJOR version when you make incompatible API changes
> * MINOR version when you add functionality in a backward compatible manner
> * PATCH version when you make backward compatible bug fixes

[semver.org](https://semver.org) goes much deeper into the specifications with explanations and high-level examples for package versioning.
Below, the version numbers are discussed in the context of TeamTomo.

### MAJOR versioning number

Incrementing the major versioning number (e.g. `1.y.z -> 2.0.0`) should only be done when there are major, breaking changes or feature improvements made across TeamTomo.
This includes changes to function signatures, removal of deprecated features, or substantial architectural changes that require users to modify their code.
All packages in the monorepo receive the new major version simultaneously, even if only a subset of packages have breaking changes.

**Examples of major version changes:**

* Removing or renaming public functions/classes
* Changing required function parameters or return types
* Restructuring package namespaces

### MINOR versioning number

Minor version increments (e.g. `x.3.z -> x.4.0`) are intended for releasing new package functionalities, modifying underlying implementations (without changing user-facing API), or adjusting how TeamTomo packages depend on each other.
New features should maintain backward compatibility with the existing API.
All packages receive the minor version bump together, coordinating new features and improvements across the ecosystem.

**Examples of minor version changes:**

* Adding new functions, classes, or modules
* Adding optional parameters to existing functions
* Deprecating features (with warnings) while maintaining functionality
* Performance improvements without API changes
* Adding new packages to the monorepo

> **Note**: A new minor release _should not_ be made for any modification that matches one of the increment scenarios. Rather, modifications should be staged and grouped together so minor version releases are not made too often.

### PATCH versioning number

The patch version increment (e.g. `x.y.6 -> x.y.7`) is intended for minor bug fixes that only affect one or a few packages.
Note that the patch versions of TeamTomo packages will not always match, as individual packages can be patched independently without requiring a full ecosystem release.

**Examples of patch version changes:**

* Fixing calculation errors or edge cases
* Correcting documentation or type hints
* Resolving test failures

## Package Examples

### Example 1: Adding a new feature to `torch-grid-utils`

**Scenario:** A new coordinate transformation function is added to `torch-grid-utils`.

**Version changes:**

* All packages: `1.2.x -> 1.3.0`
* Only `torch-grid-utils` has new functionality, but all packages bump minor version
* Packages that depend on `torch-grid-utils` continue working without changes

**Reasoning:** This is a backward-compatible addition to the public API, requiring a minor version bump across the ecosystem.

### Example 2: Bug fix in `torch-fourier-filter`

**Scenario:** A numerical precision issue is discovered and fixed in `torch-fourier-filter`.

**Version changes:**

* `torch-fourier-filter`: `1.3.2 -> 1.3.3`
* Other packages: remain at their current patch version (e.g., `1.3.1`, `1.3.2`)

**Reasoning:** This is an isolated bug fix affecting only one package. Other packages don't need to be re-released.

### Example 3: Coordinated restructuring for features in TeamTomo

**Scenario:** Nearly all TeamTomo packages have had some sort of modifications, and some of the new features added to the TeamTomo necessitated restructuring how the packages interact with each other while other portions of the API were removed.

**Version changes:**

* All packages: `1.3.x -> 2.0.0`
* Release notes document the migration path

**Reasoning:** The breaking API change requires a major version bump for the entire ecosystem, ensuring users understand compatibility implications.

## Preparing for a New Version Release

Releases should be prepared on staging branches.
More documentation on the integration and release process to come...
