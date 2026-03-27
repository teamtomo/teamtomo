# Creating a new TeamTomo package

If you're looking to contribute a new package to TeamTomo, please first read through the [Contributing Guidelines](../CONTRIBUTING.md) and make sure you've set up an installation for [contributing developers](../README.md#for-contributing-developers).

## Integrating a new package overview

New code is integrated into TeamTomo through pull requests. The general workflow for integrating a package looks like:

1. Creating a new branch (on your fork)
2. Copy over the `packages/skel/` directory into one of the packages directories (primitives/algorithms/utils/wip)
3. Edit the example files and add any code/tests for the new package
4. Commit your changes on this new branch (can be multiple commits)
5. Open a pull request into the teamtomo/teamtomo repo and request a review from the Core Developers group

We will then help work through the pull request process making sure to fix any failing unit tests, suggesting changes to keep the package in-line with community standards, and eventually merge the new package into the repository.

## Creating a new package from the example skeleton

The `packages/skel/` directory contains the components necessary for a package in teamtomo, namely:

- `src/your_package_name/`: source code for the package
- `tests/`: unit tests for the package
- `LICENSE`: license file for the package (BSD 3-Clause License)
- `README.md`: readme file for the package
- `pyproject.toml`: configuration file for the package

All of these files, except for the license, need to be edited to reflect the new package name, content, and dependencies.

### Copying over the skeleton

First, navigate to the root of the TeamTomo repository. Then, copy the `packages/skel/` into one of `packages/{primitives/algorithms/utils/wip}` directories. For example

```bash
cp -r packages/skel/ packages/primitives/my_new_package
```

### Editing the source directory name

The workspace layout requires the src directory to match the name of the package. Following the same example as above, change the name of `src/your_package_name/` to `src/my_new_package/`.

```bash
cd packages/primitives/my_new_package
mv src/your_package_name src/my_new_package
```

### Editing the README.md

Update the `README.md` file to reflect the new package name and description. For now, this can be quite simple. We may ask you to be more descriptive as the pull request process continues.

### Editing the pyproject.toml

This is the most important file to edit since it defines the structure of the new package. Make sure to update any instances of `your-package-name` to the new package name (in kebab-case), and `your_package_name` to the new package name (in snake_case).

The project description and dependencies also need updated as well as any package dependencies (under the `[project]` block).

## Updating root pyproject.toml

To make sure your new package is added to the uv workspace, navigate back to the root of the repository and edit the `pyproject.toml` file.

Add your package name (in kebab-case) to the project dependencies

```toml
[project]
...
dependencies = [
    ...
    "your-package-name",
]

And then add you package to the uv sources as a workspace member.

```toml
[tool.uv.sources]
torch-grid-utils = { workspace = true }  # pre-existing
...
your-package-name = { workspace = true }
```

## Check that the monorepo builds

Re-sync the repository to make sure your package builds correctly and all tests pass.

```bash
uv sync --all-extras --all-packages
```

```bash
uv run pytest
```

## Opening a pull request
